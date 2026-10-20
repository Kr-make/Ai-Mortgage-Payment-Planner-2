import os
import logging
import warnings
import joblib
import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    data_path: str = "loan_data.csv"
    model_dir: str = "models"
    model_name: str = "risk_model.pkl"
    artifact_dir: str = "artifacts"
    target_column: str = "risk"
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    n_estimators: int = 300
    max_depth: int = 8
    min_samples_leaf: int = 10
    class_weight: str = "balanced"
    calibrate: bool = True
    feature_selection: bool = True
    feature_selection_threshold: str = "median"
    ensemble: bool = True
    drop_threshold: float = 0.5
    categorical_columns: List[str] = field(default_factory=list)
    numeric_columns: List[str] = field(default_factory=list)

    @property
    def model_path(self) -> Path:
        return Path(self.model_dir) / self.model_name

    @property
    def artifact_path(self) -> Path:
        return Path(self.artifact_dir)


class DataValidator:
    def __init__(self, config: Config):
        self.config = config

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Dataset is empty.")
        if self.config.target_column not in df.columns:
            raise KeyError(f"Target column '{self.config.target_column}' not found.")

        null_ratio = df.isnull().mean()
        high_null = null_ratio[null_ratio > self.config.drop_threshold].index.tolist()
        if high_null:
            logger.warning("Dropping high-null columns: %s", high_null)
            df = df.drop(columns=high_null)

        dupes = df.duplicated().sum()
        if dupes:
            logger.warning("Removing %d duplicate rows.", dupes)
            df = df.drop_duplicates()

        return df


class FeatureEngineer:
    def __init__(self, config: Config):
        self.config = config
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._encode_categoricals(df, fit=True)
        df = self._impute(df)
        self._fitted = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise NotFittedError("FeatureEngineer must be fitted before transform.")
        df = df.copy()
        df = self._encode_categoricals(df, fit=False)
        df = self._impute(df)
        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        cat_cols = (
            self.config.categorical_columns
            or df.select_dtypes(include="object").columns.tolist()
        )
        for col in cat_cols:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le:
                    df[col] = (
                        df[col]
                        .astype(str)
                        .map(
                            lambda x, le=le: (
                                le.transform([x])[0] if x in le.classes_ else -1
                            )
                        )
                    )
        return df

    def _impute(self, df: pd.DataFrame) -> pd.DataFrame:
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        return df


class ModelFactory:
    def __init__(self, config: Config):
        self.config = config

    def build_random_forest(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            n_jobs=-1,
        )

    def build_gradient_boosting(self) -> GradientBoostingClassifier:
        return GradientBoostingClassifier(
            n_estimators=self.config.n_estimators // 2,
            max_depth=self.config.max_depth - 2,
            random_state=self.config.random_state,
        )

    def build_ensemble(self) -> VotingClassifier:
        return VotingClassifier(
            estimators=[
                ("rf", self.build_random_forest()),
                ("gb", self.build_gradient_boosting()),
            ],
            voting="soft",
            n_jobs=-1,
        )

    def build(self) -> Any:
        base = (
            self.build_ensemble()
            if self.config.ensemble
            else self.build_random_forest()
        )
        if self.config.calibrate and not self.config.ensemble:
            return CalibratedClassifierCV(base, cv=3, method="isotonic")
        return base


class Evaluator:
    def __init__(self, config: Config):
        self.config = config

    def evaluate(
        self, model, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, Any]:
        preds = model.predict(X_test)
        proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        metrics = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "f1_weighted": round(f1_score(y_test, preds, average="weighted"), 4),
            "f1_macro": round(f1_score(y_test, preds, average="macro"), 4),
        }
        if proba is not None and len(np.unique(y_test)) == 2:
            metrics["roc_auc"] = round(roc_auc_score(y_test, proba), 4)

        metrics["classification_report"] = classification_report(y_test, preds)
        metrics["confusion_matrix"] = confusion_matrix(y_test, preds).tolist()
        return metrics

    def cross_validate(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        return {
            "cv_roc_auc_mean": round(scores.mean(), 4),
            "cv_roc_auc_std": round(scores.std(), 4),
        }


class ArtifactManager:
    def __init__(self, config: Config):
        self.config = config
        self.config.artifact_path.mkdir(parents=True, exist_ok=True)
        Path(config.model_dir).mkdir(parents=True, exist_ok=True)

    def save_model(
        self, model, feature_engineer: FeatureEngineer, feature_names: List[str]
    ) -> Path:
        bundle = {
            "model": model,
            "feature_engineer": feature_engineer,
            "feature_names": feature_names,
            "trained_at": datetime.utcnow().isoformat(),
            "config": self.config,
        }
        joblib.dump(bundle, self.config.model_path)
        logger.info("Model saved → %s", self.config.model_path)
        return self.config.model_path

    def save_metrics(self, metrics: Dict[str, Any]) -> Path:
        import json

        path = self.config.artifact_path / "metrics.json"
        serializable = {
            k: v for k, v in metrics.items() if k != "classification_report"
        }
        serializable["classification_report"] = metrics.get("classification_report", "")
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info("Metrics saved → %s", path)
        return path

    def save_feature_importance(
        self, model, feature_names: List[str]
    ) -> Optional[Path]:
        estimator = model
        if hasattr(model, "estimators_"):
            estimator = (
                model.estimators_[0][1]
                if isinstance(model.estimators_[0], tuple)
                else model.estimators_[0]
            )
        if not hasattr(estimator, "feature_importances_"):
            return None
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": estimator.feature_importances_,
            }
        ).sort_values("importance", ascending=False).to_csv(
            self.config.artifact_path / "feature_importance.csv", index=False
        )
        logger.info("Feature importance saved.")
        return self.config.artifact_path / "feature_importance.csv"

    @staticmethod
    def load(model_path: str) -> Dict[str, Any]:
        bundle = joblib.load(model_path)
        logger.info("Bundle loaded from %s", model_path)
        return bundle


class MortgagePipeline:
    def __init__(self, config: Config):
        self.config = config
        self.validator = DataValidator(config)
        self.feature_engineer = FeatureEngineer(config)
        self.model_factory = ModelFactory(config)
        self.evaluator = Evaluator(config)
        self.artifact_manager = ArtifactManager(config)
        self.model = None
        self.feature_names: Optional[List[str]] = None

    def _load_data(self) -> pd.DataFrame:
        path = Path(self.config.data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        df = pd.read_csv(path)
        logger.info("Loaded %d rows × %d cols from %s", *df.shape, path)
        return df

    def _split(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            stratify=y,
            random_state=self.config.random_state,
        )
        logger.info("Train: %d | Test: %d", len(X_train), len(X_test))
        return X_train, X_test, y_train, y_test

    def _apply_feature_selection(self, X_train, y_train, X_test):
        logger.info("Running feature selection...")
        selector_model = RandomForestClassifier(
            n_estimators=100, random_state=self.config.random_state, n_jobs=-1
        )
        selector_model.fit(X_train, y_train)
        selector = SelectFromModel(
            selector_model,
            threshold=self.config.feature_selection_threshold,
            prefit=True,
        )
        X_train_sel = selector.transform(X_train)
        X_test_sel = selector.transform(X_test)
        selected = [f for f, s in zip(X_train.columns, selector.get_support()) if s]
        logger.info("Selected %d / %d features.", len(selected), len(X_train.columns))
        return (
            pd.DataFrame(X_train_sel, columns=selected),
            pd.DataFrame(X_test_sel, columns=selected),
            selected,
        )

    def train(self) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("Starting Mortgage Risk Pipeline")
        logger.info("=" * 60)

        df = self._load_data()
        df = self.validator.validate(df)

        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column]
        X = self.feature_engineer.fit_transform(X)

        X_train, X_test, y_train, y_test = self._split(X, y)

        if self.config.feature_selection:
            X_train, X_test, selected = self._apply_feature_selection(
                X_train, y_train, X_test
            )
        else:
            selected = list(X_train.columns)

        self.feature_names = selected
        self.model = self.model_factory.build()

        logger.info("Running cross-validation...")
        cv_metrics = self.evaluator.cross_validate(self.model, X_train, y_train)
        logger.info(
            "CV ROC-AUC: %.4f ± %.4f",
            cv_metrics["cv_roc_auc_mean"],
            cv_metrics["cv_roc_auc_std"],
        )

        logger.info("Fitting final model...")
        self.model.fit(X_train, y_train)

        test_metrics = self.evaluator.evaluate(self.model, X_test, y_test)
        logger.info(
            "Accuracy: %.4f | ROC-AUC: %.4f | F1: %.4f",
            test_metrics["accuracy"],
            test_metrics.get("roc_auc", 0),
            test_metrics["f1_macro"],
        )
        logger.info("\n%s", test_metrics["classification_report"])

        model_path = self.artifact_manager.save_model(
            self.model, self.feature_engineer, self.feature_names
        )
        self.artifact_manager.save_metrics({**cv_metrics, **test_metrics})
        self.artifact_manager.save_feature_importance(self.model, self.feature_names)

        return {
            **cv_metrics,
            **{
                k: v
                for k, v in test_metrics.items()
                if k not in ("classification_report", "confusion_matrix")
            },
            "model_path": str(model_path),
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise NotFittedError("Pipeline has not been trained.")
        X = self.feature_engineer.transform(df)[self.feature_names]
        return self.model.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise NotFittedError("Pipeline has not been trained.")
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("Model does not support probability estimates.")
        X = self.feature_engineer.transform(df)[self.feature_names]
        return self.model.predict_proba(X)

    @classmethod
    def load(cls, model_path: str) -> "MortgagePipeline":
        bundle = ArtifactManager.load(model_path)
        pipeline = cls(bundle["config"])
        pipeline.model = bundle["model"]
        pipeline.feature_engineer = bundle["feature_engineer"]
        pipeline.feature_names = bundle["feature_names"]
        return pipeline


def train_pipeline(config: Optional[Config] = None) -> Dict[str, Any]:
    config = config or Config()
    pipeline = MortgagePipeline(config)
    return pipeline.train()


if __name__ == "__main__":
    result = train_pipeline()
    logger.info("Pipeline complete: %s", result)
