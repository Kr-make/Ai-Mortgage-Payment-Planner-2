# 🏛️ MortgageIQ — Loan Intelligence & Payoff Platform

An AI-powered mortgage analytics, cash flow planning, and loan payoff optimization platform.

---

## ✨ Features Added & Updated

### 1. 🛡️ Groq AI Chatbot with Strict Mortgage Domain Guardrail
- **Domain Guard Enforcement**: The assistant is strictly restricted to answering questions related to mortgages, home loans, EMI calculations, credit score optimization, and loan repayment strategies.
- **Off-Topic Protection**: Any question outside this domain (e.g., cooking, programming, movies, sports, weather, politics, general trivia) is politely declined with a standard refusal message and guidance back to mortgage topics.
- **Intelligent Fallback Engine**: If the Groq API key is expired or missing, built-in offline mortgage intelligence automatically handles common queries without crashing.
- **Interactive UI**: Includes popular clickable mortgage questions, Groq API key override settings, and a chat history reset button.

### 2. 📊 4 Interactive Visual Analytics Graphs
- **1. Debt Burden Gauge (DTI Ratio)**: Real-time gauge reflecting EMI-to-income ratio (Safe <30%, Moderate 30–50%, High Risk >50%).
- **2. Monthly Income Spend & Cash Flow Allocation (Donut Chart)**: Visually breaks down monthly income into Mortgage EMI, Existing Debt Commitments, and Disposable Living Cushion with center DTI spend percentage.
- **3. Yearly Amortization Schedule (Stacked Bar Chart)**: Year-by-year distribution of Principal vs. Interest paid over the loan lifetime.
- **4. Outstanding Loan Balance & Paydown Trajectory (Area / Line Chart)**: Side-by-side comparison of standard amortization vs. accelerated payoff curve showing exact remaining balance year by year and time saved.

### 3. 💡 AI Smart Repayment & Early Payoff Optimizer (Suggestions Section)
- **Credit Score Optimization Strategy**:
  - Dynamically calculates interest rate discounts (0.50% – 0.85% drop) when improving credit score from current score to 780+.
  - Calculates the exact **number of months / years saved earlier** (e.g., "Pay off 28 months earlier!") and **total interest saved in Lakhs**.
  - Actionable steps to boost CIBIL/Experian credit score by 50–80 points.
- **Extra Monthly Micro-Prepayment Strategy**:
  - Simulates the payoff impact of adding ₹2,000 to ₹10,000/month directly to principal.
- **13th EMI / Annual Bonus Strategy**:
  - Simulates the compounding effect of paying 1 additional EMI every year using annual bonuses or tax refunds.
- **Combined Power Plan**:
  - Combines credit score rate discounts with extra prepayments for maximum interest relief.
- **Dynamic Strategy Comparison Matrix**: Side-by-side summary table comparing new tenures, months saved, and interest saved across all strategies.

### 4. 📥 Downloadable Schedule Section (Income Spend & Remaining Payoff)
- **KPI Metrics**: Monthly Income, Total Monthly Debt Spend, Disposable Income Left, and Total Remaining Amount to Pay.
- **One-Click Export**: `📥 Download Detailed Schedule (CSV)` downloads `mortgage_monthly_income_spend_and_payoff_schedule.csv` (UTF-8-SIG formatted for Microsoft Excel compatibility).
- **Comprehensive Columns**:
  - `Month` & `Year`
  - `Monthly Income (₹)`
  - `Mortgage EMI (₹)`
  - `Existing Debt (₹)`
  - `Total Monthly Spend (₹)`
  - `Monthly Spend (% of Income)`
  - `Disposable Income (₹)`
  - `Principal Paid (₹)`
  - `Interest Paid (₹)`
  - `Cumulative Paid (₹)`
  - `Remaining Principal (₹)`
  - `Total More Amount to Pay (₹)`
- **Interactive Preview Table**: Expandable data viewer showing the first 24 months directly in the dashboard.

---

## 🚀 How to Run the App

1. Open a terminal in the project directory:
   ```powershell
   cd "F:\download_F\krish_project_final (2)\krish_project"
   ```

2. Run the Streamlit application:
   ```powershell
   streamlit run app.py
   ```

3. Open your browser at `http://localhost:8501`.
