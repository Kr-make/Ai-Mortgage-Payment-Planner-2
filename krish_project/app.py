import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Only initialise Groq client if key is present — avoids crash on startup
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="MortgageIQ — Loan Intelligence Platform",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# STYLES
# =========================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    letter-spacing: -0.02em;
    line-height: 1.1;
    color: #0f1923;
    margin-bottom: 0.2rem;
}

.subtitle {
    font-size: 1rem;
    color: #64748b;
    font-weight: 300;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    border-left: 4px solid #1e40af;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.3rem;
}

.kpi-value {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    color: #0f172a;
    line-height: 1;
}

.kpi-sub {
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.3rem;
}

.risk-safe {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border-left-color: #059669;
    border-color: #a7f3d0;
}

.risk-warning {
    background: linear-gradient(135deg, #fffbeb, #fef3c7);
    border-left-color: #d97706;
    border-color: #fde68a;
}

.risk-danger {
    background: linear-gradient(135deg, #fff1f2, #ffe4e6);
    border-left-color: #dc2626;
    border-color: #fecdd3;
}

.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #0f172a;
    margin-bottom: 0.2rem;
    margin-top: 2rem;
}

.section-sub {
    font-size: 0.82rem;
    color: #94a3b8;
    margin-bottom: 1rem;
    font-weight: 300;
}

.stButton > button {
    background: #1e40af;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 2rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.05em;
    width: 100%;
    transition: all 0.2s;
}

.stButton > button:hover {
    background: #1e3a8a;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(30,64,175,0.3);
}

.insight-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    font-size: 0.88rem;
    color: #334155;
    line-height: 1.6;
}

div[data-testid="stSidebar"] {
    background: #0f172a;
}

div[data-testid="stSidebar"] * {
    color: #cbd5e1 !important;
}

div[data-testid="stSidebar"] label {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

.sidebar-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: #f1f5f9 !important;
    margin-bottom: 0.2rem;
}

.sidebar-sub {
    font-size: 0.75rem;
    color: #64748b !important;
    margin-bottom: 1.5rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# HELPER FUNCTIONS
# =========================


def calculate_emi(principal: float, annual_rate: float, years: int) -> float:
    """Calculate Equated Monthly Instalment (EMI)."""
    r = annual_rate / (12 * 100)
    n = years * 12
    if r == 0:
        return principal / n if n > 0 else 0.0
    return (principal * r * (1 + r) ** n) / ((1 + r) ** n - 1)


def risk_gauge(ratio: float) -> go.Figure:
    """Return a Plotly gauge chart for EMI/Income ratio."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=round(ratio * 100, 1),
            number={"suffix": "%", "font": {"size": 32, "family": "DM Serif Display"}},
            delta={"reference": 30, "valueformat": ".1f", "suffix": "%"},
            title={
                "text": "EMI / Income Ratio",
                "font": {"size": 13, "color": "#64748b", "family": "DM Sans"},
            },
            gauge={
                "axis": {"range": [0, 80], "ticksuffix": "%"},
                "bar": {"color": "#1e40af"},
                "steps": [
                    {"range": [0, 30], "color": "#d1fae5"},
                    {"range": [30, 50], "color": "#fef3c7"},
                    {"range": [50, 80], "color": "#ffe4e6"},
                ],
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans"},
    )
    return fig


def amortization_chart(
    principal: float, annual_rate: float, years: int, emi: float
) -> go.Figure:
    """Return a stacked-bar chart showing yearly principal vs interest paid."""
    r = annual_rate / (12 * 100)
    balance = principal
    yearly_principal = []
    yearly_interest = []

    for _ in range(years):
        yr_p = yr_i = 0.0
        for _ in range(12):
            interest = balance * r
            p_part = emi - interest
            yr_i += interest
            yr_p += p_part
            balance = max(balance - p_part, 0)
        yearly_principal.append(yr_p)
        yearly_interest.append(yr_i)

    x_labels = [f"Y{i+1}" for i in range(years)]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="Principal", x=x_labels, y=yearly_principal, marker_color="#1e40af")
    )
    fig.add_trace(
        go.Bar(name="Interest", x=x_labels, y=yearly_interest, marker_color="#93c5fd")
    )

    fig.update_layout(
        barmode="stack",
        height=300,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        font={"family": "DM Sans"},
    )
    return fig


# =========================
# AI CHATBOT PAGE
# =========================


def chatbot_page():
    st.markdown(
        '<div class="main-title">Mortgage AI Assistant</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">Ask anything about loans, EMI, mortgages, risk, finance, or banking</div>',
        unsafe_allow_html=True,
    )

    if not GROQ_API_KEY or client is None:
        st.error(
            "⚠️ GROQ_API_KEY not found. Please add it to your `.env` file and restart the app."
        )
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask your question...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a professional mortgage and finance assistant. "
                                "Help users understand loans, EMI, credit scores, mortgage risks, "
                                "refinancing, banking terms, financial planning, and interest calculations. "
                                "Be concise, accurate, and friendly."
                            ),
                        },
                        *[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ],
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                    stream=True,
                )

                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

            except Exception as e:
                full_response = f"❌ Error communicating with AI: {e}"
                response_placeholder.error(full_response)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )


# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.markdown('<div class="sidebar-title">MortgageIQ</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-sub">Loan Intelligence Platform</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    page = st.radio("Navigation", ["Loan Dashboard", "AI Chatbot"])

    st.markdown("---")

    income = st.number_input(
        "Monthly Income (₹)",
        min_value=10_000,
        max_value=10_000_000,
        value=100_000,
        step=5_000,
    )
    loan_amount = st.number_input(
        "Loan Amount (₹)",
        min_value=100_000,
        max_value=50_000_000,
        value=3_000_000,
        step=100_000,
    )
    interest_rate = st.number_input(
        "Interest Rate (%)", min_value=1.0, max_value=20.0, value=8.5, step=0.1
    )
    tenure = st.number_input(
        "Tenure (Years)", min_value=1, max_value=30, value=20, step=1
    )
    existing_emi = st.number_input(
        "Existing EMIs / Month (₹)", min_value=0, value=0, step=1_000
    )

    analyze = st.button("⚡ Analyze Loan")

# =========================
# CHATBOT PAGE ROUTE
# =========================

if page == "AI Chatbot":
    chatbot_page()
    st.stop()

# =========================
# LOAN DASHBOARD
# =========================

st.markdown(
    '<div class="main-title">Loan Intelligence Dashboard</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">AI-powered mortgage risk & affordability analysis</div>',
    unsafe_allow_html=True,
)

if analyze:
    # ---- Core calculations ----
    emi = calculate_emi(loan_amount, interest_rate, int(tenure))
    total_emi_burden = emi + existing_emi  # include existing obligations
    total_payment = emi * tenure * 12
    total_interest = total_payment - loan_amount
    emi_ratio = total_emi_burden / income if income > 0 else 0

    # ---- Risk classification ----
    if emi_ratio <= 0.30:
        risk_css, risk_label, risk_icon = "risk-safe", "SAFE", "✅"
    elif emi_ratio <= 0.50:
        risk_css, risk_label, risk_icon = "risk-warning", "MODERATE", "⚠️"
    else:
        risk_css, risk_label, risk_icon = "risk-danger", "HIGH RISK", "🚨"

    # ---- KPI Cards ----
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">Monthly EMI</div>
            <div class="kpi-value">₹{emi:,.0f}</div>
            <div class="kpi-sub">Per Month</div>
        </div>""",
            unsafe_allow_html=True,
        )

    with k2:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Interest</div>
            <div class="kpi-value">₹{total_interest / 1e5:.1f}L</div>
            <div class="kpi-sub">Over {int(tenure)} years</div>
        </div>""",
            unsafe_allow_html=True,
        )

    with k3:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Payment</div>
            <div class="kpi-value">₹{total_payment / 1e5:.1f}L</div>
            <div class="kpi-sub">Principal + Interest</div>
        </div>""",
            unsafe_allow_html=True,
        )

    with k4:
        st.markdown(
            f"""
        <div class="kpi-card {risk_css}">
            <div class="kpi-label">Risk Level</div>
            <div class="kpi-value">{risk_icon}</div>
            <div class="kpi-sub">{risk_label}</div>
        </div>""",
            unsafe_allow_html=True,
        )

    # ---- Charts ----
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown(
            '<div class="section-header">Risk Gauge</div>', unsafe_allow_html=True
        )
        st.plotly_chart(risk_gauge(emi_ratio), use_container_width=True)

    with col_right:
        st.markdown(
            '<div class="section-header">Amortisation Schedule</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            amortization_chart(loan_amount, interest_rate, int(tenure), emi),
            use_container_width=True,
        )

    # ---- AI Insights ----
    st.markdown('<div class="section-header">AI Insights</div>', unsafe_allow_html=True)

    insights = []

    if emi_ratio <= 0.30:
        insights.append(
            "✅ Your EMI-to-income ratio is healthy — well within the safe 30% threshold."
        )
    elif emi_ratio <= 0.50:
        insights.append(
            "⚠️ Your EMI burden is moderate (30–50% of income). Keep discretionary expenses low."
        )
    else:
        insights.append(
            "🚨 Your EMI burden exceeds 50% of income — this is high risk. "
            "Consider a smaller loan, longer tenure, or increasing income before proceeding."
        )

    if interest_rate > 10:
        insights.append(
            "📉 Your interest rate is above 10%. Explore balance-transfer options or negotiate "
            "with lenders — refinancing could save lakhs over the loan tenure."
        )

    if existing_emi > 0:
        total_obligation_pct = (total_emi_burden / income) * 100
        insights.append(
            f"💳 Existing EMIs of ₹{existing_emi:,}/month are included in the risk calculation. "
            f"Total monthly debt obligation is ₹{total_emi_burden:,.0f} "
            f"({total_obligation_pct:.1f}% of income)."
        )

    if total_interest > loan_amount:
        insights.append(
            f"📊 You will pay more in interest (₹{total_interest/1e5:.1f}L) than the original "
            f"principal (₹{loan_amount/1e5:.1f}L). Making part-prepayments early can significantly "
            "reduce your interest outgo."
        )

    for insight in insights:
        st.markdown(
            f'<div class="insight-box">{insight}</div>',
            unsafe_allow_html=True,
        )

else:
    st.markdown(
        """
    <div style="text-align:center; padding:5rem 2rem; color:#94a3b8;">
        <div style="font-size:4rem; margin-bottom:1rem;">🏛️</div>
        <div style="
            font-family:'DM Serif Display', serif;
            font-size:1.8rem;
            color:#64748b;
            margin-bottom:0.5rem;
        ">Enter your loan details</div>
        <div style="font-size:0.95rem;">
            Fill in the sidebar and click <strong>Analyze Loan</strong>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
