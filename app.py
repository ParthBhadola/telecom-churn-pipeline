import streamlit as st
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.predict import predict_customer
from src.impact import calculate_impact

st.set_page_config(page_title="ChurnIQ", page_icon="📡", layout="wide")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding: 2rem 3rem;
        max-width: 1100px;
    }

    .app-header {
        border-bottom: 1px solid #2a2d35;
        padding-bottom: 1.2rem;
        margin-bottom: 2rem;
    }

    .app-title {
        font-size: 1.6rem;
        font-weight: 600;
        color: #e8eaf0;
        margin: 0;
        letter-spacing: -0.3px;
    }

    .app-sub {
        font-size: 0.88rem;
        color: #6b7280;
        margin-top: 0.3rem;
    }

    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 1rem;
    }

    .result-card {
        background: #161b22;
        border: 1px solid #2a2d35;
        border-radius: 10px;
        padding: 1.4rem 1.8rem;
        margin-top: 1.5rem;
    }

    .result-label {
        font-size: 0.75rem;
        color: #6b7280;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .result-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.2rem 0 0;
        color: #e8eaf0;
    }

    .risk-high { color: #f87171; }
    .risk-medium { color: #fbbf24; }
    .risk-low { color: #34d399; }

    .alert-churn {
        background: #1f1a1a;
        border-left: 3px solid #f87171;
        padding: 0.9rem 1.2rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.88rem;
        color: #fca5a5;
        margin-top: 1rem;
    }

    .alert-safe {
        background: #141f1a;
        border-left: 3px solid #34d399;
        padding: 0.9rem 1.2rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.88rem;
        color: #6ee7b7;
        margin-top: 1rem;
    }

    .impact-card {
        background: #161b22;
        border: 1px solid #2a2d35;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
    }

    .impact-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: #e8eaf0;
        margin: 0;
    }

    .impact-label {
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0 0 4px;
    }

    div[data-testid="stButton"] button {
        background: #1e3a5f !important;
        color: #93c5fd !important;
        border: 1px solid #2d5a8e !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stButton"] button:hover {
        background: #2d5a8e !important;
        border-color: #3d7ab8 !important;
        color: #bfdbfe !important;
    }

    div[data-testid="stTabs"] button {
        font-size: 0.88rem;
        color: #6b7280;
        font-weight: 500;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #e8eaf0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <p class="app-title">📡 ChurnIQ</p>
    <p class="app-sub">Customer churn intelligence — predict, explain and quantify retention value</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Customer Prediction", "Financial Impact"])

with tab1:
    st.markdown("<p class='section-label'>Customer Details</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1], gap="large")

    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=1.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly_charges * tenure), step=10.0)

    with col2:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

    with col3:
        payment = st.selectbox("Payment Method", [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("Run Prediction", use_container_width=True)

    if predict_btn:
        customer = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Contract_Month-to-month": 1 if contract == "Month-to-month" else 0,
            "Contract_One year": 1 if contract == "One year" else 0,
            "Contract_Two year": 1 if contract == "Two year" else 0,
            "InternetService_Fiber optic": 1 if internet == "Fiber optic" else 0,
            "InternetService_DSL": 1 if internet == "DSL" else 0,
            "InternetService_No": 1 if internet == "No" else 0,
            "PaymentMethod_Electronic check": 1 if payment == "Electronic check" else 0,
            "PaymentMethod_Mailed check": 1 if payment == "Mailed check" else 0,
            "PaymentMethod_Bank transfer (automatic)": 1 if payment == "Bank transfer (automatic)" else 0,
            "PaymentMethod_Credit card (automatic)": 1 if payment == "Credit card (automatic)" else 0,
        }

        result = predict_customer(customer)
        prob_pct = result["churn_probability"] * 100
        risk = result["risk_level"]
        risk_class = f"risk-{risk.lower()}"

        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown(f"""
                <p class='result-label'>Churn Probability</p>
                <p class='result-value'>{prob_pct:.1f}%</p>
            """, unsafe_allow_html=True)

        with r2:
            verdict = "Will Churn" if result["will_churn"] else "Will Stay"
            st.markdown(f"""
                <p class='result-label'>Prediction</p>
                <p class='result-value'>{verdict}</p>
            """, unsafe_allow_html=True)

        with r3:
            st.markdown(f"""
                <p class='result-label'>Risk Level</p>
                <p class='result-value {risk_class}'>{risk}</p>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        if result["will_churn"]:
            st.markdown("""
                <div class='alert-churn'>
                    This customer is at risk of churning. Consider a targeted retention offer —
                    a contract upgrade or a discount on monthly charges may help.
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='alert-safe'>
                    This customer is likely to stay. No immediate intervention needed.
                </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("<p class='section-label'>Configure Parameters</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1], gap="large")
    avg_revenue = col1.number_input("Avg Monthly Revenue per Customer ($)", 0.0, 500.0, 74.0, step=1.0)
    retention_cost = col2.number_input("Retention Cost per Customer ($)", 0.0, 5000.0, 500.0, step=10.0)
    top_n = col3.slider("At-Risk Customers to Target", 10, 500, 100)

    st.markdown("<br>", unsafe_allow_html=True)
    calc_btn = st.button("Calculate Financial Impact", use_container_width=True)

    if calc_btn:
        impact = calculate_impact(avg_revenue, retention_cost, top_n)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4, gap="medium")

        with c1:
            st.markdown(f"""
                <div class='impact-card'>
                    <p class='impact-label'>Revenue Saved</p>
                    <p class='impact-number'>${impact['revenue_saved']:,.0f}</p>
                </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
                <div class='impact-card'>
                    <p class='impact-label'>Retention Cost</p>
                    <p class='impact-number'>${impact['retention_cost']:,.0f}</p>
                </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
                <div class='impact-card'>
                    <p class='impact-label'>Net Benefit</p>
                    <p class='impact-number'>${impact['net_benefit']:,.0f}</p>
                </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown(f"""
                <div class='impact-card'>
                    <p class='impact-label'>ROI</p>
                    <p class='impact-number'>{impact['roi']:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor("#161b22")
        ax.set_facecolor("#161b22")

        categories = ["Revenue Saved", "Retention Cost", "Net Benefit"]
        values = [impact["revenue_saved"], impact["retention_cost"], impact["net_benefit"]]
        colors = ["#34d399", "#f87171", "#60a5fa"]

        bars = ax.barh(categories, values, color=colors, height=0.4)
        ax.set_xlabel("Amount ($)", color="#6b7280", fontsize=9)
        ax.tick_params(colors="#6b7280", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("#2a2d35")
        ax.spines["left"].set_color("#2a2d35")

        for bar, val in zip(bars, values):
            ax.text(val + max(values)*0.01, bar.get_y() + bar.get_height()/2,
                   f"${val:,.0f}", va="center", color="#9ca3af", fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)