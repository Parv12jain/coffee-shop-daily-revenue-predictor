import streamlit as st
import numpy as np
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Coffee Shop Revenue Predictor ☕",
    page_icon="☕",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
with open("model_coffee.pkl", "rb") as f:
    model = pickle.load(f)

# If you used StandardScaler during training, uncomment below
# with open("scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
}

.main-card {
    background: white;
    padding: 35px;
    border-radius: 22px;
    box-shadow: 0px 20px 50px rgba(0,0,0,0.12);
    animation: fadeIn 1.1s ease-in-out;
}

.title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    color: #6f4e37;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    color: #777;
    margin-bottom: 30px;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #6f4e37, #c19a6b);
    color: white;
    border-radius: 14px;
    padding: 14px;
    font-size: 18px;
    font-weight: 700;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #c19a6b, #6f4e37);
}

.result-box {
    background: #f6efe9;
    padding: 22px;
    border-radius: 16px;
    margin-top: 25px;
    text-align: center;
    font-size: 24px;
    font-weight: 700;
    color: #6f4e37;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)

# ---------------- UI ----------------
st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.markdown('<div class="title">☕ Coffee Shop Revenue Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered daily revenue estimation</div>', unsafe_allow_html=True)

st.subheader("📊 Enter Daily Business Details")

# ---------------- INPUTS (EXACT 6 FEATURES) ----------------
customers = st.number_input(
    "👥 Number of Customers Per Day",
    min_value=0,
    step=1
)

avg_order = st.number_input(
    "💳 Average Order Value (₹)",
    min_value=0.0
)

hours = st.number_input(
    "⏰ Operating Hours Per Day",
    min_value=0.0,
    max_value=24.0
)

employees = st.number_input(
    "👨‍🍳 Number of Employees",
    min_value=0,
    step=1
)

marketing = st.number_input(
    "📢 Marketing Spend Per Day (₹)",
    min_value=0.0
)

foot_traffic = st.number_input(
    "🚶 Location Foot Traffic",
    min_value=0,
    step=1
)

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Daily Revenue"):
    input_data = np.array([[
        customers,
        avg_order,
        hours,
        employees,
        marketing,
        foot_traffic
    ]])

    # If scaler was used during training
    # input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)[0]

    st.markdown(
        f'<div class="result-box">💰 Estimated Daily Revenue: ₹ {prediction:,.2f}</div>',
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown(
    "<p style='text-align:center;color:gray;margin-top:30px;'>Built with ❤️ using Streamlit & Machine Learning</p>",
    unsafe_allow_html=True
)
