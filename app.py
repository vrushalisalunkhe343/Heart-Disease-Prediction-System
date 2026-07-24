import streamlit as st
import pandas as pd
import joblib

# ----------------------------------
# PAGE CONFIG
# ----------------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# ----------------------------------
# LOAD MODEL
# ----------------------------------

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

# ----------------------------------
# CUSTOM CSS
# ----------------------------------

st.markdown("""
<style>

.stApp{
background:#f4f8fc;
}

h1{
text-align:center;
color:#c1121f;
font-weight:bold;
}

h3{
color:#003049;
}

.block-container{
padding-top:2rem;
padding-bottom:2rem;
max-width:850px;
}

div[data-testid="stForm"]{
background:white;
padding:25px;
border-radius:15px;
box-shadow:0px 0px 15px rgba(0,0,0,0.15);
}

.stButton>button{
background:#d62828;
color:white;
width:100%;
height:55px;
font-size:20px;
font-weight:bold;
border-radius:10px;
border:none;
}

.stButton>button:hover{
background:#9d0208;
color:white;
}

.result-good{

background:#d8f3dc;

padding:20px;

border-radius:12px;

font-size:22px;

font-weight:bold;

color:#2d6a4f;

text-align:center;

}

.result-bad{

background:#ffe5e5;

padding:20px;

border-radius:12px;

font-size:22px;

font-weight:bold;

color:#d00000;

text-align:center;

}

.footer{

text-align:center;

color:gray;

padding-top:25px;

}

</style>

""",unsafe_allow_html=True)

# ----------------------------------
# TITLE
# ----------------------------------

st.title("❤️ Heart Disease Prediction System")

st.caption("AI Powered Clinical Decision Support")

st.divider()

st.subheader("👤 Patient Details")
# ----------------------------------
# PATIENT DETAILS FORM
# ----------------------------------

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=35
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

bp = st.number_input(
    "Blood Pressure (mmHg)",
    min_value=50,
    max_value=250,
    value=120
)

cholesterol = st.number_input(
    "Cholesterol Level",
    min_value=100,
    max_value=400,
    value=180
)

exercise = st.selectbox(
    "Exercise Habits",
    ["Low", "Medium", "High"]
)

smoking = st.selectbox(
    "Smoking",
    ["No", "Yes"]
)

family = st.selectbox(
    "Family Heart Disease",
    ["No", "Yes"]
)

diabetes = st.selectbox(
    "Diabetes",
    ["No", "Yes"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=50.0,
    value=25.0
)

high_bp = st.selectbox(
    "High Blood Pressure",
    ["No", "Yes"]
)

low_hdl = st.selectbox(
    "Low HDL Cholesterol",
    ["No", "Yes"]
)

high_ldl = st.selectbox(
    "High LDL Cholesterol",
    ["No", "Yes"]
)

alcohol = st.selectbox(
    "Alcohol Consumption",
    ["Low", "Medium", "High"]
)

stress = st.selectbox(
    "Stress Level",
    ["Low", "Medium", "High"]
)

sleep = st.number_input(
    "Sleep Hours",
    min_value=0.0,
    max_value=15.0,
    value=7.0
)

sugar = st.selectbox(
    "Sugar Consumption",
    ["Low", "Medium", "High"]
)

triglyceride = st.number_input(
    "Triglyceride Level",
    min_value=50,
    max_value=600,
    value=150
)

fasting = st.number_input(
    "Fasting Blood Sugar",
    min_value=50.0,
    max_value=300.0,
    value=90.0
)

crp = st.number_input(
    "CRP Level",
    min_value=0.0,
    max_value=30.0,
    value=2.0
)

homocysteine = st.number_input(
    "Homocysteine Level",
    min_value=0.0,
    max_value=50.0,
    value=10.0
)

st.write("")

predict = st.button("❤️ Predict Heart Disease", use_container_width=True)

# ----------------------------------
# CONVERT INPUTS TO NUMERIC
# ----------------------------------

gender = 1 if gender == "Male" else 0

exercise = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}[exercise]

smoking = 1 if smoking == "Yes" else 0

family = 1 if family == "Yes" else 0

diabetes = 1 if diabetes == "Yes" else 0

high_bp = 1 if high_bp == "Yes" else 0

low_hdl = 1 if low_hdl == "Yes" else 0

high_ldl = 1 if high_ldl == "Yes" else 0

alcohol = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}[alcohol]

stress = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}[stress]

sugar = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}[sugar]
# ----------------------------------
# PREDICTION
# ----------------------------------

if predict:

    input_data = pd.DataFrame([[
        age,
        gender,
        bp,
        cholesterol,
        exercise,
        smoking,
        family,
        diabetes,
        bmi,
        high_bp,
        low_hdl,
        high_ldl,
        alcohol,
        stress,
        sleep,
        sugar,
        triglyceride,
        fasting,
        crp,
        homocysteine

    ]], columns=[

        'Age',
        'Gender',
        'Blood Pressure',
        'Cholesterol Level',
        'Exercise Habits',
        'Smoking',
        'Family Heart Disease',
        'Diabetes',
        'BMI',
        'High Blood Pressure',
        'Low HDL Cholesterol',
        'High LDL Cholesterol',
        'Alcohol Consumption',
        'Stress Level',
        'Sleep Hours',
        'Sugar Consumption',
        'Triglyceride Level',
        'Fasting Blood Sugar',
        'CRP Level',
        'Homocysteine Level'

    ])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)

    healthy = probability[0][0] * 100
    disease = probability[0][1] * 100

    st.divider()

    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:

        st.markdown("""
        <div class="result-bad">
        ❤️ High Risk of Heart Disease
        <br><br>
        Please consult a Cardiologist immediately.
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result-good">
        💚 No Heart Disease Detected
        <br><br>
        Keep following a healthy lifestyle.
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.subheader("Prediction Probability")

    st.write(f"💚 Healthy : {healthy:.2f}%")
    st.progress(int(healthy))

    st.write(f"❤️ Heart Disease : {disease:.2f}%")
    st.progress(int(disease))

    st.divider()

    st.subheader("🩺 Patient Summary")

    summary = pd.DataFrame({

        "Parameter":[
            "Age",
            "Blood Pressure",
            "Cholesterol",
            "BMI",
            "Sleep Hours",
            "Triglyceride",
            "Fasting Blood Sugar",
            "CRP",
            "Homocysteine"
        ],

        "Value":[
            age,
            bp,
            cholesterol,
            bmi,
            sleep,
            triglyceride,
            fasting,
            crp,
            homocysteine
        ]

    })

    st.dataframe(summary, use_container_width=True)

    st.success("Prediction Completed Successfully ✅")
    # ----------------------------------
# SIDEBAR
# ----------------------------------

with st.sidebar:

    st.markdown("## 🏥 Hospital Dashboard")

    st.info("""
Welcome to the Heart Disease Prediction System.

This application uses a trained Machine Learning model
(Logistic Regression) to predict the likelihood of
heart disease based on patient health parameters.
""")

    st.markdown("---")

    st.markdown("### ❤️ Healthy Heart Tips")

    st.success("""
🥗 Eat a balanced diet

🚶 Exercise 30 minutes daily

😴 Sleep 7-8 hours

🚭 Avoid Smoking

🍺 Limit Alcohol

🧘 Reduce Stress

💧 Drink Enough Water
""")

    st.markdown("---")

    st.markdown("### 📊 Model Information")

    st.write("✅ Algorithm : Logistic Regression")

    st.write("✅ Features : 20")

    st.write("✅ Prediction : Binary Classification")

    st.markdown("---")

    st.caption("Developed using Python + Streamlit")

# ----------------------------------
# FOOTER
# ----------------------------------

st.divider()

st.markdown("""

<div style="text-align:center;
padding:20px;
background:#ffffff;
border-radius:12px;
box-shadow:0px 0px 8px rgba(0,0,0,0.10);">

<h3 style="color:#c1121f;">
❤️ Heart Disease Prediction System
</h3>

<p style="font-size:16px;color:#555555;">
AI Powered Clinical Decision Support
</p>

<hr>

<p style="color:#003049;font-size:17px;">
Developed by <b>Vrushali Salunkhe</b>
</p>

<p style="color:gray;">
B.Tech Artificial Intelligence & Machine Learning
</p>

</div>

""", unsafe_allow_html=True)