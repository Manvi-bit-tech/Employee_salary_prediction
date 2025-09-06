import streamlit as st
import pandas as pd
import joblib
import pickle

# --- Load All Artifacts ---
# This function will cache the models to avoid reloading on every interaction
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("best_model.pkl")
        scaler = joblib.load("scaler.pkl")
        with open('label_encoders.pkl', 'rb') as f:
            le_dict = pickle.load(f)
        return model, scaler, le_dict
    except FileNotFoundError:
        st.error("Error: One or more required files (.pkl) not found. Please run the final notebook to generate them.")
        st.stop()

model, scaler, le_dict = load_artifacts()

# --- Page Configuration ---
st.set_page_config(
    page_title="Salary-Predix | AI Financial Analytics",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a "Financial Analytics" Dashboard Look ---
st.markdown("""
<style>
    /* --- Keyframes for Animations --- */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* --- General Styling (Dark Theme) --- */
    [data-testid="stAppViewContainer"] {
        background-color: #0d1117; /* Dark background */
        background-image: 
            linear-gradient(rgba(13, 17, 23, 0.95), rgba(13, 17, 23, 0.95)),
            url("data:image/svg+xml,%3Csvg width='52' height='26' viewBox='0 0 52 26' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%231c2e4a' fill-opacity='0.4'%3E%3Cpath d='M10 10c0-2.21-1.79-4-4-4-3.314 0-6-2.686-6-6h2c0 2.21 1.79 4 4 4 3.314 0 6 2.686 6 6 0 2.21 1.79 4 4 4 3.314 0 6 2.686 6 6 0 2.21 1.79 4 4 4v2c-3.314 0-6-2.686-6-6 0-2.21-1.79-4-4-4-3.314 0-6-2.686-6-6zm25.464-1.95l8.486 8.486-1.414 1.414-8.486-8.486 1.414-1.414z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        color: #EAEAEA;
    }

    /* --- Glassmorphism Cards --- */
    .main .block-container, [data-testid="stSidebar"] {
        background: rgba(22, 27, 34, 0.6); /* Semi-transparent dark card */
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(88, 166, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        animation: fadeIn 0.8s ease-out;
    }
    
    .main .block-container { padding: 2rem 3rem; }
    
    /* Text styling */
    h1, h2, h3 { color: #FFFFFF !important; }
    p, label, .stMarkdown { color: #c9d1d9 !important; } /* Lighter gray for readability */
    h1 {
        font-weight: 700 !important;
        color: #F2C94C !important; /* Gold for main title */
        text-shadow: 0 0 8px rgba(242, 201, 76, 0.4);
    }
    h2, h3 { color: #58A6FF !important; } /* Corporate Blue for headers */

    /* --- Sidebar Widgets --- */
    [data-testid="stSidebar"] .stSlider, 
    [data-testid="stSidebar"] .stSelectbox, 
    [data-testid="stSidebar"] .stNumberInput {
        background-color: rgba(13, 17, 23, 0.5);
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
        border: 1px solid rgba(88, 166, 255, 0.2);
    }
    
    /* Slider styling */
    div[data-baseweb="slider"] > div:nth-child(2) { background: linear-gradient(to right, #58A6FF, #F2C94C); }
    div[data-baseweb="slider"] > div:nth-child(3) { background-color: #FFFFFF !important; }

    /* --- Component Styling --- */
    .result-card {
        background: #0d1117;
        border-radius: 10px;
        padding: 25px;
        text-align: center;
        border: 1px solid #F2C94C;
        box-shadow: 0 0 20px rgba(242, 201, 76, 0.2);
    }
    .result-card h3 { margin-bottom: 10px; }
    .result-card p { font-size: 1.6rem; font-weight: bold; }
    
    .stButton>button {
        background: linear-gradient(90deg, #F2C94C, #F2994A);
        color: #0d1117;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(242, 201, 76, 0.2);
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(242, 201, 76, 0.4);
    }
    
    /* Icon-based summary styling */
    .summary-item {
        display: flex;
        align-items: center;
        margin-bottom: 12px;
        font-size: 1.1rem;
        background-color: rgba(13, 17, 23, 0.5);
        padding: 8px 12px;
        border-radius: 6px;
    }
    .summary-item span {
        font-size: 1.5rem;
        margin-right: 12px;
        color: #F2C94C;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar For User Input ---
with st.sidebar:
    st.header("👤 Employee Details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 17, 90, 35)
        workclass = st.selectbox("Work Class", le_dict['workclass'].classes_)
        capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
        gender = st.selectbox("Gender", le_dict['gender'].classes_)
    with col2:
        hours_per_week = st.slider("Hours per Week", 1, 99, 40)
        marital_status = st.selectbox("Marital Status", le_dict['marital-status'].classes_)
        capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
        race = st.selectbox("Race", le_dict['race'].classes_)

    occupation = st.selectbox("Occupation", le_dict['occupation'].classes_)
    native_country = st.selectbox("Native Country", le_dict['native-country'].classes_)
    educational_num = st.slider("Years of Education", 1, 16, 10, key="edu_num")
    fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=1, value=77516, key="fnlwgt_num")


# --- Main Page Content ---
st.title("Salary-Predix AI Analytics")
st.markdown("An intelligent system to predict employee income class based on census data.")
st.divider()

left_col, right_col = st.columns([2, 1.5], gap="large")

with left_col:
    st.header("📊 Profile Summary")
    # Using icons for a more thematic look
    summary_html = f"""
        <div class='summary-item'><span>🎂</span> Age: {age}</div>
        <div class='summary-item'><span>🏢</span> Work Class: {workclass}</div>
        <div class='summary-item'><span>🎓</span> Years of Education: {educational_num}</div>
        <div class='summary-item'><span>💍</span> Marital Status: {marital_status}</div>
        <div class='summary-item'><span>💼</span> Occupation: {occupation}</div>
        <div class='summary-item'><span>🚻</span> Gender: {gender}</div>
        <div class='summary-item'><span>⏰</span> Hours per Week: {hours_per_week}</div>
        <div class='summary-item'><span>📈</span> Capital Gain: ${capital_gain:,}</div>
        <div class='summary-item'><span>📉</span> Capital Loss: ${capital_loss:,}</div>
    """
    st.markdown(summary_html, unsafe_allow_html=True)

with right_col:
    st.header("💡 Prediction Result")
    if st.button("Analyze & Predict Salary Class"):
        # --- Prepare Data for Prediction ---
        input_data = {
            'age': age, 'workclass': workclass, 'fnlwgt': fnlwgt, 'educational-num': educational_num,
            'marital-status': marital_status, 'occupation': occupation, 'race': race,
            'gender': gender, 'capital-gain': capital_gain, 'capital-loss': capital_loss,
            'hours-per-week': hours_per_week, 'native-country': native_country,
        }
        input_df = pd.DataFrame([input_data])

        # Apply saved Label Encoders
        for col, le in le_dict.items():
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col])

        # Scale numerical features
        numerical_cols = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        # Enforce column order to match model's training
        expected_features = ['age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status', 'occupation',
                             'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
        input_df = input_df[expected_features]

        # --- Prediction ---
        prediction = model.predict(input_df)
        
        # --- Display Result ---
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        if prediction[0] == 1:
            st.markdown("<h3>Predicted Salary Class</h3>", unsafe_allow_html=True)
            st.markdown("<p style='color: #2ECC71;'>> $50K per year</p>", unsafe_allow_html=True) # Bright Green
            st.balloons()
        else:
            st.markdown("<h3>Predicted Salary Class</h3>", unsafe_allow_html=True)
            st.markdown("<p style='color: #E74C3C;'><= $50K per year</p>", unsafe_allow_html=True) # Bright Red
        st.markdown('</div>', unsafe_allow_html=True)
        
        with st.expander("Show Technical Details"):
            st.write("Data sent to the model (after encoding and scaling):")
            st.dataframe(input_df)
    else:
        st.info("Input employee details in the sidebar and click the button above to see the prediction.")


