import streamlit as st
import pandas as pd
import joblib
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time

# --- Load All Artifacts ---
try:
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open('label_encoders.pkl', 'rb') as f:
        le_dict = pickle.load(f)
except FileNotFoundError:
    st.error("Error: One or more required files (.pkl) not found. Please run the final notebook.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Employee Salary Prediction | AI-Powered Income Classification",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Theme State Management ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# --- Theme Toggle Function ---
def toggle_theme():
    if st.session_state.theme == 'light':
        st.session_state.theme = 'dark'
    else:
        st.session_state.theme = 'light'

# --- Dynamic CSS based on theme ---
def get_theme_css(theme):
    if theme == 'dark':
        return """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
            
            * {
                font-family: 'Inter', sans-serif;
            }
            
            /* Hide Streamlit default elements */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            /* Global styles */
            .main .block-container {
                padding: 0;
                max-width: 1400px;
            }
            
            /* Animated Background */
            body {
                background: linear-gradient(-45deg, #0f0f23, #1a1a2e, #16213e, #0f3460);
                background-size: 400% 400%;
                animation: gradientShift 15s ease infinite;
            }
            
            @keyframes gradientShift {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
            
            /* Floating particles */
            .particles {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: -1;
            }
            
            .particle {
                position: absolute;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 50%;
                animation: floatParticle 20s infinite linear;
            }
            
            @keyframes floatParticle {
                0% { transform: translateY(100vh) rotate(0deg); opacity: 0; }
                10% { opacity: 1; }
                90% { opacity: 1; }
                100% { transform: translateY(-100vh) rotate(360deg); opacity: 0; }
            }
            
            /* Hero Section */
            .hero-section {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 4rem 2rem;
                text-align: center;
                color: white;
                position: relative;
                overflow: hidden;
            }
            
            .hero-section::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 100" fill="white" opacity="0.1"><polygon points="0,0 1000,100 1000,0"/></svg>');
                background-size: cover;
            }
            
            .hero-content {
                position: relative;
                z-index: 1;
            }
            
            .hero-title {
                font-size: 3.5rem;
                font-weight: 800;
                margin-bottom: 1rem;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                background: linear-gradient(45deg, #fff, #f0f0f0);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .hero-subtitle {
                font-size: 1.3rem;
                font-weight: 400;
                opacity: 0.9;
                margin-bottom: 2rem;
            }
            
            .hero-stats {
                display: flex;
                justify-content: center;
                gap: 3rem;
                margin-top: 2rem;
            }
            
            .stat-item {
                text-align: center;
            }
            
            .stat-number {
                font-size: 2rem;
                font-weight: 700;
                display: block;
            }
            
            .stat-label {
                font-size: 0.9rem;
                opacity: 0.8;
            }
            
            /* Main Content */
            .main-content {
                background: rgba(15, 15, 35, 0.9);
                min-height: 100vh;
                padding: 0.5rem 0;
                backdrop-filter: blur(10px);
            }
            
            .content-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 2rem;
            }
            
            /* Cards */
            .card {
                background: rgba(26, 26, 46, 0.8);
                border-radius: 20px;
                padding: 1rem;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                border: 1px solid rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
                margin-bottom: 0.5rem;
                transition: all 0.3s ease;
            }
            
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 20px 60px rgba(0,0,0,0.4);
            }
            
            .card-header {
                display: flex;
                align-items: center;
                margin-bottom: 0.8rem;
                padding-bottom: 0.6rem;
                border-bottom: 2px solid rgba(255,255,255,0.1);
            }
            
            .card-icon {
                font-size: 2rem;
                margin-right: 1rem;
            }
            
            .card-title {
                font-size: 1.5rem;
                font-weight: 700;
                color: #ffffff;
                margin: 0;
            }
            
            /* Input Styling */
            .stSelectbox > div > div {
                background: rgba(26, 26, 46, 0.8);
                border-radius: 12px;
                border: 2px solid rgba(255,255,255,0.2);
                transition: all 0.3s ease;
                color: white;
            }
            
            .stSelectbox > div > div:hover {
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            
            .stSlider > div > div > div > div {
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                border-radius: 10px;
            }
            
            .stNumberInput > div > div > input {
                background: rgba(26, 26, 46, 0.8);
                border-radius: 12px;
                border: 2px solid rgba(255,255,255,0.2);
                transition: all 0.3s ease;
                color: white;
            }
            
            .stNumberInput > div > div > input:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            
            /* Button Styling */
            .predict-button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 50px;
                padding: 1rem 3rem;
                font-size: 1.2rem;
                font-weight: 600;
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
                transition: all 0.3s ease;
                width: 100%;
                cursor: pointer;
                position: relative;
                overflow: hidden;
            }
            
            .predict-button::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                transition: left 0.5s;
            }
            
            .predict-button:hover::before {
                left: 100%;
            }
            
            .predict-button:hover {
                transform: translateY(-3px);
                box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
            }
            
            /* Prediction Results */
            .prediction-result {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 3rem;
                border-radius: 20px;
                text-align: center;
                margin: 2rem 0;
                position: relative;
                overflow: hidden;
            }
            
            .prediction-result::before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
                animation: float 6s ease-in-out infinite;
            }
            
            @keyframes float {
                0%, 100% { transform: translateY(0px) rotate(0deg); }
                50% { transform: translateY(-20px) rotate(180deg); }
            }
            
            .prediction-title {
                font-size: 2.5rem;
                font-weight: 800;
                margin-bottom: 1rem;
                position: relative;
                z-index: 1;
            }
            
            .prediction-text {
                font-size: 1.3rem;
                opacity: 0.9;
                position: relative;
                z-index: 1;
            }
            
            /* Metrics */
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin: 2rem 0;
            }
            
            .metric-card {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                position: relative;
                overflow: hidden;
            }
            
            .metric-card::before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
                animation: pulse 4s ease-in-out infinite;
            }
            
            @keyframes pulse {
                0%, 100% { transform: scale(1); opacity: 0.5; }
                50% { transform: scale(1.1); opacity: 0.8; }
            }
            
            .metric-value {
                font-size: 2rem;
                font-weight: 800;
                margin-bottom: 0.5rem;
                position: relative;
                z-index: 1;
            }
            
            .metric-label {
                font-size: 0.9rem;
                opacity: 0.9;
                position: relative;
                z-index: 1;
            }
            
            /* Chart Container */
            .chart-container {
                background: rgba(26, 26, 46, 0.8);
                padding: 2rem;
                border-radius: 20px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                margin: 2rem 0;
                backdrop-filter: blur(10px);
            }
            
            /* Profile Summary */
            .profile-table {
                background: rgba(26, 26, 46, 0.8);
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 5px 20px rgba(0,0,0,0.3);
            }
            
            /* Sidebar */
            .sidebar .sidebar-content {
                background: linear-gradient(180deg, rgba(15, 15, 35, 0.9) 0%, rgba(26, 26, 46, 0.9) 100%);
            }
            
            .sidebar .sidebar-content .block-container {
                padding: 1rem 0.8rem;
            }
            
            /* Section Headers */
            .section-header {
                display: flex;
                align-items: center;
                margin: 0.5rem 0 0.3rem 0;
                padding: 0.6rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
                font-weight: 600;
                font-size: 0.9rem;
            }
            
            .section-icon {
                font-size: 1.5rem;
                margin-right: 1rem;
            }
            
            /* Theme Toggle */
            .theme-toggle {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 1000;
                background: rgba(26, 26, 46, 0.8);
                border: 2px solid rgba(255,255,255,0.2);
                border-radius: 50px;
                padding: 0.5rem 1rem;
                color: white;
                cursor: pointer;
                transition: all 0.3s ease;
                backdrop-filter: blur(10px);
            }
            
            .theme-toggle:hover {
                background: rgba(102, 126, 234, 0.8);
                transform: scale(1.05);
            }
            
            /* Responsive Design */
            @media (max-width: 768px) {
                .hero-title {
                    font-size: 2.5rem;
                }
                .hero-stats {
                    flex-direction: column;
                    gap: 1rem;
                }
                .content-container {
                    padding: 0 1rem;
                }
            }
            
            /* Loading Animation */
            .loading-spinner {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid rgba(255,255,255,.3);
                border-radius: 50%;
                border-top-color: #fff;
                animation: spin 1s ease-in-out infinite;
            }
            
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
        </style>
        """
    else:  # Light theme
        return """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
            
            * {
                font-family: 'Inter', sans-serif;
            }
            
            /* Hide Streamlit default elements */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            /* Global styles */
            .main .block-container {
                padding: 0;
                max-width: 1400px;
            }
            
            /* Animated Background */
            body {
                background: linear-gradient(-45deg, #f8fafc, #e2e8f0, #cbd5e1, #94a3b8);
                background-size: 400% 400%;
                animation: gradientShift 15s ease infinite;
            }
            
            @keyframes gradientShift {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }
            
            /* Floating particles */
            .particles {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: -1;
            }
            
            .particle {
                position: absolute;
                background: rgba(102, 126, 234, 0.1);
                border-radius: 50%;
                animation: floatParticle 20s infinite linear;
            }
            
            @keyframes floatParticle {
                0% { transform: translateY(100vh) rotate(0deg); opacity: 0; }
                10% { opacity: 1; }
                90% { opacity: 1; }
                100% { transform: translateY(-100vh) rotate(360deg); opacity: 0; }
            }
            
            /* Hero Section */
            .hero-section {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 4rem 2rem;
                text-align: center;
                color: white;
                position: relative;
                overflow: hidden;
            }
            
            .hero-section::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 100" fill="white" opacity="0.1"><polygon points="0,0 1000,100 1000,0"/></svg>');
                background-size: cover;
            }
            
            .hero-content {
                position: relative;
                z-index: 1;
            }
            
            .hero-title {
                font-size: 3.5rem;
                font-weight: 800;
                margin-bottom: 1rem;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                background: linear-gradient(45deg, #fff, #f0f0f0);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .hero-subtitle {
                font-size: 1.3rem;
                font-weight: 400;
                opacity: 0.9;
                margin-bottom: 2rem;
            }
            
            .hero-stats {
                display: flex;
                justify-content: center;
                gap: 3rem;
                margin-top: 2rem;
            }
            
            .stat-item {
                text-align: center;
            }
            
            .stat-number {
                font-size: 2rem;
                font-weight: 700;
                display: block;
            }
            
            .stat-label {
                font-size: 0.9rem;
                opacity: 0.8;
            }
            
            /* Main Content */
            .main-content {
                background: rgba(248, 250, 252, 0.9);
                min-height: 100vh;
                padding: 2rem 0;
                backdrop-filter: blur(10px);
            }
            
            .content-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 2rem;
            }
            
            /* Cards */
            .card {
                background: rgba(255, 255, 255, 0.9);
                border-radius: 20px;
                padding: 2rem;
                box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                border: 1px solid rgba(255,255,255,0.2);
                backdrop-filter: blur(10px);
                margin-bottom: 2rem;
                transition: all 0.3s ease;
            }
            
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 20px 60px rgba(0,0,0,0.15);
            }
            
            .card-header {
                display: flex;
                align-items: center;
                margin-bottom: 1.5rem;
                padding-bottom: 1rem;
                border-bottom: 2px solid #f1f5f9;
            }
            
            .card-icon {
                font-size: 2rem;
                margin-right: 1rem;
            }
            
            .card-title {
                font-size: 1.5rem;
                font-weight: 700;
                color: #1e293b;
                margin: 0;
            }
            
            /* Input Styling */
            .stSelectbox > div > div {
                background: white;
                border-radius: 12px;
                border: 2px solid #e2e8f0;
                transition: all 0.3s ease;
            }
            
            .stSelectbox > div > div:hover {
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            
            .stSlider > div > div > div > div {
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                border-radius: 10px;
            }
            
            .stNumberInput > div > div > input {
                background: white;
                border-radius: 12px;
                border: 2px solid #e2e8f0;
                transition: all 0.3s ease;
            }
            
            .stNumberInput > div > div > input:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            
            /* Button Styling */
            .predict-button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 50px;
                padding: 1rem 3rem;
                font-size: 1.2rem;
                font-weight: 600;
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
                transition: all 0.3s ease;
                width: 100%;
                cursor: pointer;
                position: relative;
                overflow: hidden;
            }
            
            .predict-button::before {
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
                transition: left 0.5s;
            }
            
            .predict-button:hover::before {
                left: 100%;
            }
            
            .predict-button:hover {
                transform: translateY(-3px);
                box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
            }
            
            /* Prediction Results */
            .prediction-result {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 3rem;
                border-radius: 20px;
                text-align: center;
                margin: 2rem 0;
                position: relative;
                overflow: hidden;
            }
            
            .prediction-result::before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
                animation: float 6s ease-in-out infinite;
            }
            
            @keyframes float {
                0%, 100% { transform: translateY(0px) rotate(0deg); }
                50% { transform: translateY(-20px) rotate(180deg); }
            }
            
            .prediction-title {
                font-size: 2.5rem;
                font-weight: 800;
                margin-bottom: 1rem;
                position: relative;
                z-index: 1;
            }
            
            .prediction-text {
                font-size: 1.3rem;
                opacity: 0.9;
                position: relative;
                z-index: 1;
            }
            
            /* Metrics */
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin: 2rem 0;
            }
            
            .metric-card {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                position: relative;
                overflow: hidden;
            }
            
            .metric-card::before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
                animation: pulse 4s ease-in-out infinite;
            }
            
            @keyframes pulse {
                0%, 100% { transform: scale(1); opacity: 0.5; }
                50% { transform: scale(1.1); opacity: 0.8; }
            }
            
            .metric-value {
                font-size: 2rem;
                font-weight: 800;
                margin-bottom: 0.5rem;
                position: relative;
                z-index: 1;
            }
            
            .metric-label {
                font-size: 0.9rem;
                opacity: 0.9;
                position: relative;
                z-index: 1;
            }
            
            /* Chart Container */
            .chart-container {
                background: white;
                padding: 2rem;
                border-radius: 20px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.1);
                margin: 2rem 0;
            }
            
            /* Profile Summary */
            .profile-table {
                background: white;
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            }
            
            /* Sidebar */
            .sidebar .sidebar-content {
                background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
            }
            
            .sidebar .sidebar-content .block-container {
                padding: 1rem 0.8rem;
            }
            
            /* Section Headers */
            .section-header {
                display: flex;
                align-items: center;
                margin: 0.5rem 0 0.3rem 0;
                padding: 0.6rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
                font-weight: 600;
                font-size: 0.9rem;
            }
            
            .section-icon {
                font-size: 1.5rem;
                margin-right: 1rem;
            }
            
            /* Theme Toggle */
            .theme-toggle {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 1000;
                background: rgba(255, 255, 255, 0.9);
                border: 2px solid rgba(102, 126, 234, 0.2);
                border-radius: 50px;
                padding: 0.5rem 1rem;
                color: #1e293b;
                cursor: pointer;
                transition: all 0.3s ease;
                backdrop-filter: blur(10px);
            }
            
            .theme-toggle:hover {
                background: rgba(102, 126, 234, 0.8);
                color: white;
                transform: scale(1.05);
            }
            
            /* Responsive Design */
            @media (max-width: 768px) {
                .hero-title {
                    font-size: 2.5rem;
                }
                .hero-stats {
                    flex-direction: column;
                    gap: 1rem;
                }
                .content-container {
                    padding: 0 1rem;
                }
            }
            
            /* Loading Animation */
            .loading-spinner {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid rgba(255,255,255,.3);
                border-radius: 50%;
                border-top-color: #fff;
                animation: spin 1s ease-in-out infinite;
            }
            
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
        </style>
        """

# --- Apply Theme CSS ---
st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)

# --- Theme Toggle Button ---
col1, col2, col3 = st.columns([1, 1, 1])
with col3:
    if st.button(f"{'🌙 Dark Mode' if st.session_state.theme == 'light' else '☀️ Light Mode'}", key="theme_toggle"):
        toggle_theme()
        st.rerun()

# --- Floating Particles Background ---
st.markdown("""
<div class="particles" id="particles"></div>
<script>
    // Create floating particles
    function createParticles() {
        const particlesContainer = document.getElementById('particles');
        const particleCount = 50;
        
        for (let i = 0; i < particleCount; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.width = Math.random() * 4 + 2 + 'px';
            particle.style.height = particle.style.width;
            particle.style.animationDelay = Math.random() * 20 + 's';
            particle.style.animationDuration = (Math.random() * 10 + 15) + 's';
            particlesContainer.appendChild(particle);
        }
    }
    
    createParticles();
</script>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown("""
<div class="hero-section">
    <div class="hero-content">
        <h1 class="hero-title">💼 Employee Salary Prediction</h1>
        <p class="hero-subtitle">AI-Powered Income Classification System</p>
        <div class="hero-stats">
            <div class="stat-item">
                <span class="stat-number">95%</span>
                <span class="stat-label">Accuracy</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">12</span>
                <span class="stat-label">Features</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">ML</span>
                <span class="stat-label">Powered</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Main Content ---
st.markdown('<div class="main-content">', unsafe_allow_html=True)
st.markdown('<div class="content-container">', unsafe_allow_html=True)

# --- Main Layout ---
col1, col2 = st.columns([1, 1.5])

with col1:
    # --- Input Form ---
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <span class="card-icon">📝</span>
            <h3 class="card-title">Employee Profile</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Personal Information
    st.markdown('<div class="section-header"><span class="section-icon">👤</span>Personal Information</div>', unsafe_allow_html=True)
    age = st.slider("Age", 17, 90, 35, help="Employee's age in years")
    gender = st.selectbox("Gender", le_dict['gender'].classes_, help="Employee's gender")
    race = st.selectbox("Race", le_dict['race'].classes_, help="Employee's race/ethnicity")
    marital_status = st.selectbox("Marital Status", le_dict['marital-status'].classes_, help="Current marital status")
    
    # Professional Information
    st.markdown('<div class="section-header"><span class="section-icon">💼</span>Professional Information</div>', unsafe_allow_html=True)
    workclass = st.selectbox("Work Class", le_dict['workclass'].classes_, help="Type of employment")
    occupation = st.selectbox("Occupation", le_dict['occupation'].classes_, help="Job category")
    educational_num = st.slider("Years of Education", 1, 16, 10, help="Number of years of formal education")
    hours_per_week = st.slider("Hours per Week", 1, 99, 40, help="Average hours worked per week")
    
    # Financial Information
    st.markdown('<div class="section-header"><span class="section-icon">💰</span>Financial Information</div>', unsafe_allow_html=True)
    capital_gain = st.number_input("Capital Gain", min_value=0, value=0, help="Capital gains in the past year")
    capital_loss = st.number_input("Capital Loss", min_value=0, value=0, help="Capital losses in the past year")
    
    # Additional Information
    st.markdown('<div class="section-header"><span class="section-icon">🌍</span>Additional Information</div>', unsafe_allow_html=True)
    fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=1, value=77516, help="Statistical weight for this record")
    native_country = st.selectbox("Native Country", le_dict['native-country'].classes_, help="Country of origin")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # --- Prediction Section ---
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <span class="card-icon">🎯</span>
            <h3 class="card-title">Salary Prediction</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Prepare Data for Prediction
    input_data = {
    'age': age, 'workclass': workclass, 'fnlwgt': fnlwgt, 'educational-num': educational_num,
    'marital-status': marital_status, 'occupation': occupation, 'race': race,
    'gender': gender, 'capital-gain': capital_gain, 'capital-loss': capital_loss,
    'hours-per-week': hours_per_week, 'native-country': native_country,
}
input_df = pd.DataFrame([input_data])

    # Apply Label Encoders
for col, le in le_dict.items():
    if col in input_df.columns:
        input_df[col] = le.transform(input_df[col])

    # Scale numerical features
numerical_cols = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Enforce column order
expected_features = ['age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status', 'occupation', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
input_df = input_df[expected_features]

    # Prediction Button
if st.button("🚀 Predict Salary Class", type="primary", key="predict_btn"):
        with st.spinner("Analyzing employee profile..."):
            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)
            
            # Display Results
            if prediction[0] == 1:
                st.markdown("""
                <div class="prediction-result">
                    <h2 class="prediction-title">🎉 High Income Prediction</h2>
                    <p class="prediction-text">This individual is predicted to earn <strong>more than $50K</strong> per year.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-result">
                    <h2 class="prediction-title">📊 Standard Income Prediction</h2>
                    <p class="prediction-text">This individual is predicted to earn <strong>$50K or less</strong> per year.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Metrics
            st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)
            col_metric1, col_metric2 = st.columns(2)
            
            with col_metric1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{max(prediction_proba[0]):.1%}</div>
                    <div class="metric-label">Confidence Level</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_metric2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{prediction_proba[0][1]:.1%}</div>
                    <div class="metric-label">High Income Probability</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Visualization
            st.markdown("""
            <div class="chart-container">
                <h3 style="text-align: center; margin-bottom: 2rem; color: #1e293b;">📈 Prediction Analysis</h3>
            """, unsafe_allow_html=True)
            
            # Probability Bar Chart
            prob_data = pd.DataFrame({
                'Income Category': ['≤ $50K', '> $50K'],
                'Probability': [prediction_proba[0][0], prediction_proba[0][1]]
            })
            
            fig = px.bar(
                prob_data, 
                x='Income Category', 
                y='Probability',
                color='Income Category',
                color_discrete_map={'≤ $50K': '#3498db', '> $50K': '#2ca02c'},
                title="Income Prediction Probabilities"
            )
            fig.update_layout(
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=14, family="Inter"),
                title_font_size=18,
                title_font_color="#1e293b"
            )
            fig.update_layout(yaxis=dict(tickformat='.1%'))
            fig.update_traces(marker_line_width=0)
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Profile Summary
            st.markdown("""
            <div class="card">
                <div class="card-header">
                    <span class="card-icon">🔍</span>
                    <h3 class="card-title">Profile Summary</h3>
                </div>
            """, unsafe_allow_html=True)
            
            profile_summary = pd.DataFrame({
                'Attribute': ['Age', 'Education (Years)', 'Hours/Week', 'Work Class', 'Occupation'],
                'Value': [age, educational_num, hours_per_week, workclass, occupation]
            })
            
            st.dataframe(
                profile_summary,
                use_container_width=True,
                hide_index=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
    
st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div style="text-align: center; padding: 1.5rem 0; color: #64748b;">
    <h3 style="color: #1e293b; margin-bottom: 1rem;">🤖 Powered by Machine Learning</h3>
    <p style="margin-bottom: 0.5rem;">Built with Streamlit & Python</p>
    <p style="font-size: 0.9rem; opacity: 0.7; margin-bottom: 0.5rem;">This prediction is based on historical data patterns and should be used for informational purposes only.</p>
    <div style="margin-top: 1rem; padding: 1rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); border-radius: 15px; border: 1px solid rgba(102, 126, 234, 0.2);">
        <p style="font-size: 1.1rem; font-weight: 600; color: #667eea; margin: 0; background: linear-gradient(45deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">✨ Crafted with passion by <strong>Manvi Dhamija</strong> ✨</p>
        <p style="font-size: 0.8rem; color: #94a3b8; margin: 0.3rem 0 0 0;">Turning data into insights, one prediction at a time</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


