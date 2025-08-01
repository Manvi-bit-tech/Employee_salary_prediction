
# 💼 Employee Salary Prediction

This project predicts whether an employee earns **more than 50K or less than or equal to 50K** annually based on features like age, education, occupation, work hours, and experience. The model is deployed using a simple yet effective **Streamlit web application** that supports real-time as well as batch predictions.

---

## 📌 Table of Contents

- [Problem Statement](#problem-statement)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [How It Works](#how-it-works)
- [How to Run the Project](#how-to-run-the-project)
- [Results](#results)
- [Future Scope](#future-scope)
- [License](#license)

---

## 💡 Problem Statement

Manually predicting or benchmarking employee salaries is time-consuming and prone to bias. This project aims to use machine learning techniques to build a system that classifies salaries into two categories: `>50K` or `<=50K`, providing a reliable tool for HR and recruitment teams.

---

## ⚙️ Tech Stack

- **Programming Language**: Python  
- **Libraries/Frameworks**:  
  - `pandas`, `numpy`, `scikit-learn`, `joblib`  
  - `matplotlib`, `seaborn`, `streamlit`

---

## 🌟 Features

- 🎛️ Interactive sidebar input for real-time predictions
- 📂 Batch prediction through CSV upload
- 📈 Displays processed input data and results
- 💾 Downloadable CSV file of predicted outputs
- 🧠 Model loaded from serialized `.pkl` file
- 🖥️ Clean, responsive user interface with Streamlit

---

## 🚀 How It Works

1. User provides employee details via sidebar or CSV upload.
2. Input is preprocessed (encoding categorical values).
3. Trained ML model (`RandomForestClassifier`) makes a prediction.
4. Results are displayed on the interface and available for download.

---

## 🛠️ How to Run the Project

1. **Clone the repository**  
```bash
git clone https://github.com/your-username/employee-salary-prediction.git
cd employee-salary-prediction
```

2. **Install dependencies**  
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**  
```bash
streamlit run app.py
```

4. The app will open in your default browser. Enter employee details or upload a CSV and click "Predict Salary Class".

---

## 📊 Results

- **Model Used**: Random Forest Classifier  
- **Accuracy**: ~87%  
- **Performance**: Good precision and recall for both salary classes  
- **Deployment**: Streamlit UI for easy interaction

---

## 🔭 Future Scope

- Add model explainability with SHAP/LIME  
- Use larger and more diverse datasets  
- Extend to predict exact salary using regression  
- Integrate with company HRMS systems  

---

## 🔗 Live Demo & Resources

- 🖥️ Streamlit App: *[Insert Streamlit link here]*  
- 💻 GitHub Repo: https://github.com/Manvi-bit-tech/Employee_Salary_Prediction

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙋‍♀️ Author

**Manvi Dhamija**  
B.Tech ECE | Engineering Student | ML Enthusiast  
🔗 [LinkedIn](https://www.linkedin.com/in/your-profile)  
🔗 [GitHub](https://github.com/Manvi-bit-tech)

---
