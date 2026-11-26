# 💳 Credit Risk Intelligence

A Machine Learning project to predict whether a loan applicant will default or not, and classify them into Low / Medium / High Risk categories. This helps banks make smarter lending decisions.

**Live Demo:** https://credit-risk-intelligence.vercel.app

## 🚀 Features
- Upload CSV file of loan applicants
- Predicts Default Probability for each applicant
- Classifies Risk: Low Risk / Medium Risk / High Risk
- Shows Summary Dashboard with charts
- Download Prediction Results as CSV

## 🛠️ Tech Stack
- **Frontend:** HTML, CSS, JavaScript (Vercel)
- **Backend:** Python, FastAPI (Render)
- **Machine Learning:** Scikit-learn, XGBoost, Pandas
- **Model:** Random Forest / XGBoost Classifier

## 📁 Project Structure
- `backend/` - FastAPI Backend API
- `frontend/` - Website UI
- `model/` - Trained ML Model (.pkl)
- `dataset/` - Dataset files
- `results/` - Prediction Output

## 📊 Results
- Total Applicants Tested: 200
- Low Risk: 78
- Medium Risk: 92
- High Risk: 30
- Model Accuracy: ~85%+

## ▶️ How to Run Locally
```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app:app --reload

# Frontend
Open frontend/index.html