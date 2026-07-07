# NeuroTwin AI

## Cognitive Digital Twin for Brain Health Intelligence

NeuroTwin AI is an AI-powered Cognitive Digital Twin that evaluates lifestyle, digital behavior, sleep quality, productivity, and daily habits to assess cognitive health.

The system predicts multiple cognitive health indicators using Machine Learning and provides personalized recommendations to improve overall brain wellness.

---

# Features

- Brain Fog Prediction
- Digital Addiction Assessment
- Attention Fragmentation Prediction
- Digital Overstimulation Analysis
- Memory Retention Prediction
- Cognitive Age Estimation
- Personalized Lifestyle Recommendations
- Lifestyle Simulation

---

# AI Models

| Model | Description |
| :--- | :--- |
| Brain Fog | Predicts mental fatigue level |
| Digital Addiction | Estimates dependency on digital devices |
| Attention Fragmentation | Measures concentration disruption |
| Digital Overstimulation | Evaluates excessive digital exposure |
| Memory Retention | Predicts memory performance |
| Cognitive Age | Estimates functional cognitive age |

---

# Dataset

The dataset contains lifestyle and digital behavior attributes including:

- Age
- Sleep Hours
- Sleep Quality
- Daily Screen Time
- Phone Unlocks
- Social Media Usage
- Gaming Usage
- Streaming Usage
- Exercise Minutes
- Study Hours
- Productivity Score
- Focus Index
- Stress Level
- Burnout Level
- Heart Rate
- Daily Steps
- Water Intake
- Nutrition Score
- Additional cognitive health indicators

---

# Technology Stack

| Category | Technologies |
| :--- | :--- |
| Frontend | React, Vite, HTML, CSS, JavaScript |
| Backend | Flask, Python |
| Machine Learning | Scikit-learn, Random Forest Regressor |
| Data Processing | Pandas, NumPy |
| Model Storage | Joblib |

---

# Project Structure

```text
NeuroTwin-AI/
│
├── backend/
│   ├── app.py
│   ├── predictor.py
│   ├── preprocess.py
│   ├── recommendations.py
│   ├── models_loader.py
│   ├── cognitive_age.py
│   ├── confidence.py
│   ├── overall_risk.py
│   ├── risk_levels.py
│   ├── sample_input.py
│   └── simulator.py
│
├── frontend/
│
├── data/
│   └── cognitive_digital_twin.csv
│
├── models/
│   ├── scaler.pkl
│   └── scaler_columns.json
│
├── reports/
│
├── train_models.py
├── train_brainfog.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Installation

## Clone the repository

```bash
git clone https://github.com/DharanJr/NeuroTwin-AI.git

cd NeuroTwin-AI
```

## Create a virtual environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run the backend

```bash
cd backend

python app.py
```

Backend:

```text
http://127.0.0.1:5000
```

## Run the frontend

```bash
cd frontend

npm install

npm run dev
```

Frontend:

```text
http://localhost:5173
```

---

# Machine Learning Workflow

```text
Dataset
    │
    ▼
Data Preprocessing
    │
    ▼
Feature Engineering
    │
    ▼
Random Forest Regression
    │
    ▼
Prediction
    │
    ▼
Risk Assessment
    │
    ▼
Personalized Recommendations
```

---

# Model Outputs

| Prediction |
| :--- |
| Brain Fog Score |
| Digital Addiction Score |
| Attention Fragmentation Score |
| Digital Overstimulation Score |
| Memory Retention Score |
| Cognitive Age |

---

# Future Improvements

- Deep Learning Models
- Explainable AI (SHAP)
- Mobile Application
- Wearable Device Integration
- User Authentication
- Cloud Deployment
- Real-Time Monitoring

---

# Model Files

The trained model files are not included in this repository because they exceed GitHub's file size limit.

Generate the trained models using:

```bash
python train_models.py
```

---

# Author

**Dharanidharan**

Bachelor of Technology in Artificial Intelligence & Data Science

GitHub: https://github.com/DharanJr
