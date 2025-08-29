# NHS Hospital Readmission Prediction System

A machine learning system that predicts 30-day hospital readmissions using NHS Electronic Health Record (EHR) data. Built with scikit-learn and designed for production deployment.

## 🎯 Project Overview

Predicts whether a patient will be readmitted within 30 days of discharge using structured NHS-style EHR data, helping hospitals:
- Identify high-risk patients
- Optimize discharge planning
- Reduce unnecessary readmissions

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic NHS data and setup project structure
python quick_start.py

# Run preprocessing pipeline
python src/preprocessing/preprocessing.py

# Train and evaluate models
python src/models/train_evaluate.py
```

## 📊 Dataset

- **Source**: Synthetic NHS-aligned EHR data
- **Size**: 1,500 admission episodes
- **Features**: 
  - Patient demographics
  - ICD-10 diagnosis codes
  - Admission details
  - Length of stay
  - Discharge destination
  - Comorbidities
- **Target**: 30-day readmission (binary)

## 📋 Project Status

- ✅ Phase 1: Data Generation & Setup
  - Created project structure
  - Generated synthetic NHS data
  - Basic data validation
  
- ✅ Phase 2: Data Preprocessing
  - Handled missing values
  - Encoded categorical variables
  - Generated temporal features
  - Created patient history features
  
- ✅ Phase 3: Model Training
  - Implemented multiple models
  - Handled class imbalance
  - Performed hyperparameter tuning
  - Generated evaluation metrics

- 🔄 Next Steps:
  - Phase 4: Model Interpretability (SHAP)
  - Phase 5: Flask Web Application
  - Phase 6: Production Deployment

## 📁 Project Structure

```
nhs-readmission/
├── data/
│   ├── raw/               # Original dataset
│   └── processed/         # Preprocessed features
├── src/
│   ├── preprocessing/     # Data cleaning & features
│   ├── models/           # Model training & evaluation
│   └── visualization/    # Analysis plots
├── results/              # Model performance metrics
├── models/              # Saved model files
└── app/                 # Flask web application
```

## 🛠 Technologies

- Python 3.11
- scikit-learn
- pandas
- XGBoost
- Flask (coming soon)
- SHAP (coming soon)

## 📈 Current Performance

Latest model metrics on test set:
- Logistic Regression: ROC-AUC = 0.552
- Random Forest: ROC-AUC = 0.467
- XGBoost: ROC-AUC = 0.441

## 👥 Contributors

- NHS ML Team
- Last Updated: August 2025

## 📝 License

MIT License
