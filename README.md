# Lung-cancer-prediction

**LUNG CANCER PREDICTION USING DEEP LEARNING**
This project applies Feedforward Neural Networks (FNN) to predict the likelihood of lung cancer based on patient clinical data such as smoking history, coughing, and allergies. It demonstrates how deep learning can be leveraged for early disease detection, reducing dependency on expensive imaging methods like CT scans and X-rays.

>>> Overview

Early detection of lung cancer plays a crucial role in improving patient survival rates. Traditional detection methods rely heavily on medical imaging and biopsies, which are costly and time-consuming.
This project proposes a data-driven, affordable, and accessible approach using deep learning models trained on tabular patient data to predict lung cancer risk accurately.

>>>Features

Input Module: Accepts patient history and clinical symptoms as input.

Preprocessing Module: Handles data cleaning, encoding, normalization, and scaling.

Model Module: Trains a Feedforward Neural Network (FNN) using PyTorch.

Prediction Module: Predicts “Cancer” or “No Cancer” with a confidence score.

Result Module: Displays results clearly for healthcare professionals.

>>>System Architecture
Patient Data → Preprocessing (Encoding, Scaling) → Deep Learning Model (FNN)
        ↓
Prediction (Cancer / No Cancer + Confidence Score)
        ↓
Doctor’s Decision Support (Further Diagnosis / Treatment)

>>>Technologies Used
Category	Tools
Programming Language	Python 3.8+
Libraries	PyTorch, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
IDE	Jupyter Notebook / VS Code
Hardware (Recommended)	8GB+ RAM, GPU (GTX 1650 or higher)
>>>How It Works

Data Input: Collects synthetic patient data (age, smoking habits, symptoms, etc.).

Data Preprocessing: Normalizes and encodes data for model compatibility.

Model Training: FNN model with 3 hidden layers trained using Binary Cross-Entropy loss.

Prediction: Generates a binary output (0 – Cancer, 1 – No Cancer) with confidence score.

Evaluation: Metrics such as accuracy, confusion matrix, and BCE loss are used.

>>>Results

Achieved high accuracy on synthetic datasets.

Outperformed traditional machine learning models.

Demonstrated strong generalization using Optuna for hyperparameter tuning.

>>>Limitations

Trained on synthetic data, not real patient records.

Performs binary classification only (no cancer / cancer).

Excludes genetic or environmental factors.

Lacks clinical validation.

>>>Future Scope

Integrate real-world medical datasets for improved accuracy.

Extend to multi-class classification (cancer stages/types).

Combine tabular + imaging data (CT scans, X-rays).   
Develop a mobile app for remote screening.

Integrate with Electronic Health Record (EHR) systems.                                                            
