import streamlit as st
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Dataset
data = pd.read_csv("lcs_synthetic_20000_augmented.csv")

if 'GENDER' in data.columns:
    data.drop(['GENDER'], axis=1, inplace=True)

for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = data[column].astype(str).str.strip()
        data[column] = data[column].map({'NO': 0, 'YES': 1})

data = data.dropna()

x = data.drop('LUNG_CANCER', axis=1).values.astype(np.float32)
data['LUNG_CANCER'] = data['LUNG_CANCER'].astype(str).str.strip()
data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'NO': 0, 'YES': 1})

y = data['LUNG_CANCER'].values.astype(np.float32)

scaler = StandardScaler()
x = scaler.fit_transform(x)

# Model class
class BinClr(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_size, 64)
        self.layer2 = torch.nn.Linear(64, 32)
        self.output = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.output(x)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create model
model = BinClr(x.shape[1]).to(device)

# Streamlit UI

st.title("Lung Cancer Prediction System")

age = st.number_input("Age", 1, 100, 30)

smoking = st.radio("Do you smoke?", [1, 2])

yellow_fingers = st.radio("Do you have yellow fingers?", [1, 2])

anxiety = st.radio("Do you have anxiety?", [1, 2])

peer_pressure = st.radio("Do you experience peer pressure?", [1, 2])

chronic_disease = st.radio("Do you have a chronic disease?", [1, 2])

fatigue = st.radio("Do you feel fatigue?", [1, 2])

allergies = st.radio("Do you have allergies?", [1, 2])

wheezing = st.radio("Do you have wheezing?", [1, 2])

alcohol = st.radio("Do you consume alcohol?", [1, 2])

coughing = st.radio("Do you cough frequently?", [1, 2])

shortness_breath = st.radio("Do you have shortness of breath?", [1, 2])

swallowing = st.radio("Do you have difficulty swallowing?", [1, 2])

chest_pain = st.radio("Do you have chest pain?", [1, 2])

if st.button("Predict"):

    sample_input = np.array([[
        age,
        smoking,
        yellow_fingers,
        anxiety,
        peer_pressure,
        chronic_disease,
        fatigue,
        allergies,
        wheezing,
        alcohol,
        coughing,
        shortness_breath,
        swallowing,
        chest_pain
    ]], dtype=np.float32)

    sample_input = scaler.transform(sample_input)

    sample_tensor = torch.from_numpy(sample_input).type(torch.float32).to(device)

    model.eval()

    with torch.no_grad():
        prediction = torch.sigmoid(model(sample_tensor)).item()

    if prediction >= 0.5:
        st.error("High Risk of Lung Cancer")
    else:
        st.success("Low Risk of Lung Cancer")
