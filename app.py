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

if st.button("Predict"):
    st.success("Prediction Generated Successfully")
