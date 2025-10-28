import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import optuna

# Load and preprocess data
data = pd.read_csv("lcs_synthetic_20000_augmented.csv")
data.drop(['GENDER'], axis=1, inplace=True)
x = data.drop('LUNG_CANCER', axis=1)
y = data['LUNG_CANCER'].values

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define dataset class
class TabularData(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).type(torch.float32)
        self.y = torch.from_numpy(y).type(torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Define model class
class BinClr(torch.nn.Module):
    def __init__(self, input_size, hidden1, hidden2, dropout_rate):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden1)
        self.layer2 = torch.nn.Linear(hidden1, hidden2)
        self.output = torch.nn.Linear(hidden2, 1)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        return self.output(x)

# Optuna objective function

def objective(trial):
    hidden1 = trial.suggest_int("hidden1", 32, 128)
    hidden2 = trial.suggest_int("hidden2", 16, 64)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    dropout_rate = trial.suggest_float("dropout", 0.0, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    model = BinClr(input_size=14, hidden1=hidden1, hidden2=hidden2, dropout_rate=dropout_rate).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_data = TabularData(x_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    for epoch in range(10):  # Keep epochs low for tuning speed
        model.train()
        for features, target in train_loader:
            features, target = features.to(device), target.to(device)
            y_pred = model(features).squeeze()
            loss = loss_fn(y_pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(torch.from_numpy(x_test).type(torch.float32).to(device))).cpu().numpy()
        preds = (preds >= 0.5).astype(int).squeeze()
        acc = accuracy_score(y_test, preds)
    return acc

# Run Optuna study
device = 'cuda' if torch.cuda.is_available() else 'cpu'
study = optuna.create_study(direction="maximize")
study.optimize(objective,n_trials=20)

print("Best trial:", study.best_trial.params)

# Train best model
best_params = study.best_trial.params
model = BinClr(14, best_params['hidden1'], best_params['hidden2'], best_params['dropout']).to(device)
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'])
train_data = TabularData(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=best_params['batch_size'], shuffle=True)

# Train final model
for epoch in range(15):
    model.train()
    temp_loss = []
    for features, target in train_loader:
        features, target = features.to(device), target.to(device)
        y_pred = model(features).squeeze()
        loss = loss_fn(y_pred, target)
        temp_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} | loss: {np.mean(temp_loss):.4f}")

# Evaluate
model.eval()
y_pred = torch.sigmoid(model(torch.from_numpy(x_test).type(torch.float32).to(device))).cpu().numpy()
y_pred = (y_pred >= 0.5).astype(int).squeeze()
score = 100 * accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(score))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
plt.show()

def predict_lung_cancer(age, smoke, yellow_fingers, anxiety, peer_pressure, chronic_disease,
                        fatigue, allergies, wheezing, alcohol_consumption, coughing,
                        shortness_of_breath, swallowing_difficulty, chest_pain):
    try:
        user_input = np.array([[age, smoke, yellow_fingers, anxiety, peer_pressure, chronic_disease,
                                fatigue, allergies, wheezing, alcohol_consumption, coughing,
                                shortness_of_breath, swallowing_difficulty, chest_pain]])
        user_input_scaled = scaler.transform(user_input)
        user_input_tensor = torch.from_numpy(user_input_scaled).type(torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            prediction = torch.sigmoid(model(user_input_tensor)).cpu().numpy()[0][0]
        result = "Not suffering from lung cancer" if prediction >= 0.5 else "Suffering from lung cancer"
        return f"Prediction: {result} (Confidence: {prediction:.2f})"
    except Exception as e:
        return f"Error: {e}"

inputs = [
    gr.Number(label="Age"),
    gr.Radio(choices=[1, 2], label="Do you smoke? (1=No, 2=Yes)"),
    gr.Radio(choices=[1, 2], label="Do you have yellow fingers?"),
    gr.Radio(choices=[1, 2], label="Do you have anxiety?"),
    gr.Radio(choices=[1, 2], label="Do you experience peer pressure?"),
    gr.Radio(choices=[1, 2], label="Do you have a chronic disease?"),
    gr.Radio(choices=[1, 2], label="Do you feel fatigue?"),
    gr.Radio(choices=[1, 2], label="Do you have allergies?"),
    gr.Radio(choices=[1, 2], label="Do you have wheezing?"),
    gr.Radio(choices=[1, 2], label="Do you consume alcohol?"),
    gr.Radio(choices=[1, 2], label="Do you cough frequently?"),
    gr.Radio(choices=[1, 2], label="Do you have shortness of breath?"),
    gr.Radio(choices=[1, 2], label="Do you have difficulty swallowing?"),
    gr.Radio(choices=[1, 2], label="Do you have chest pain?")
]

gr.Interface(fn=predict_lung_cancer, inputs=inputs, outputs="text", title="Lung Cancer Predictor").launch()












