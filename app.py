import streamlit as st

# Model setup
model = BinClr(x.shape[1]).to(device)
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# DataLoader
train_data = TabularData(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Quick training
for epoch in range(3):
    model.train()
    for features, target in train_loader:
        features, target = features.to(device), target.to(device)

        y_pred = model(features).squeeze()
        loss = loss_fn(y_pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Streamlit UI
st.title("Lung Cancer Prediction System")

age = st.number_input("Age", 1, 100, 30)
smoke = st.selectbox("Smoking", [1, 2])
yellow_fingers = st.selectbox("Yellow Fingers", [1, 2])
anxiety = st.selectbox("Anxiety", [1, 2])
peer_pressure = st.selectbox("Peer Pressure", [1, 2])
chronic_disease = st.selectbox("Chronic Disease", [1, 2])
fatigue = st.selectbox("Fatigue", [1, 2])
allergies = st.selectbox("Allergies", [1, 2])
wheezing = st.selectbox("Wheezing", [1, 2])
alcohol_consumption = st.selectbox("Alcohol Consumption", [1, 2])
coughing = st.selectbox("Coughing", [1, 2])
shortness_of_breath = st.selectbox("Shortness of Breath", [1, 2])
swallowing_difficulty = st.selectbox("Swallowing Difficulty", [1, 2])
chest_pain = st.selectbox("Chest Pain", [1, 2])

if st.button("Predict"):
    user_input = np.array([[age, smoke, yellow_fingers, anxiety,
                            peer_pressure, chronic_disease,
                            fatigue, allergies, wheezing,
                            alcohol_consumption, coughing,
                            shortness_of_breath,
                            swallowing_difficulty,
                            chest_pain]])

    user_input_scaled = scaler.transform(user_input)

    user_input_tensor = torch.from_numpy(user_input_scaled).type(torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        prediction = torch.sigmoid(model(user_input_tensor)).item()

    if prediction >= 0.5:
        st.error("High Risk of Lung Cancer")
    else:
        st.success("Low Risk of Lung Cancer")
