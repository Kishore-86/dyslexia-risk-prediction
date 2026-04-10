import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Load dataset
df = pd.read_csv("static/dataset/dataset.csv")

# Features
X = df[[
"age",
"reading_speed_wpm",
"reading_accuracy",
"spelling_error_rate",
"phoneme_error_rate",
"speech_fluency",
"handwriting_score"
]]

# Target
y = df["dyslexia_label"]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = MLPClassifier(hidden_layer_sizes=(50,30), max_iter=500)

model.fit(X_scaled,y)

# Save model
pickle.dump(model,open("dyslexia_model.pkl","wb"))

# Save scaler
pickle.dump(scaler,open("scaler.pkl","wb"))

print("Model and scaler saved successfully")
