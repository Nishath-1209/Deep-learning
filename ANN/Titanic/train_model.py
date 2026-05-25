import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("Titanic-Dataset.csv")

# =========================
# SELECT FEATURES
# =========================

X = df[['Pclass', 'Age', 'Fare']]

# Fill missing Age values
X['Age'].fillna(X['Age'].mean(), inplace=True)

y = df['Survived']

# =========================
# NORMALIZATION
# =========================

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# BUILD ANN MODEL
# =========================

model = Sequential()

# Hidden Layer
model.add(Dense(2, input_dim=3, activation='sigmoid'))

# Output Layer
model.add(Dense(1, activation='sigmoid'))

# =========================
# COMPILE MODEL
# =========================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# TRAIN MODEL
# =========================

model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=10
)

# =========================
# SAVE MODEL
# =========================

model.save("model/titanic_ann_model.h5")

print("Model Saved Successfully!")