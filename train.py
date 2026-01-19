import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# 1. Download Dataset from Internet
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "selva86/datasets/master/GermanCredit.csv"
)

print("Downloading dataset...")
df = pd.read_csv(DATA_URL)


# 2. Feature Engineering

df["income"] = df["CreditAmount"] / 1000
df["credit_score"] = 100 - df["Duration"]

# Target: 1 = Approved, 0 = Rejected
df["approved"] = df["Class"].map({1: 1, 2: 0})

X = df[["income", "credit_score"]]
y = df["approved"]


# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. Deep Learning Model
def build_model():
    model = Sequential([
        Dense(64, activation="relu", input_shape=(2,)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(32, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),

        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# 5. sklearn-Compatible Wrapper
dl_model = KerasClassifier(
    model=build_model,
    epochs=60,
    batch_size=32,
    verbose=0,
    validation_split=0.1,
    callbacks=[
        EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True
        )
    ]
)


# 6. Pipeline (Scaler + DL Model)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", dl_model)
])


# 7. Train Model
print("Training deep learning model...")
pipeline.fit(X_train, y_train)


# 8. Evaluation
accuracy = pipeline.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")


# 9. Save Model (FastAPI compatible)
joblib.dump(pipeline, "loan_model.pkl")
print("Model saved as loan_model.pkl")
