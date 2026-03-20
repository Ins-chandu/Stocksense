from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.config import TOP_BSE_STOCKS
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
)
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    concatenate,
    Bidirectional,
)
import os
import joblib
from imblearn.over_sampling import SVMSMOTE
from app.data_loader import fetch_data

app = FastAPI()

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

short_window = 30
long_window = 120

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or replace with 'http://localhost:5173' for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    symbol: str
    retrain: bool = False
    threshold: float = 0.1
    future_days: int = 3


def create_sequences(data, threshold=0.1, future_days=3):
    X_short, X_long, y = [], [], []
    for i in range(len(data) - long_window - future_days):
        X_short.append(data[i + long_window - short_window : i + long_window])
        X_long.append(data[i : i + long_window])
        p0 = data[i + long_window - 1][0]
        future_avg = np.mean([data[i + long_window + j][0] for j in range(future_days)])
        if p0 == 0:
            continue
        delta = (future_avg - p0) / p0
        y.append(1 if delta > threshold else 0)
    min_len = min(len(X_short), len(X_long), len(y))
    return (
        np.array(X_short[:min_len]),
        np.array(X_long[:min_len]),
        np.array(y[:min_len]),
    )


def build_model(input_short_shape, input_long_shape):
    input_short = Input(shape=input_short_shape)
    input_long = Input(shape=input_long_shape)
    x1 = Bidirectional(LSTM(64))(input_short)
    x2 = Bidirectional(LSTM(64))(input_long)
    x = concatenate([x1, x2])
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=[input_short, input_long], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


@app.get("/")
def read_root():
    return {"message": "📈 Welcome to Stock Movement Prediction API"}


@app.get("/stocks")
def get_stocks():
    return {"stocks": TOP_BSE_STOCKS}


@app.get("/stocks/{stock_id}")
def get_stock_info(stock_id: str):
    name = TOP_BSE_STOCKS.get(stock_id, "Unknown")
    try:
        df = fetch_data(stock_id)
    except ValueError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    data = df.tail(60).reset_index().to_dict(orient="records")
    return {"name": name, "data": data}


def process_prediction(
    symbol: str, retrain: bool, threshold: float, future_days: int
) -> Dict[str, Any]:
    try:
        df = fetch_data(symbol)
    except ValueError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    features = [
        "Close",
        "MACD",
        "MACD_diff",
        "RSI",
        "SMA20",
        "SMA200",
        "Volume",
        "RET1",
        "VOL",
    ]
    data = df[features].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    scaler_path = os.path.join(MODEL_DIR, f"{symbol.replace('.', '_')}_scaler.pkl")
    if retrain or not os.path.exists(scaler_path):
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        scaled_data = scaler.transform(data)

    X_short, X_long, y = create_sequences(
        scaled_data, threshold=threshold, future_days=future_days
    )
    if len(y) == 0:
        return {"error": "Not enough data to create sequences for prediction."}

    Xs = X_short.reshape(X_short.shape[0], short_window, len(features))
    Xl = X_long.reshape(X_long.shape[0], long_window, len(features))
    Xs_flat = Xs.reshape(Xs.shape[0], -1)
    Xl_flat = Xl.reshape(Xl.shape[0], -1)

    smote = SVMSMOTE(random_state=42)
    Xs_flat_res, y_resampled = smote.fit_resample(Xs_flat, y)
    Xl_flat_res, _ = smote.fit_resample(Xl_flat, y)
    Xs = Xs_flat_res.reshape(-1, short_window, len(features))
    Xl = Xl_flat_res.reshape(-1, long_window, len(features))
    y_resampled = y_resampled[: min(len(Xs), len(Xl))]
    Xs = Xs[: len(y_resampled)]
    Xl = Xl[: len(y_resampled)]

    Xs_train, Xs_test, Xl_train, Xl_test, y_train, y_test = train_test_split(
        Xs, Xl, y_resampled, test_size=0.2, shuffle=True, random_state=42
    )

    model_path = os.path.join(MODEL_DIR, f"{symbol.replace('.', '_')}.h5")
    if os.path.exists(model_path) and not retrain:
        try:
            model = load_model(model_path)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Pre-trained model is incompatible with the current TensorFlow/Keras version. "
                    "Please retry with retrain=true."
                ),
            ) from exc
    elif retrain:
        model = build_model((short_window, len(features)), (long_window, len(features)))
        class_weights = compute_class_weight(
            "balanced", classes=np.unique(y_train), y=y_train
        )
        class_weights = {i: class_weights[i] for i in range(len(class_weights))}
        model.fit(
            [Xs_train, Xl_train],
            y_train,
            epochs=20,
            batch_size=32,
            validation_data=([Xs_test, Xl_test], y_test),
            class_weight=class_weights,
            verbose=0,
        )
        model.save(model_path)
    else:
        return {
            "error": f"Model not found for symbol {symbol}. Please call this endpoint with `retrain=true` to train a new model."
        }

    # Metrics
    y_probs = model.predict([Xs_test, Xl_test])
    y_pred = (y_probs > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Next day prediction (from latest data)
    last_short = scaled_data[-short_window:].reshape(1, short_window, len(features))
    last_long = scaled_data[-long_window:].reshape(1, long_window, len(features))
    next_prob = float(model.predict([last_short, last_long])[0][0])
    next_percent = round(next_prob * 100, 2)

    if 0.48 <= next_prob <= 0.52:
        direction = "UNCERTAIN"
        probability = next_percent
        explanation = "🤔 Low confidence prediction — proceed with caution."
    else:
        direction = "UP" if next_prob > 0.5 else "DOWN"
        probability = next_percent if direction == "UP" else 100 - next_percent
        explanation = (
            "📈 Expected to go UP!" if direction == "UP" else "📉 Expected to go DOWN!"
        )

    return {
        "accuracy": round(acc, 4),
        "roc_auc": round(roc, 4),
        "pr_auc": round(pr_auc, 4),
        "classification_report": report,
        "next_day_prediction": {
            "direction": direction,
            "probability_percent": probability,
            "explanation": explanation,
        },
    }


@app.post("/predict")
def predict_post(request: PredictRequest):
    return process_prediction(
        symbol=request.symbol,
        retrain=request.retrain,
        threshold=request.threshold,
        future_days=request.future_days,
    )


@app.get("/predict")
def predict_get(
    symbol: str, retrain: bool = False, threshold: float = 0.02, future_days: int = 3
):
    return process_prediction(
        symbol=symbol, retrain=retrain, threshold=threshold, future_days=future_days
    )
