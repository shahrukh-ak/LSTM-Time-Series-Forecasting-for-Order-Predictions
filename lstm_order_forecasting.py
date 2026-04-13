"""
LSTM Time Series Forecasting for Order Predictions
====================================================
Forecasts daily order volumes using two LSTM architectures.
The pipeline covers time series decomposition (STL), MinMax scaling,
sequence preparation, and evaluation with MAE and MAPE on the final
31 days of the dataset.

Dataset: orders.csv  (columns: date, orders, marketing_investment, discount_rate)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load orders CSV with a datetime index."""
    df = pd.read_csv(filepath, parse_dates=True, index_col=["date"])
    print(f"Shape: {df.shape}")
    print("\nSummary Statistics:")
    print(df.describe())
    return df


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_time_series(df: pd.DataFrame):
    """Plot orders, marketing investment, and discount rate over time."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(df.index, df["orders"])
    axes[0].set_title("Orders Over Time")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Orders")

    axes[1].plot(df.index, df["marketing_investment"], color="orange")
    axes[1].set_title("Marketing Investment Over Time")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Marketing Investment")

    axes[2].plot(df.index, df["discount_rate"], color="green")
    axes[2].set_title("Discount Rate Over Time")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Discount Rate")

    plt.tight_layout()
    plt.savefig("time_series_overview.png", dpi=150)
    plt.show()
    print("Saved: time_series_overview.png")


# ── STL Decomposition ─────────────────────────────────────────────────────────

def decompose_series(df: pd.DataFrame, period: int = 365) -> pd.DataFrame:
    """
    Apply STL decomposition to the orders series.
    Append trend and seasonal components to the dataframe.
    """
    stl = STL(df["orders"], period=period)
    result = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    for ax, data, title in zip(
        axes,
        [df["orders"], result.trend, result.seasonal, result.resid],
        ["Original", "Trend", "Seasonal", "Residual"],
    ):
        ax.plot(df.index, data, label=title)
        ax.legend(loc="upper left")
        ax.set_title(f"{title} Component")
    plt.tight_layout()
    plt.savefig("stl_decomposition.png", dpi=150)
    plt.show()
    print("Saved: stl_decomposition.png")

    df["trend"] = result.trend
    df["seasonal"] = result.seasonal
    df = df.dropna()
    return df


# ── Preprocessing for LSTM ────────────────────────────────────────────────────

def prepare_lstm_data(df: pd.DataFrame, test_days: int = 31):
    """
    Scale data with MinMaxScaler and split into train/test.
    Returns scaled arrays shaped for LSTM input.
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X = scaled[:, 1:]   # all columns except orders
    y = scaled[:, 0]    # orders column

    X = X.reshape((X.shape[0], 1, X.shape[1]))

    X_train, X_test = X[:-test_days], X[-test_days:]
    y_train, y_test = y[:-test_days], y[-test_days:]

    return X_train, X_test, y_train, y_test, scaler


# ── Model Architectures ───────────────────────────────────────────────────────

def build_simple_lstm(input_shape: tuple) -> Sequential:
    """Single LSTM layer with 50 units and a Dense output layer."""
    model = Sequential([
        LSTM(50, activation="relu", input_shape=input_shape),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mae")
    return model


def build_advanced_lstm(input_shape: tuple) -> Sequential:
    """
    Stacked LSTM architecture with dropout regularisation.
    Layer 1: 100 units, return_sequences=True, Dropout 0.2
    Layer 2: 50 units, Dropout 0.2
    """
    model = Sequential([
        LSTM(100, activation="relu", return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, activation="relu"),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mae")
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(model, X_train, y_train, epochs: int = 50, verbose: int = 0):
    """Fit the model and return the trained instance."""
    model.fit(X_train, y_train, epochs=epochs, verbose=verbose)
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_predictions(model, X_test, scaler, df, num_features: int, label: str = ""):
    """
    Generate predictions, reverse-scale them, and compute MAE and MAPE
    against the actual last N days of orders.
    """
    y_pred = model.predict(X_test)

    zero_matrix = np.zeros((y_pred.shape[0], num_features - 1))
    inverse_input = np.hstack((y_pred, zero_matrix))
    predicted_orders = scaler.inverse_transform(inverse_input)[:, 0]
    actual_orders = df["orders"].values[-X_test.shape[0]:]

    mae = mean_absolute_error(actual_orders, predicted_orders)
    mape = mean_absolute_percentage_error(actual_orders, predicted_orders)

    print(f"\n{label} Results:")
    print(f"  MAE  : {mae:.4f}")
    print(f"  MAPE : {mape:.4f}")
    return predicted_orders, actual_orders, mae, mape


def plot_forecast(actual, predicted, label: str = ""):
    """Compare actual vs predicted order volumes."""
    plt.figure(figsize=(10, 4))
    plt.plot(actual, label="Actual", marker="o")
    plt.plot(predicted, label="Predicted", marker="x")
    plt.title(f"{label} Order Forecast vs Actual")
    plt.xlabel("Day")
    plt.ylabel("Orders")
    plt.legend()
    plt.tight_layout()
    fname = f"forecast_{label.lower().replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = "orders.csv"
    TEST_DAYS = 31

    df = load_data(DATA_PATH)
    plot_time_series(df)
    df = decompose_series(df)

    X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(df, test_days=TEST_DAYS)
    num_features = df.shape[1]

    print("\nTraining Simple LSTM...")
    simple_model = build_simple_lstm(input_shape=(X_train.shape[1], X_train.shape[2]))
    simple_model = train_model(simple_model, X_train, y_train)
    preds_simple, actual, _, _ = evaluate_predictions(
        simple_model, X_test, scaler, df, num_features, label="Simple LSTM"
    )
    plot_forecast(actual, preds_simple, label="Simple LSTM")

    print("\nTraining Advanced LSTM (Stacked + Dropout)...")
    advanced_model = build_advanced_lstm(input_shape=(X_train.shape[1], X_train.shape[2]))
    advanced_model = train_model(advanced_model, X_train, y_train, epochs=50)
    preds_advanced, _, _, _ = evaluate_predictions(
        advanced_model, X_test, scaler, df, num_features, label="Advanced LSTM"
    )
    plot_forecast(actual, preds_advanced, label="Advanced LSTM")
