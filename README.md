# LSTM Time Series Forecasting for Order Predictions

Forecasts daily order volumes using two LSTM architectures. The pipeline includes STL time series decomposition to extract trend and seasonal signals, MinMax scaling, and evaluation using MAE and MAPE on the final 31 days of data.

## Business Context

Accurate short-term order forecasting enables better inventory management, staffing, and marketing spend decisions. This project compares a simple single-layer LSTM against a deeper stacked architecture with dropout regularisation to assess the benefit of added model complexity.

## Dataset

`orders.csv` contains daily records with the following columns: `date`, `orders`, `marketing_investment`, `discount_rate`.

## Methodology

**EDA:** Line charts for orders, marketing investment, and discount rate over time.

**STL Decomposition:** The `orders` series is decomposed into trend, seasonal (period=365), and residual components. Trend and seasonal components are appended as additional features.

**Preprocessing:** MinMaxScaler applied to the full feature set. The target (`orders`) is column 0; remaining columns are features. Data is reshaped to `(samples, 1, features)` for LSTM input. The last 31 records form the test set.

**Models:**
- Simple LSTM: 50 units, ReLU activation, Dense output
- Advanced LSTM: 100-unit LSTM (return_sequences) + Dropout(0.2), 50-unit LSTM + Dropout(0.2), Dense output

**Evaluation:** Predictions are inverse-scaled and compared against actual orders using MAE and MAPE. Forecast vs actual plots are saved for each model.

## Project Structure

```
05_lstm_order_forecasting/
├── lstm_order_forecasting.py  # Full pipeline
├── requirements.txt
└── README.md
```

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
statsmodels
tensorflow
```

Install with:

```bash
pip install -r requirements.txt
```

## Usage

Place `orders.csv` in the same directory and run:

```bash
python lstm_order_forecasting.py
```

Outputs: `time_series_overview.png`, `stl_decomposition.png`, `forecast_simple_lstm.png`, `forecast_advanced_lstm.png`.
