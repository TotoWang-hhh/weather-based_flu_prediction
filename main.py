import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "weather-and-sickness"


def load_weather_data(dataset_dir: pathlib.Path) -> pd.DataFrame:
    weather_frames = []
    for file_path in sorted(dataset_dir.glob("weather_*.csv")):
        match = re.fullmatch(r"weather_(\d{4}).csv", file_path.name)
        if not match:
            continue

        df = pd.read_csv(file_path)
        df["date"] = pd.to_datetime(
            {"year": df["year"], "month": df["month"], "day": df["day"]},
            errors="coerce",
        )
        df = df.dropna(subset=["date"]).copy()
        df["week"] = df["date"].dt.isocalendar().week
        df["year"] = df["date"].dt.year
        weather_frames.append(df)

    if not weather_frames:
        raise FileNotFoundError(f"No weather files found in {dataset_dir}")

    return pd.concat(weather_frames, ignore_index=True)


def load_influenza_data(dataset_dir: pathlib.Path) -> pd.DataFrame:
    influenza = pd.read_csv(dataset_dir / "influenza.csv", encoding="latin-1")

    influenza = influenza.rename(columns={"Neuerkrankungen pro Woche": "new_cases"})
    influenza["new_cases"] = pd.to_numeric(influenza["new_cases"], errors="coerce")
    influenza["year"] = pd.to_numeric(influenza["Jahr"], errors="coerce")
    influenza["week"] = (
        influenza["Kalenderwoche"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(float)
    )

    return influenza[["year", "week", "new_cases"]].dropna().reset_index(drop=True)


def clean_outliers_with_iqr(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    cleaned = df.copy()
    for column in columns:
        if column not in cleaned.columns:
            continue
        series = pd.to_numeric(cleaned[column], errors="coerce")
        series = series.fillna(series.median())
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        cleaned[column] = series.clip(lower=lower, upper=upper)
    return cleaned


def build_weekly_weather_features(weather: pd.DataFrame) -> pd.DataFrame:
    features = weather.groupby(["year", "week"], as_index=False).agg(
        temp_mean=("temp_dailyMean", "mean"),
        temp_min=("temp_dailyMin", "mean"),
        temp_max=("temp_dailyMax", "mean"),
        humidity_mean=("hum_dailyMean", "mean"),
        humidity_7h=("hum_7h", "mean"),
        wind_mean=("wind_mSec", "mean"),
        precip_sum=("precip", "sum"),
        precip_mean=("precip", "mean"),
        sun_hours=("sun_hours", "mean"),
    )
    return features


def create_dataset(influenza: pd.DataFrame, weather_features: pd.DataFrame) -> pd.DataFrame:
    merged = influenza.merge(weather_features, on=["year", "week"], how="inner")
    merged = merged.sort_values(["year", "week"]).reset_index(drop=True)

    weather_columns = [
        "temp_mean",
        "temp_min",
        "temp_max",
        "humidity_mean",
        "humidity_7h",
        "wind_mean",
        "precip_sum",
        "precip_mean",
        "sun_hours",
    ]
    merged[weather_columns] = merged[weather_columns].apply(pd.to_numeric, errors="coerce")
    merged[weather_columns] = merged[weather_columns].interpolate(method="linear", limit_direction="both")
    merged = clean_outliers_with_iqr(merged, weather_columns)

    for lag in [1, 2, 3]:
        for column in weather_columns:
            merged[f"{column}_lag_{lag}"] = merged[column].shift(lag)

    for lag in [1, 2, 3]:
        merged[f"temp_mean_change_{lag}"] = merged["temp_mean"].diff(lag)
        merged[f"temp_max_change_{lag}"] = merged["temp_max"].diff(lag)
        merged[f"temp_min_change_{lag}"] = merged["temp_min"].diff(lag)

    merged = merged.dropna().reset_index(drop=True)
    return merged


def create_sequence_dataset(dataset: pd.DataFrame, history_steps: int = 4) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sequence_features = []
    targets = []
    metadata = []

    feature_columns = [
        col for col in dataset.columns if col not in {"year", "week", "new_cases"}
    ]

    for i in range(history_steps, len(dataset)):
        window = dataset.iloc[i - history_steps : i][feature_columns].to_numpy(dtype=np.float32)
        target = dataset.iloc[i]["new_cases"]
        sequence_features.append(window)
        targets.append(target)
        metadata.append(dataset.iloc[i][["year", "week", "new_cases"]].to_dict())

    X = np.array(sequence_features, dtype=np.float32)
    y = np.array(targets, dtype=np.float32).reshape(-1, 1)
    meta = pd.DataFrame(metadata)
    return X, y, meta


def train_history_weather_lstm(dataset: pd.DataFrame, history_steps: int = 4) -> dict:
    feature_columns = [
        col for col in dataset.columns if col not in {"year", "week", "new_cases"}
    ]

    X_seq, y_seq, meta_seq = create_sequence_dataset(dataset, history_steps=history_steps)
    if len(X_seq) < history_steps + 2:
        return {
            "metrics": {"mae": np.nan, "rmse": np.nan, "r2": np.nan},
            "predictions": np.array([]),
            "actual": np.array([]),
            "test_rows": pd.DataFrame(columns=["year", "week", "new_cases"]),
            "feature_columns": feature_columns,
            "model": None,
        }

    split_idx = int(len(X_seq) * 0.8)
    X_seq_train, X_seq_test = X_seq[:split_idx], X_seq[split_idx:]
    y_seq_train, y_seq_test = y_seq[:split_idx], y_seq[split_idx:]
    meta_seq_test = meta_seq.iloc[split_idx:].reset_index(drop=True)

    feature_scaler = StandardScaler()
    X_seq_train_scaled = feature_scaler.fit_transform(X_seq_train.reshape(-1, X_seq_train.shape[-1])).reshape(X_seq_train.shape)
    X_seq_test_scaled = feature_scaler.transform(X_seq_test.reshape(-1, X_seq_test.shape[-1])).reshape(X_seq_test.shape)

    target_scaler = StandardScaler()
    y_seq_train_scaled = target_scaler.fit_transform(y_seq_train).reshape(-1)

    class HistoryWeatherLSTM(nn.Module):
        def __init__(self, input_size: int, hidden_size: int = 24):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    torch.manual_seed(42)
    model = HistoryWeatherLSTM(input_size=X_seq_train_scaled.shape[-1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    X_seq_train_torch = torch.tensor(X_seq_train_scaled, dtype=torch.float32)
    y_seq_train_torch = torch.tensor(y_seq_train_scaled, dtype=torch.float32).reshape(-1, 1)
    X_seq_test_torch = torch.tensor(X_seq_test_scaled, dtype=torch.float32)

    for epoch in range(250):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_seq_train_torch)
        loss = criterion(outputs, y_seq_train_torch)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        scaled_predictions = model(X_seq_test_torch).cpu().numpy().reshape(-1)

    predictions = target_scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).reshape(-1)
    actual = y_seq_test.reshape(-1)

    return {
        "metrics": {
            "mae": mean_absolute_error(actual, predictions),
            "rmse": np.sqrt(mean_squared_error(actual, predictions)),
            "r2": r2_score(actual, predictions),
        },
        "predictions": predictions,
        "actual": actual,
        "test_rows": meta_seq_test[["year", "week", "new_cases"]].copy().reset_index(drop=True),
        "feature_columns": feature_columns,
        "model": model,
    }


def train_notebook_inspired_lstm(dataset: pd.DataFrame, history_steps: int = 4) -> dict:
    feature_columns = [
        col for col in dataset.columns if col not in {"year", "week", "new_cases"}
    ]

    X_seq, y_seq, meta_seq = create_sequence_dataset(dataset, history_steps=history_steps)
    if len(X_seq) < history_steps + 2:
        return {
            "metrics": {"mae": np.nan, "rmse": np.nan, "r2": np.nan},
            "predictions": np.array([]),
            "actual": np.array([]),
            "test_rows": pd.DataFrame(columns=["year", "week", "new_cases"]),
            "feature_columns": feature_columns,
            "model": None,
        }

    split_idx = int(len(X_seq) * 0.8)
    X_seq_train, X_seq_test = X_seq[:split_idx], X_seq[split_idx:]
    y_seq_train, y_seq_test = y_seq[:split_idx], y_seq[split_idx:]
    meta_seq_test = meta_seq.iloc[split_idx:].reset_index(drop=True)

    feature_scaler = MinMaxScaler()
    X_seq_train_scaled = feature_scaler.fit_transform(X_seq_train.reshape(-1, X_seq_train.shape[-1])).reshape(X_seq_train.shape)
    X_seq_test_scaled = feature_scaler.transform(X_seq_test.reshape(-1, X_seq_test.shape[-1])).reshape(X_seq_test.shape)

    target_scaler = MinMaxScaler()
    y_seq_train_scaled = target_scaler.fit_transform(y_seq_train).reshape(-1)

    class NotebookInspiredLSTM(nn.Module):
        def __init__(self, input_size: int, hidden_size: int = 48):
            super().__init__()
            self.lstm1 = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=2,
                batch_first=True,
                dropout=0.2,
            )
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm1(x)
            out = self.dropout(out[:, -1, :])
            return self.fc(out)

    torch.manual_seed(42)
    model = NotebookInspiredLSTM(input_size=X_seq_train_scaled.shape[-1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    X_seq_train_torch = torch.tensor(X_seq_train_scaled, dtype=torch.float32)
    y_seq_train_torch = torch.tensor(y_seq_train_scaled, dtype=torch.float32).reshape(-1, 1)
    X_seq_test_torch = torch.tensor(X_seq_test_scaled, dtype=torch.float32)

    for epoch in range(220):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_seq_train_torch)
        loss = criterion(outputs, y_seq_train_torch)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        scaled_predictions = model(X_seq_test_torch).cpu().numpy().reshape(-1)

    predictions = target_scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).reshape(-1)
    actual = y_seq_test.reshape(-1)

    return {
        "metrics": {
            "mae": mean_absolute_error(actual, predictions),
            "rmse": np.sqrt(mean_squared_error(actual, predictions)),
            "r2": r2_score(actual, predictions),
        },
        "predictions": predictions,
        "actual": actual,
        "test_rows": meta_seq_test[["year", "week", "new_cases"]].copy().reset_index(drop=True),
        "feature_columns": feature_columns,
        "model": model,
    }


def train_stacking_meta_model(dataset: pd.DataFrame) -> dict:
    feature_columns = [
        col for col in dataset.columns if col not in {"year", "week", "new_cases"}
    ]

    split_index = int(len(dataset) * 0.8)
    train = dataset.iloc[:split_index]
    test = dataset.iloc[split_index:]

    if len(train) < 10 or len(test) < 5:
        return {
            "metrics": {"mae": np.nan, "rmse": np.nan, "r2": np.nan},
            "predictions": np.array([]),
            "actual": np.array([]),
            "test_rows": pd.DataFrame(columns=["year", "week", "new_cases"]),
            "feature_columns": feature_columns,
            "model": None,
        }

    inner_split = int(len(train) * 0.8)
    inner_train = train.iloc[:inner_split]
    meta_val = train.iloc[inner_split:]

    X_inner_train = inner_train[feature_columns]
    y_inner_train = inner_train["new_cases"]
    X_meta_val = meta_val[feature_columns]
    y_meta_val = meta_val["new_cases"]

    X_test = test[feature_columns]
    y_test = test["new_cases"]

    base_models = {
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
            ]
        ),
    }

    meta_features_train = []
    meta_features_val = []
    for name, model in base_models.items():
        model.fit(X_inner_train, y_inner_train)
        train_pred = model.predict(X_meta_val)
        meta_features_train.append(train_pred)

    meta_features_train = np.column_stack(meta_features_train)
    meta_model = LinearRegression()
    meta_model.fit(meta_features_train, y_meta_val.to_numpy())

    final_base_predictions = []
    for model in base_models.values():
        model.fit(train[feature_columns], train["new_cases"])
        final_base_predictions.append(model.predict(X_test))

    meta_features_test = np.column_stack(final_base_predictions)
    predictions = meta_model.predict(meta_features_test)
    actual = y_test.to_numpy()

    return {
        "metrics": {
            "mae": mean_absolute_error(actual, predictions),
            "rmse": np.sqrt(mean_squared_error(actual, predictions)),
            "r2": r2_score(actual, predictions),
        },
        "predictions": predictions,
        "actual": actual,
        "test_rows": test[["year", "week", "new_cases"]].copy().reset_index(drop=True),
        "feature_columns": feature_columns,
        "model": meta_model,
    }


def train_models(dataset: pd.DataFrame):
    feature_columns = [
        col for col in dataset.columns if col not in {"year", "week", "new_cases"}
    ]

    split_index = int(len(dataset) * 0.8)
    train = dataset.iloc[:split_index]
    test = dataset.iloc[split_index:]

    X_train = train[feature_columns]
    y_train = train["new_cases"]
    X_test = test[feature_columns]
    y_test = test["new_cases"]

    models = {
        "linear_regression": Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("model", LinearRegression())]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
            ]
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        results[name] = {
            "metrics": {
                "mae": mean_absolute_error(y_test, predictions),
                "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
                "r2": r2_score(y_test, predictions),
            },
            "predictions": predictions,
            "actual": y_test.to_numpy(),
            "test_rows": test[["year", "week", "new_cases"]].copy().reset_index(drop=True),
            "feature_columns": feature_columns,
            "model": model,
        }

    X_train_seq = X_train.to_numpy(dtype=np.float32)
    y_train_seq = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)
    X_test_seq = X_test.to_numpy(dtype=np.float32)
    y_test_seq = y_test.to_numpy(dtype=np.float32).reshape(-1, 1)

    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train_seq)
    X_test_scaled = feature_scaler.transform(X_test_seq)

    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train_seq).reshape(-1)
    y_test_scaled = target_scaler.transform(y_test_seq).reshape(-1)

    class LSTMRegressor(nn.Module):
        def __init__(self, input_size: int, hidden_size: int = 24):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    if len(X_train_scaled) >= 4:
        seq_len = 4
        X_train_windows = []
        y_train_windows = []
        for i in range(len(X_train_scaled) - seq_len + 1):
            X_train_windows.append(X_train_scaled[i : i + seq_len])
            y_train_windows.append(y_train_scaled[i + seq_len - 1])

        X_test_windows = []
        y_test_windows = []
        for i in range(len(X_test_scaled) - seq_len + 1):
            X_test_windows.append(X_test_scaled[i : i + seq_len])
            y_test_windows.append(y_test_scaled[i + seq_len - 1])

        if len(X_train_windows) > 0 and len(X_test_windows) > 0:
            X_train_torch = torch.tensor(np.array(X_train_windows), dtype=torch.float32)
            y_train_torch = torch.tensor(np.array(y_train_windows), dtype=torch.float32).reshape(-1, 1)
            X_test_torch = torch.tensor(np.array(X_test_windows), dtype=torch.float32)
            y_test_torch = torch.tensor(np.array(y_test_windows), dtype=torch.float32).reshape(-1, 1)

            torch.manual_seed(42)
            model = LSTMRegressor(input_size=X_train_torch.shape[2])
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

            for epoch in range(250):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_torch)
                loss = criterion(outputs, y_train_torch)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                scaled_predictions = model(X_test_torch).cpu().numpy().reshape(-1)
                scaled_actual = y_test_torch.cpu().numpy().reshape(-1)

            predictions = target_scaler.inverse_transform(scaled_predictions.reshape(-1, 1)).reshape(-1)
            actual = target_scaler.inverse_transform(scaled_actual.reshape(-1, 1)).reshape(-1)

            results["lstm"] = {
                "metrics": {
                    "mae": mean_absolute_error(actual, predictions),
                    "rmse": np.sqrt(mean_squared_error(actual, predictions)),
                    "r2": r2_score(actual, predictions),
                },
                "predictions": predictions,
                "actual": actual,
                "test_rows": test.iloc[seq_len - 1 :].reset_index(drop=True)[["year", "week", "new_cases"]].copy(),
                "feature_columns": feature_columns,
                "model": model,
            }

    results["history_weather_lstm"] = train_history_weather_lstm(dataset)
    results["notebook_inspired_lstm"] = train_notebook_inspired_lstm(dataset)
    results["stacking_meta_model"] = train_stacking_meta_model(dataset)

    return results


def create_ensemble_result(results: dict) -> dict:
    base_models = [
        name for name in ["random_forest", "lstm", "history_weather_lstm", "notebook_inspired_lstm"]
        if name in results and len(results[name].get("predictions", [])) > 0
    ]
    if len(base_models) < 2:
        return {}

    def _metric_weight(payload: dict) -> float:
        mae = payload.get("metrics", {}).get("mae")
        if not np.isfinite(mae) or mae <= 0:
            return 1.0
        return 1.0 / float(mae)

    weights = {name: _metric_weight(results[name]) for name in base_models}
    total_weight = sum(weights.values())
    weights = {name: weight / total_weight for name, weight in weights.items()}

    min_len = min(len(results[name].get("predictions", [])) for name in base_models)
    if min_len < 2:
        return {}

    aligned_rows = []
    predictions_by_model = {}
    actuals = []
    for name in base_models:
        payload = results[name]
        rows = payload.get("test_rows", pd.DataFrame(columns=["year", "week", "new_cases"]))
        if len(rows) < min_len:
            rows = rows.iloc[:min_len]
        rows = rows.iloc[:min_len].copy().reset_index(drop=True)
        predictions = np.asarray(payload.get("predictions", []), dtype=float)[:min_len]
        actual = np.asarray(payload.get("actual", []), dtype=float)[:min_len]
        aligned_rows.append(rows)
        predictions_by_model[name] = predictions
        actuals.append(actual)

    common_rows = aligned_rows[0].copy().reset_index(drop=True)
    common_rows = common_rows.sort_values(["year", "week"]).reset_index(drop=True)
    common_rows["new_cases"] = actuals[0]

    ensemble_predictions = np.zeros(min_len, dtype=float)
    for name in base_models:
        ensemble_predictions += predictions_by_model[name] * weights[name]

    actual_values = common_rows["new_cases"].to_numpy(dtype=float)
    ensemble_metrics = {
        "mae": mean_absolute_error(actual_values, ensemble_predictions),
        "rmse": np.sqrt(mean_squared_error(actual_values, ensemble_predictions)),
        "r2": r2_score(actual_values, ensemble_predictions),
    }

    return {
        "metrics": ensemble_metrics,
        "predictions": ensemble_predictions,
        "actual": actual_values,
        "test_rows": common_rows[["year", "week", "new_cases"]].copy().reset_index(drop=True),
        "feature_columns": [],
        "model": None,
    }


def save_tables_and_plots(results: dict, output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    prediction_frames = []
    for name, payload in results.items():
        metrics_rows.append({"model": name, **payload["metrics"]})

        pred_df = payload["test_rows"].copy()
        pred_df["model"] = name
        pred_df["actual_new_cases"] = payload["actual"]
        pred_df["predicted_new_cases"] = payload["predictions"]
        pred_df["error"] = pred_df["predicted_new_cases"] - pred_df["actual_new_cases"]
        prediction_frames.append(pred_df)

    metrics_df = pd.DataFrame(metrics_rows).set_index("model")
    metrics_df.to_csv(output_dir / "model_metrics.csv")

    comparison_predictions = pd.concat(prediction_frames, ignore_index=True)
    comparison_predictions.to_csv(output_dir / "comparison_predictions.csv", index=False)

    max_len = max(len(payload["actual"]) for payload in results.values())
    actual_series = pd.Series(results["linear_regression"]["actual"], index=np.arange(len(results["linear_regression"]["actual"])))
    actual_padded = pd.Series(np.nan, index=np.arange(max_len), dtype=float)
    actual_padded.iloc[: len(actual_series)] = actual_series.to_numpy()

    plt.figure(figsize=(12, 6))
    plt.plot(actual_padded.index, actual_padded, label="Actual", color="black", linewidth=2)
    for name, payload in results.items():
        preds = pd.Series(payload["predictions"], index=np.arange(len(payload["predictions"])))
        padded_preds = pd.Series(np.nan, index=np.arange(max_len), dtype=float)
        padded_preds.iloc[: len(preds)] = preds.to_numpy()
        plt.plot(padded_preds.index, padded_preds, marker="o", linewidth=1.2, label=f"{name} predicted")
    plt.xlabel("Test index")
    plt.ylabel("New cases")
    plt.title("Comparison of all tested models")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / "all_models_comparison.png", dpi=200)
    plt.close()

    time_index = np.arange(len(results["linear_regression"]["actual"]))
    time_labels = [f"{row['year']}-{int(row['week'])}" for _, row in results["linear_regression"]["test_rows"].iterrows()]
    plt.figure(figsize=(14, 6))
    plt.plot(time_index, results["linear_regression"]["actual"], label="Actual", color="black", linewidth=2.2)
    for name, payload in results.items():
        plt.plot(time_index[: len(payload["predictions"])], payload["predictions"], marker="o", linewidth=1.2, label=f"{name}")
    plt.xticks(time_index[:: max(1, len(time_index) // 10)], time_labels[:: max(1, len(time_labels) // 10)], rotation=45, ha="right")
    plt.xlabel("Year-Week")
    plt.ylabel("New cases")
    plt.title("Actual vs predicted influenza cases by year-week")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / "time_series_comparison.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    ax = plt.gca()
    metrics_df[["mae", "rmse"]].plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518"])
    plt.title("MAE and RMSE comparison")
    plt.ylabel("Error")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "error_metrics_comparison.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    metrics_df[["r2"]].plot(kind="bar", color="#2CA02C")
    plt.title("R² comparison")
    plt.ylabel("R²")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison.png", dpi=200)
    plt.close()

    if "random_forest" in results:
        rf_payload = results["random_forest"]
        rf_model = rf_payload["model"]
        if hasattr(rf_model.named_steps["model"], "feature_importances_"):
            importances = rf_model.named_steps["model"].feature_importances_
            feature_df = pd.DataFrame(
                {
                    "feature": rf_payload["feature_columns"],
                    "importance": importances,
                }
            ).sort_values("importance", ascending=False)
            feature_df.head(10).to_csv(output_dir / "feature_importance.csv", index=False)

            plt.figure(figsize=(9, 5))
            feature_df.head(10).plot(kind="bar", x="feature", y="importance", legend=False)
            plt.title("Top 10 Feature Importances")
            plt.ylabel("Importance")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(output_dir / "feature_importance.png", dpi=200)
            plt.close()

    print("\nModel metrics table:")
    print(metrics_df.to_string())
    print("\nComparison predictions table:")
    print(comparison_predictions.head(15).to_string(index=False))


def main() -> None:
    weather = load_weather_data(DATASET_DIR)
    influenza = load_influenza_data(DATASET_DIR)
    weather_features = build_weekly_weather_features(weather)
    dataset = create_dataset(influenza, weather_features)

    print(f"Rows after merge: {len(dataset)}")
    print("\nPreview of merged dataset:")
    print(dataset.head(5).to_string(index=False))

    results = train_models(dataset)
    ensemble_result = create_ensemble_result(results)
    if ensemble_result:
        results["ensemble"] = ensemble_result
    save_tables_and_plots(results, PROJECT_ROOT / "outputs")


if __name__ == "__main__":
    main()

