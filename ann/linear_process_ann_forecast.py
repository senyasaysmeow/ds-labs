import argparse
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def generate_dataset(
    n_samples: int,
    noise_std: float,
    anomaly_ratio: float,
    slope: float,
    intercept: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    t = np.arange(n_samples)
    clean_signal = intercept + slope * t
    noisy_signal = clean_signal + rng.normal(loc=0.0, scale=noise_std, size=n_samples)

    dataset = noisy_signal.copy()
    anomaly_count = int(n_samples * anomaly_ratio)
    anomaly_idx = rng.choice(n_samples, size=anomaly_count, replace=False)

    lower, upper = float(noisy_signal.min()), float(noisy_signal.max())
    dataset[anomaly_idx] = rng.uniform(lower, upper, size=anomaly_count)

    return t, clean_signal, noisy_signal, dataset, anomaly_idx


def make_supervised(series: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    x_data = []
    y_data = []

    for i in range(window, len(series)):
        x_data.append(series[i - window : i])
        y_data.append(series[i])

    return np.array(x_data), np.array(y_data)


def architecture_name(arch: Sequence[int]) -> str:
    return "-".join(str(unit) for unit in arch)


def train_one_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    architecture: Sequence[int],
    seed: int,
) -> Tuple[Dict[str, float], np.ndarray, List[float]]:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train_scaled = x_scaler.fit_transform(x_train)
    x_test_scaled = x_scaler.transform(x_test)

    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    model = MLPRegressor(
        hidden_layer_sizes=tuple(architecture),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=450,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=seed,
    )
    model.fit(x_train_scaled, y_train_scaled)

    y_pred_scaled = model.predict(x_test_scaled)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}
    return metrics, y_pred, model.loss_curve_


def evaluate_architectures(
    series: np.ndarray,
    window: int,
    test_ratio: float,
    architectures: Sequence[Sequence[int]],
    seed: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object], np.ndarray, np.ndarray]:
    x_all, y_all = make_supervised(series, window)
    split = int(len(x_all) * (1 - test_ratio))

    x_train, y_train = x_all[:split], y_all[:split]
    x_test, y_test = x_all[split:], y_all[split:]

    test_start_idx = window + split
    test_time = np.arange(test_start_idx, test_start_idx + len(y_test))

    results: List[Dict[str, object]] = []
    best: Dict[str, object] = {}

    for i, arch in enumerate(architectures):
        metrics, y_pred, loss_curve = train_one_model(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            architecture=arch,
            seed=seed + i,
        )

        result: Dict[str, object] = {
            "architecture": tuple(arch),
            "arch_name": architecture_name(arch),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "y_pred": y_pred,
            "loss_curve": loss_curve,
        }
        results.append(result)

        if not best or result["rmse"] < best["rmse"]:
            best = result

    return results, best, y_test, test_time


def plot_dataset(
    t: np.ndarray,
    clean_signal: np.ndarray,
    noisy_signal: np.ndarray,
    dataset: np.ndarray,
    anomaly_idx: np.ndarray,
    output_path: str,
) -> None:
    plt.figure(figsize=(13, 5))
    plt.plot(t, clean_signal, linewidth=1.6, label="Linear process (clean)")
    plt.plot(t, noisy_signal, linewidth=1.0, alpha=0.7, label="Linear + Gaussian noise")
    plt.scatter(
        anomaly_idx,
        dataset[anomaly_idx],
        s=16,
        color="crimson",
        alpha=0.75,
        label="Uniform anomalies (10%)",
        zorder=5,
    )
    plt.plot(t, dataset, linewidth=1.1, alpha=0.65, label="Final dataset")
    plt.title("Generated dataset")
    plt.xlabel("Discrete sample index")
    plt.ylabel("Value")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_forecast(
    test_time: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    loss_curve: Sequence[float],
    output_path: str,
) -> None:
    abs_error = np.abs(y_test - y_pred)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)

    axes[0].plot(test_time, y_test, label="Actual series", linewidth=1.3)
    axes[0].plot(test_time, y_pred, label="ANN forecast", linewidth=1.3, alpha=0.9)
    axes[0].set_title("Forecast on test segment")
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("Value")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(test_time, abs_error, color="darkorange", linewidth=1.1)
    axes[1].set_title("Absolute prediction error")
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("|Error|")
    axes[1].grid(alpha=0.25)

    axes[2].plot(np.arange(1, len(loss_curve) + 1), loss_curve, color="seagreen")
    axes[2].set_title("Training loss curve (best architecture)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_architecture_study(
    results: Sequence[Dict[str, object]], output_path: str
) -> None:
    sorted_results = sorted(results, key=lambda item: item["rmse"])

    names = [str(item["arch_name"]) for item in sorted_results]
    rmse_values = [float(item["rmse"]) for item in sorted_results]
    mae_values = [float(item["mae"]) for item in sorted_results]

    x = np.arange(len(names))
    width = 0.4

    plt.figure(figsize=(12, 5))
    plt.bar(x - width / 2, rmse_values, width=width, label="RMSE")
    plt.bar(x + width / 2, mae_values, width=width, label="MAE")
    plt.xticks(x, names)
    plt.xlabel("Hidden layer architecture")
    plt.ylabel("Error")
    plt.title("Prediction accuracy vs network structure")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_results_table(
    results: Sequence[Dict[str, object]], best: Dict[str, object]
) -> None:
    print("\nArchitecture comparison (lower MAE/RMSE is better):")
    print(f"{'Architecture':<16}{'MAE':>12}{'RMSE':>12}{'R2':>12}")
    print("-" * 52)

    for result in results:
        print(
            f"{result['arch_name']:<16}"
            f"{result['mae']:>12.4f}"
            f"{result['rmse']:>12.4f}"
            f"{result['r2']:>12.4f}"
        )

    print("\nBest architecture by RMSE:")
    print(
        f"{best['arch_name']}  |  MAE={best['mae']:.4f}, "
        f"RMSE={best['rmse']:.4f}, R2={best['r2']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a noisy linear series with anomalies and forecast it with ANN."
    )
    parser.add_argument(
        "--samples", type=int, default=5000, help="Number of discrete samples"
    )
    parser.add_argument(
        "--noise-std", type=float, default=20.0, help="Gaussian noise std"
    )
    parser.add_argument(
        "--anomaly-ratio",
        type=float,
        default=0.10,
        help="Ratio of anomalous measurements",
    )
    parser.add_argument("--slope", type=float, default=0.09, help="Linear law slope")
    parser.add_argument(
        "--intercept", type=float, default=40.0, help="Linear law intercept"
    )
    parser.add_argument("--window", type=int, default=20, help="Sliding window size")
    parser.add_argument(
        "--test-ratio", type=float, default=0.20, help="Test segment ratio"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--show", action="store_true", help="Display plots interactively"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    architectures = [
        (8,),
        (16,),
        (32,),
        (64,),
        (32, 16),
        (64, 32),
        (128, 64, 32),
    ]

    t, clean_signal, noisy_signal, dataset, anomaly_idx = generate_dataset(
        n_samples=args.samples,
        noise_std=args.noise_std,
        anomaly_ratio=args.anomaly_ratio,
        slope=args.slope,
        intercept=args.intercept,
        seed=args.seed,
    )

    results, best, y_test, test_time = evaluate_architectures(
        series=dataset,
        window=args.window,
        test_ratio=args.test_ratio,
        architectures=architectures,
        seed=args.seed,
    )

    print_results_table(results, best)

    dataset_plot = "dataset_generation.png"
    forecast_plot = "forecast_process.png"
    study_plot = "architecture_accuracy_study.png"

    plot_dataset(
        t=t,
        clean_signal=clean_signal,
        noisy_signal=noisy_signal,
        dataset=dataset,
        anomaly_idx=anomaly_idx,
        output_path=dataset_plot,
    )
    plot_forecast(
        test_time=test_time,
        y_test=y_test,
        y_pred=np.asarray(best["y_pred"]),
        loss_curve=np.asarray(best["loss_curve"]),
        output_path=forecast_plot,
    )
    plot_architecture_study(results, study_plot)

    print("\nPlots saved:")
    print(f"- {dataset_plot}")
    print(f"- {forecast_plot}")
    print(f"- {study_plot}")

    if args.show:
        for image in (dataset_plot, forecast_plot, study_plot):
            img = plt.imread(image)
            plt.figure(figsize=(12, 6))
            plt.imshow(img)
            plt.axis("off")
            plt.title(image)
            plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
