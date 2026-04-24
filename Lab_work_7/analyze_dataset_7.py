from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_FILE = Path("Data_Set_tabl_2/Data_Set_7.xlsx")
DEFAULT_SHEET = "qrySales"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze sales data from Excel file")
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_FILE,
        help=f"Path to Excel file (default: {DEFAULT_FILE})",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default=DEFAULT_SHEET,
        help=f"Sheet name (default: {DEFAULT_SHEET})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs"),
        help="Output directory",
    )
    parser.add_argument(
        "--forecast-months",
        type=int,
        default=6,
        help="Forecast horizon in months",
    )
    return parser.parse_args()


def forecast_poly(series: pd.Series, horizon: int, degree: int = 2) -> pd.Series:
    y = series.values.astype(float)
    x = np.arange(len(y), dtype=float)
    deg = min(degree, max(1, len(y) - 1))
    coef = np.polyfit(x, y, deg=deg)
    model = np.poly1d(coef)

    xf = np.arange(len(y), len(y) + horizon, dtype=float)
    yf = model(xf)
    yf = np.maximum(yf, 0.0)

    future_index = pd.date_range(
        start=(series.index.max() + pd.offsets.MonthBegin(1)),
        periods=horizon,
        freq="MS",
    )
    return pd.Series(yf, index=future_index, name=f"forecast_{series.name}")


def poly_fit(series: pd.Series, degree: int = 2) -> pd.Series:
    y = series.values.astype(float)
    x = np.arange(len(y), dtype=float)
    deg = min(degree, max(1, len(y) - 1))
    coef = np.polyfit(x, y, deg=deg)
    model = np.poly1d(coef)
    y_fit = model(x)
    y_fit = np.maximum(y_fit, 0.0)
    return pd.Series(y_fit, index=series.index, name=f"polyfit_{series.name}")


def save_line_plot(
    actual: pd.Series,
    forecast: pd.Series,
    trend: pd.Series | None,
    title: str,
    ylabel: str,
    out_file: Path,
) -> None:
    plt.figure(figsize=(11, 5))
    plt.plot(actual.index, actual.values, marker="o", label="Actual")
    if trend is not None:
        plt.plot(trend.index, trend.values, linewidth=2, label="Polynomial fit")
    plt.plot(
        forecast.index, forecast.values, marker="o", linestyle="--", label="Forecast"
    )
    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=130)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.file, sheet_name=args.sheet)

    required = {
        "OrderDate",
        "Revenue",
        "Quantity",
        "CustomerName",
        "ProductName",
        "EmployeeName",
        "SupplierName",
        "Paid?",
        "CustomerCountry",
        "CustomerCity",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["OrderDate"] = pd.to_datetime(df["OrderDate"])
    df = df.dropna(subset=["OrderDate", "Revenue", "Quantity"]).copy()
    df["YearMonth"] = df["OrderDate"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby("YearMonth", as_index=True)
        .agg(
            revenue=("Revenue", "sum"),
            quantity=("Quantity", "sum"),
            orders=("CustomerID", "count"),
        )
        .sort_index()
    )

    paid_split = (
        df.groupby(["YearMonth", "Paid?"], as_index=False)["Revenue"]
        .sum()
        .pivot(index="YearMonth", columns="Paid?", values="Revenue")
        .fillna(0.0)
        .sort_index()
    )

    top_customers = (
        df.groupby("CustomerName", as_index=False)
        .agg(
            revenue=("Revenue", "sum"),
            quantity=("Quantity", "sum"),
            orders=("CustomerID", "count"),
        )
        .sort_values("revenue", ascending=False)
        .head(10)
    )

    top_products = (
        df.groupby("ProductName", as_index=False)
        .agg(
            revenue=("Revenue", "sum"),
            quantity=("Quantity", "sum"),
            orders=("CustomerID", "count"),
        )
        .sort_values("revenue", ascending=False)
        .head(10)
    )

    region = (
        df.groupby("CustomerCountry", as_index=False)
        .agg(
            revenue=("Revenue", "sum"),
            quantity=("Quantity", "sum"),
            orders=("CustomerID", "count"),
        )
        .sort_values("revenue", ascending=False)
    )

    monthly_revenue_fc = forecast_poly(
        monthly["revenue"], args.forecast_months, degree=2
    )
    monthly_revenue_trend = poly_fit(monthly["revenue"], degree=2)
    monthly_quantity_fc = forecast_poly(
        monthly["quantity"], args.forecast_months, degree=2
    )
    monthly_quantity_trend = poly_fit(monthly["quantity"], degree=2)

    top_product_name = top_products.iloc[0]["ProductName"]
    top_product_monthly = (
        df[df["ProductName"] == top_product_name]
        .groupby("YearMonth", as_index=True)["Quantity"]
        .sum()
        .sort_index()
    )
    top_product_fc = forecast_poly(top_product_monthly, args.forecast_months, degree=2)
    top_product_trend = poly_fit(top_product_monthly, degree=2)

    monthly.to_csv(out_dir / "monthly_metrics.csv", float_format="%.4f")
    paid_split.to_csv(out_dir / "monthly_paid_split.csv", float_format="%.4f")
    top_customers.to_csv(
        out_dir / "top_customers.csv", index=False, float_format="%.4f"
    )
    top_products.to_csv(out_dir / "top_products.csv", index=False, float_format="%.4f")
    region.to_csv(out_dir / "country_metrics.csv", index=False, float_format="%.4f")

    pd.concat([monthly["revenue"], monthly_revenue_fc]).to_csv(
        out_dir / "forecast_revenue.csv",
        header=["revenue_or_forecast"],
        float_format="%.4f",
    )
    pd.concat([monthly["quantity"], monthly_quantity_fc]).to_csv(
        out_dir / "forecast_quantity.csv",
        header=["quantity_or_forecast"],
        float_format="%.4f",
    )
    pd.concat([top_product_monthly, top_product_fc]).to_csv(
        out_dir / "forecast_top_product_quantity.csv",
        header=["top_product_quantity_or_forecast"],
        float_format="%.4f",
    )

    save_line_plot(
        actual=monthly["revenue"],
        forecast=monthly_revenue_fc,
        trend=monthly_revenue_trend,
        title="Monthly Revenue + Forecast",
        ylabel="Revenue",
        out_file=out_dir / "revenue_forecast.png",
    )
    save_line_plot(
        actual=monthly["quantity"],
        forecast=monthly_quantity_fc,
        trend=monthly_quantity_trend,
        title="Monthly Quantity + Forecast",
        ylabel="Quantity",
        out_file=out_dir / "quantity_forecast.png",
    )
    save_line_plot(
        actual=top_product_monthly,
        forecast=top_product_fc,
        trend=top_product_trend,
        title=f"Top Product Quantity ({top_product_name}) + Forecast",
        ylabel="Quantity",
        out_file=out_dir / "top_product_forecast.png",
    )

    plt.figure(figsize=(11, 5))
    for col in paid_split.columns:
        plt.plot(paid_split.index, paid_split[col], marker="o", label=f"Paid={col}")
    plt.title("Revenue by Payment Status")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "paid_split.png", dpi=130)
    plt.close()

    summary_lines = [
        "Data_Set_7 analysis summary",
        f"Rows analyzed: {len(df)}",
        f"Period: {df['OrderDate'].min().date()} .. {df['OrderDate'].max().date()}",
        f"Countries: {df['CustomerCountry'].nunique()}",
        f"Customers: {df['CustomerName'].nunique()}",
        f"Products: {df['ProductName'].nunique()}",
        f"Total revenue: {df['Revenue'].sum():,.2f}",
        f"Total quantity: {df['Quantity'].sum():,.0f}",
        f"Top customer by revenue: {top_customers.iloc[0]['CustomerName']} ({top_customers.iloc[0]['revenue']:,.2f})",
        f"Top product by revenue: {top_products.iloc[0]['ProductName']} ({top_products.iloc[0]['revenue']:,.2f})",
        f"Forecast horizon: {args.forecast_months} months",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("Done. Results saved to:", out_dir.resolve())
    print("Main files:")
    print(" - summary.txt")
    print(" - monthly_metrics.csv")
    print(" - forecast_revenue.csv")
    print(" - forecast_quantity.csv")
    print(" - forecast_top_product_quantity.csv")
    print(" - revenue_forecast.png")
    print(" - quantity_forecast.png")
    print(" - top_product_forecast.png")
    print(" - paid_split.png")


if __name__ == "__main__":
    main()
