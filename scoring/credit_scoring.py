import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


SAMPLE_FILE = "sample_data.xlsx"
DESCRIPTION_FILE = "data_description.xlsx"


def load_minimax_table() -> pd.DataFrame:
    description = pd.read_excel(DESCRIPTION_FILE)
    model_fields = description[
        description["Place_of_definition"].isin(
            [
                "Вказує позичальник",
                "Визначається при подачі заявки",
                "Визначається браузером при подачі заявки",
            ]
        )
    ][["Field_in_data", "Place_of_definition"]].copy()
    model_fields["Minimax"] = np.where(
        model_fields["Field_in_data"].isin(
            ["loan_amount", "loan_days", "children_count_id", "monthly_expenses"]
        ),
        "min",
        "max",
    )
    return model_fields


def minmax_series(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    lo = values.min(skipna=True)
    hi = values.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.full(len(values), 0.5), index=series.index)
    return (values - lo) / (hi - lo)


def build_scoring_frame(
    sample: pd.DataFrame, minimax: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.DataFrame(index=sample.index)
    used = []
    for row in minimax.itertuples(index=False):
        field = row.Field_in_data
        direction = str(row.Minimax).lower()
        if field not in sample.columns or direction not in {"min", "max"}:
            continue

        values = pd.to_numeric(sample[field], errors="coerce")
        if values.notna().sum() < max(10, len(values) // 10):
            continue

        scaled = minmax_series(values)
        # Higher score = safer borrower.
        frame[field] = 1 - scaled if direction == "min" else scaled
        used.append(field)

    return frame, used


def score_borrowers(scoring_frame: pd.DataFrame) -> pd.Series:
    if scoring_frame.empty:
        return pd.Series(dtype=float)
    weights = np.ones(scoring_frame.shape[1]) / scoring_frame.shape[1]
    raw = scoring_frame.fillna(scoring_frame.mean()).to_numpy(dtype=float)
    score_0_1 = np.dot(raw, weights)
    return pd.Series(score_0_1 * 100.0, index=scoring_frame.index, name="score")


def pick_threshold(scores: pd.Series, y: pd.Series) -> float:
    candidates = np.unique(np.nanpercentile(scores, np.linspace(5, 95, 41)))
    best_threshold = float(np.nanmedian(scores))
    best_f1 = -1.0
    y_true = y.astype(int).to_numpy()

    for threshold in candidates:
        pred = (scores >= threshold).astype(int).to_numpy()
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold


def make_return_target(sample: pd.DataFrame) -> pd.Series:
    if "loan_overdue" in sample.columns:
        return (
            pd.to_numeric(sample["loan_overdue"], errors="coerce").fillna(0) == 0
        ).astype(int)
    if "OK*" in sample.columns:
        return pd.to_numeric(sample["OK*"], errors="coerce").fillna(0).astype(int)
    raise ValueError("No target column found. Expected 'loan_overdue' or 'OK*'.")


def add_age_features(sample: pd.DataFrame) -> pd.DataFrame:
    result = sample.copy()
    if {"birth_date", "applied_at"}.issubset(result.columns):
        applied = pd.to_datetime(result["applied_at"], errors="coerce")
        birth = pd.to_datetime(result["birth_date"], errors="coerce")
        result["age_years"] = (applied - birth).dt.days / 365.25
    return result


def fraud_detection(sample: pd.DataFrame) -> pd.DataFrame:
    df = add_age_features(sample)
    reasons: list[list[str]] = [[] for _ in range(len(df))]

    def mark(mask: pd.Series, reason: str) -> None:
        idx = df.index[mask.fillna(False)]
        for i in idx:
            reasons[df.index.get_loc(i)].append(reason)

    if "monthly_income" in df.columns and "monthly_expenses" in df.columns:
        inc = pd.to_numeric(df["monthly_income"], errors="coerce")
        exp = pd.to_numeric(df["monthly_expenses"], errors="coerce")
        mark((inc <= 0) | inc.isna(), "invalid_income")
        mark((exp > inc) & inc.notna() & exp.notna(), "expenses_above_income")

    if "age_years" in df.columns:
        age = pd.to_numeric(df["age_years"], errors="coerce")
        mark((age < 18) | (age > 75), "implausible_age")

    if {"loan_amount", "loan_days"}.issubset(df.columns):
        loan_amount = pd.to_numeric(df["loan_amount"], errors="coerce")
        loan_days = pd.to_numeric(df["loan_days"], errors="coerce")
        mark((loan_amount <= 0) | loan_amount.isna(), "invalid_loan_amount")
        mark((loan_days <= 0) | loan_days.isna(), "invalid_loan_days")

    if {"seniority_years", "age_years"}.issubset(df.columns):
        seniority = pd.to_numeric(df["seniority_years"], errors="coerce")
        age = pd.to_numeric(df["age_years"], errors="coerce")
        mark(
            (seniority > (age - 14)) & seniority.notna() & age.notna(),
            "employment_history_inconsistent",
        )

    if "face_id" in df.columns:
        face = df["face_id"].replace({0: np.nan})
        repeated = face.notna() & face.duplicated(keep=False)
        mark(repeated, "reused_face_id")

    numeric = df.select_dtypes(include=[np.number]).copy()
    if not numeric.empty:
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        numeric = numeric.fillna(numeric.median(numeric_only=True))
        if numeric.shape[1] >= 2:
            iso = IsolationForest(contamination=0.08, random_state=42)
            outlier = iso.fit_predict(numeric) == -1
        else:
            outlier = np.zeros(len(df), dtype=bool)
    else:
        outlier = np.zeros(len(df), dtype=bool)

    fraud_flag = np.array([bool(r) for r in reasons]) | outlier
    out = pd.DataFrame(
        {
            "fraud_flag": fraud_flag.astype(int),
            "fraud_reasons": [
                ";".join(r) if r else ("isolation_forest_outlier" if o else "")
                for r, o in zip(reasons, outlier)
            ],
        },
        index=df.index,
    )
    return out


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)

    sample = pd.read_excel(SAMPLE_FILE)
    minimax = load_minimax_table()

    scoring_frame, used_features = build_scoring_frame(sample, minimax)
    if scoring_frame.empty:
        raise RuntimeError(
            "No usable scoring features were found in the provided files."
        )

    scores = score_borrowers(scoring_frame)
    y_return = make_return_target(sample)
    threshold = pick_threshold(scores, y_return)
    prediction = (scores >= threshold).astype(int)

    fraud = fraud_detection(sample)

    result = pd.DataFrame(
        {
            "Application": sample["Application"]
            if "Application" in sample.columns
            else sample.index,
            "score": scores.round(3),
            "score_threshold": round(threshold, 3),
            "credit_return_pred": prediction.astype(int),
            "fraud_flag": fraud["fraud_flag"].astype(int),
            "fraud_reasons": fraud["fraud_reasons"],
        }
    )

    output_path = "credit_scoring_results.xlsx"
    result.to_excel(output_path, index=False)

    print("used_features:", ", ".join(used_features))
    print(f"threshold={threshold:.3f}")
    print(f"saved={output_path}")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
