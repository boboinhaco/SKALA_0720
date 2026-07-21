"""UCI Adult 소득 데이터의 통계 분석과 분류 모델 학습을 수행한다."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "income_target"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "adult_data.csv"
MODEL_PATH = SCRIPT_DIR / "best_income_classifier.joblib"
RELATION_CHART_PATH = SCRIPT_DIR / "income_fnlwgt_analysis.html"
REPORT_CHART_PATH = SCRIPT_DIR / "classification_report.html"

DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/adult/adult.data"
)

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


def load_data() -> pd.DataFrame:
    """로컬 CSV를 우선 사용하고 없으면 UCI URL에서 데이터를 읽는다."""
    if DATA_PATH.exists():
        data = pd.read_csv(DATA_PATH, na_values="?", skipinitialspace=True)
        source = DATA_PATH
    else:
        data = pd.read_csv(
            DATA_URL,
            header=None,
            names=COLUMNS,
            na_values="?",
            skipinitialspace=True,
        )
        source = DATA_URL

    data.columns = data.columns.str.strip()
    missing_columns = set(COLUMNS) - set(data.columns)
    if missing_columns:
        raise ValueError(f"필수 열이 없습니다: {sorted(missing_columns)}")

    string_columns = data.select_dtypes(include=["object", "string"]).columns
    data[string_columns] = data[string_columns].apply(
        lambda column: column.str.strip().replace("?", pd.NA)
    )
    data["income"] = data["income"].str.rstrip(".")
    data[TARGET_COLUMN] = data["income"].map({"<=50K": 0, ">50K": 1})

    if data[TARGET_COLUMN].isna().any():
        unexpected = sorted(
            data.loc[data[TARGET_COLUMN].isna(), "income"].unique())
        raise ValueError(f"알 수 없는 income 값이 있습니다: {unexpected}")

    print(f"[데이터 출처]\n{source}")
    print(f"\n[데이터 크기]\n{data.shape}")
    print("\n[열별 결측치 개수]")
    print(data[COLUMNS].isna().sum()[lambda values: values > 0])
    print("\n[소득 그룹별 개수]")
    print(data["income"].value_counts())
    return data


def analyze_fnlwgt(data: pd.DataFrame) -> None:
    """소득 그룹과 fnlwgt의 관계를 검정하고 Plotly HTML로 저장한다."""
    low_income = data.loc[data[TARGET_COLUMN] == 0, "fnlwgt"].dropna()
    high_income = data.loc[data[TARGET_COLUMN] == 1, "fnlwgt"].dropna()

    t_stat, t_pvalue = stats.ttest_ind(
        low_income, high_income, equal_var=False
    )
    correlation, correlation_pvalue = stats.pointbiserialr(
        data[TARGET_COLUMN], data["fnlwgt"]
    )

    print("\n[소득 그룹별 fnlwgt 분석]")
    print(f"<=50K 그룹 평균: {low_income.mean():,.2f}")
    print(f">50K 그룹 평균: {high_income.mean():,.2f}")
    print(f"Welch t-test: t={t_stat:.3f}, p={t_pvalue:.3g}")
    print(
        "점이연 상관분석: "
        f"r={correlation:.4f}, p={correlation_pvalue:.3g}"
    )
    if t_pvalue < 0.05:
        print("결론: 두 그룹의 fnlwgt 평균에 통계적으로 유의한 차이가 있습니다.")
    else:
        print("결론: 두 그룹의 fnlwgt 평균 차이를 확인하지 못했습니다.")

    binned = (
        data.assign(
            fnlwgt_bin=pd.qcut(data["fnlwgt"], q=10, duplicates="drop")
        )
        .groupby("fnlwgt_bin", observed=True)
        .agg(
            fnlwgt_mean=("fnlwgt", "mean"),
            high_income_rate=(TARGET_COLUMN, "mean"),
            sample_count=(TARGET_COLUMN, "size"),
        )
        .reset_index(drop=True)
    )
    binned["high_income_rate"] *= 100

    x = binned["fnlwgt_mean"].to_numpy()
    y = binned["high_income_rate"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    overall_rate = data[TARGET_COLUMN].mean() * 100

    figure = px.line(
        binned,
        x="fnlwgt_mean",
        y="high_income_rate",
        markers=True,
        title=f"fnlwgt와 고소득 비율의 관계 (r={correlation:.3f})",
        labels={
            "fnlwgt_mean": "구간별 평균 fnlwgt",
            "high_income_rate": ">50K 비율 (%)",
        },
        hover_data={
            "fnlwgt_mean": ":,.0f",
            "high_income_rate": ":.2f",
            "sample_count": ":,",
        },
    )
    figure.add_scatter(
        x=x,
        y=slope * x + intercept,
        mode="lines",
        name="선형 추세선",
        line={"color": "crimson", "dash": "dash"},
    )
    figure.add_hline(
        y=overall_rate,
        line_dash="dot",
        line_color="gray",
        annotation_text=f"전체 고소득 비율 {overall_rate:.1f}%",
    )
    figure.update_layout(template="plotly_white", hovermode="x unified")
    figure.write_html(RELATION_CHART_PATH, include_plotlyjs=True)
    print(f"Plotly 분석 차트 저장: {RELATION_CHART_PATH}")


def build_grid_search() -> GridSearchCV:
    """결측치 처리부터 모델 선택까지 포함하는 파이프라인을 만든다."""
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE),
            ),
        ]
    )
    parameter_grid = [
        {
            "classifier": [
                LogisticRegression(
                    max_iter=2_000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                )
            ],
            "classifier__C": [0.1, 1.0],
            "classifier__solver": ["liblinear"],
        },
        {
            "classifier": [
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=1,
                )
            ],
            "classifier__max_depth": [10, 20],
            "classifier__min_samples_split": [2, 5],
        },
    ]
    return GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1",
        cv=3,
        n_jobs=1,
        verbose=1,
        refit=True,
    )


def save_report_chart(y_test: pd.Series, y_pred: np.ndarray) -> None:
    """혼동행렬을 Plotly HTML로 저장한다."""
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    figure = px.imshow(
        matrix,
        x=["예측 <=50K", "예측 >50K"],
        y=["실제 <=50K", "실제 >50K"],
        text_auto=True,
        color_continuous_scale="Blues",
        title="소득 분류 혼동행렬",
        labels={"color": "인원수"},
    )
    figure.update_layout(template="plotly_white")
    figure.write_html(REPORT_CHART_PATH, include_plotlyjs=True)
    print(f"Plotly 평가 차트 저장: {REPORT_CHART_PATH}")


def train_and_evaluate(data: pd.DataFrame) -> None:
    """학습/평가 데이터를 분리하고 최적 분류 모델을 저장·검증한다."""
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    search = build_grid_search()
    search.fit(X_train, y_train)
    print(f"\n최고 교차검증 F1: {search.best_score_:.4f}")
    print(f"최적 파라미터: {search.best_params_}")

    joblib.dump(search.best_estimator_, MODEL_PATH)
    loaded_classifier = joblib.load(MODEL_PATH)
    y_pred = loaded_classifier.predict(X_test)

    print(f"\n테스트 정확도: {accuracy_score(y_test, y_pred):.4f}")
    print(f"테스트 F1(>50K): {f1_score(y_test, y_pred):.4f}")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["<=50K", ">50K"],
            zero_division=0,
        )
    )
    print(f"모델 저장: {MODEL_PATH}")
    save_report_chart(y_test, y_pred)


def main() -> None:
    data = load_data()
    analyze_fnlwgt(data)
    train_and_evaluate(data)


if __name__ == "__main__":
    main()