# -*- coding: utf-8 -*-
"""UCI Adult 데이터의 통계·연관도 시각화와 분류 모델 학습을 수행한다."""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import font_manager
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
REPORT_PNG_PATH = SCRIPT_DIR / "classification_report.png"
VARIABLE_CHART_PATH = SCRIPT_DIR / "variable_relationships.png"
ASSOCIATION_CHART_PATH = SCRIPT_DIR / "outputs" / "association_heatmap.png"

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

RELATION_NUMERIC_COLUMNS = [
    "age",
    "education-num",
    "hours-per-week",
    "capital-gain",
    "capital-loss",
]
INCOME_CATEGORICAL_COLUMNS = [
    "relationship",
    "marital-status",
    "education",
    "occupation",
    "sex",
    "workclass",
    "race",
    "native-country",
]
PAIR_CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
]


def setup_korean_font() -> None:
    """운영체제에 설치된 한글 폰트를 찾아 matplotlib에 적용한다."""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for font in ("AppleGothic", "NanumGothic", "Malgun Gothic"):
        if font in installed:
            plt.rcParams["font.family"] = [font, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def cramers_v(first: pd.Series, second: pd.Series) -> float:
    """두 범주형 변수의 편향 보정 Cramér's V를 계산한다."""
    contingency = pd.crosstab(first, second)
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return 0.0

    chi2 = stats.chi2_contingency(contingency)[0]
    sample_count = contingency.to_numpy().sum()
    if sample_count <= 1:
        return 0.0

    rows, columns = contingency.shape
    phi_squared = chi2 / sample_count
    corrected_phi = max(
        0.0,
        phi_squared
        - (columns - 1) * (rows - 1) / (sample_count - 1),
    )
    corrected_rows = rows - (rows - 1) ** 2 / (sample_count - 1)
    corrected_columns = columns - (columns - 1) ** 2 / (sample_count - 1)
    denominator = min(corrected_rows - 1, corrected_columns - 1)
    return (
        float(np.sqrt(corrected_phi / denominator))
        if denominator > 0
        else 0.0
    )


def correlation_ratio(category: pd.Series, numeric: pd.Series) -> float:
    """범주형과 수치형 변수 사이의 상관비 eta를 계산한다."""
    pair = pd.DataFrame({"category": category, "numeric": numeric}).dropna()
    if pair.empty or pair["category"].nunique() < 2:
        return 0.0

    grand_mean = pair["numeric"].mean()
    between_sum = (
        pair.groupby("category", observed=True)["numeric"]
        .apply(lambda group: len(group) * (group.mean() - grand_mean) ** 2)
        .sum()
    )
    total_sum = ((pair["numeric"] - grand_mean) ** 2).sum()
    return float(np.sqrt(between_sum / total_sum)) if total_sum > 0 else 0.0


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

    # 국적이 결측이거나 '?'인 행은 모든 후속 분석에서 제외한다.
    before_count = len(data)
    data = data.dropna(subset=["native-country"]).copy()
    removed_country_count = before_count - len(data)
    data[["workclass", "occupation"]] = data[
        ["workclass", "occupation"]
    ].fillna("Unknown")

    print(f"[데이터 출처]\n{source}")
    print(f"\n[데이터 크기]\n{data.shape}")
    print(f"\n[국적 결측 또는 '?' 제외]\n{removed_country_count:,}행")
    print("\n[열별 결측치 개수]")
    print(data[COLUMNS].isna().sum()[lambda values: values > 0])
    print("\n[소득 그룹별 개수]")
    print(data["income"].value_counts())
    return data


def build_category_pair_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """범주형 변수쌍의 관련 강도 행렬을 계산한다."""
    matrix = pd.DataFrame(
        index=PAIR_CATEGORICAL_COLUMNS,
        columns=PAIR_CATEGORICAL_COLUMNS,
        dtype=float,
    )
    for first in PAIR_CATEGORICAL_COLUMNS:
        for second in PAIR_CATEGORICAL_COLUMNS:
            matrix.loc[first, second] = (
                1.0
                if first == second
                else cramers_v(data[first], data[second])
            )
    return matrix


def save_variable_relationships(data: pd.DataFrame) -> None:
    """상관·소득 관련도·범주 중복·성별 근무시간을 2×2 PNG로 저장한다."""
    figure, axes = plt.subplots(2, 2, figsize=(16, 12))

    sns.heatmap(
        data[RELATION_NUMERIC_COLUMNS].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("수치형 변수 상관관계")

    income_relationships = sorted(
        [
            (column, cramers_v(data[column], data["income"]))
            for column in INCOME_CATEGORICAL_COLUMNS
        ],
        key=lambda item: item[1],
    )
    labels, values = zip(*income_relationships, strict=True)
    axes[0, 1].barh(labels, values, color="steelblue")
    axes[0, 1].set_title("income과의 관련 강도 (Cramer's V)")
    axes[0, 1].set_xlabel("관련 강도")
    axes[0, 1].set_xlim(0, 1)

    sns.heatmap(
        build_category_pair_matrix(data),
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("범주형 변수쌍 관련 강도 (중복 탐지)")

    sns.boxplot(data=data, x="sex", y="hours-per-week", ax=axes[1, 1])
    axes[1, 1].set_title("성별 주당 근무시간 분포")

    figure.tight_layout()
    figure.savefig(VARIABLE_CHART_PATH, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"변수 관계 시각화 저장: {VARIABLE_CHART_PATH}")


def build_association_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """수치형과 범주형 전체 변수의 0~1 연관도 행렬을 계산한다."""
    all_columns = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + ["income"]
    matrix = pd.DataFrame(
        np.zeros((len(all_columns), len(all_columns))),
        index=all_columns,
        columns=all_columns,
    )

    for index, first in enumerate(all_columns):
        for second_index in range(index, len(all_columns)):
            second = all_columns[second_index]
            if first == second:
                value = 1.0
            elif first in NUMERIC_COLUMNS and second in NUMERIC_COLUMNS:
                value = abs(data[first].corr(data[second]))
            elif first not in NUMERIC_COLUMNS and second not in NUMERIC_COLUMNS:
                value = cramers_v(data[first], data[second])
            else:
                category, numeric = (
                    (second, first)
                    if first in NUMERIC_COLUMNS
                    else (first, second)
                )
                value = correlation_ratio(data[category], data[numeric])

            if pd.isna(value):
                value = 0.0
            matrix.iloc[index, second_index] = value
            matrix.iloc[second_index, index] = value

    return matrix


def save_association_heatmap(data: pd.DataFrame) -> None:
    """전체 변수 통합 연관도 히트맵을 PNG로 저장한다."""
    matrix = build_association_matrix(data)
    figure, axis = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="rocket_r",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "연관도 (0=관계 없음 ~ 1=강한 관계)"},
        ax=axis,
    )
    axis.set_title(
        "수치형 + 범주형 통합 연관도 히트맵\n"
        "(수치-수치: |Pearson r| · 범주-범주: Cramer's V · "
        "수치-범주: 상관비 eta)",
        fontsize=13,
    )
    axis.tick_params(axis="x", rotation=45)
    axis.tick_params(axis="y", rotation=0)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    figure.tight_layout()

    ASSOCIATION_CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(ASSOCIATION_CHART_PATH, dpi=120, bbox_inches="tight")
    plt.close(figure)

    print(f"통합 연관도 히트맵 저장: {ASSOCIATION_CHART_PATH}")
    print("income과 연관도가 높은 상위 5개 변수:")
    top = matrix["income"].drop("income").sort_values(ascending=False).head(5)
    for name, value in top.items():
        print(f"  · {name:15}: {value:.3f}")


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
    """혼동행렬 HTML과 분류 지표 히트맵 PNG를 저장한다."""
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

    class_names = ["<=50K", ">50K"]
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_data = pd.DataFrame(report).transpose().loc[
        class_names + ["macro avg", "weighted avg"],
        ["precision", "recall", "f1-score"],
    ]
    report_figure, report_axis = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        report_data,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        ax=report_axis,
    )
    report_axis.set_title("Classification Report")
    report_axis.set_xlabel("Metric")
    report_axis.set_ylabel("")
    report_figure.tight_layout()
    report_figure.savefig(REPORT_PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(report_figure)
    print(f"분류 리포트 PNG 저장: {REPORT_PNG_PATH}")


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
    setup_korean_font()
    data = load_data()
    analyze_fnlwgt(data)
    save_variable_relationships(data)
    save_association_heatmap(data)
    train_and_evaluate(data)


if __name__ == "__main__":
    main()
