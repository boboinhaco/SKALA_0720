# 국적 결측 삭제 후 수치형 상관·income 관련도·범주 변수 중복·성별 근무시간을 한 화면에 시각화

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

BASE = Path(__file__).resolve().parent
CSV = BASE / "adult_data.csv"
NUM = ["age", "education-num", "hours-per-week", "capital-gain", "capital-loss"]
INCOME_CAT = ["relationship", "marital-status", "education", "occupation",
              "sex", "workclass", "race", "native-country"]
PAIR_CAT = ["workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex"]

plt.rcParams["font.family"] = "AppleGothic"  # 한글 깨짐 방지 (맥)
plt.rcParams["axes.unicode_minus"] = False

# 범주형-범주형 관련 강도 (0~1)
def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = chi2_contingency(ct)[0]
    return np.sqrt(chi2 / (ct.sum().sum() * (min(ct.shape) - 1)))


# 로딩 → 국적 결측 삭제 → 나머지 범주 결측 Unknown
def load_data():
    df = pd.read_csv(CSV, skipinitialspace=True, na_values="?")
    df = df.dropna(subset=["native-country"]).copy()  # 국적 결측 삭제
    df[["workclass", "occupation"]] = df[["workclass", "occupation"]].fillna("Unknown")
    return df


# 범주 변수쌍의 관련강도 행렬 계산 (중복 변수 탐지용)
def pair_matrix(df):
    mat = pd.DataFrame(index=PAIR_CAT, columns=PAIR_CAT, dtype=float)
    for a in PAIR_CAT:
        for b in PAIR_CAT:
            mat.loc[a, b] = 1.0 if a == b else cramers_v(df[a], df[b])
    return mat.astype(float)


def make_plots(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (1) 수치형 상관 히트맵 → 대부분 독립
    sns.heatmap(df[NUM].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=axes[0, 0])
    axes[0, 0].set_title("수치형 변수 상관관계 (대부분 독립)")

    # (2) income 관련도 순위 막대
    rel = sorted([(c, cramers_v(df[c], df["income"])) for c in INCOME_CAT],
                 key=lambda x: x[1])
    axes[0, 1].barh(*zip(*rel), color="steelblue")
    axes[0, 1].set_title("income과의 관련 강도 (Cramér's V)")
    axes[0, 1].set_xlabel("관련 강도")

    # (3) 범주 변수쌍 관련강도 히트맵 → 중복 탐지
    sns.heatmap(pair_matrix(df), annot=True, fmt=".2f", cmap="YlOrRd", ax=axes[1, 0])
    axes[1, 0].set_title("범주형 변수쌍 관련강도 (중복 탐지)")

    # (4) 성별 근무시간 박스플롯 → 분포 차이
    sns.boxplot(data=df, x="sex", y="hours-per-week", ax=axes[1, 1])
    axes[1, 1].set_title("성별 주당 근무시간 분포")

    fig.tight_layout()
    out = BASE / "variable_relationships.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")  # 파일 저장
    plt.close(fig)
    print(f"시각화 저장: {out.name}")


make_plots(load_data())  # 실행