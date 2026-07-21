# 프로그램: sales_100k.csv 심화 분석 (시각화 4종 · 통계검정 · sklearn Pipeline · Plotly)
# 기능: 실습3 IQR 정제 데이터 연계→2×2 시각화→t-test/카이제곱→Pipeline 학습·저장·재로딩→Plotly 저장
# 변경내역: v2.0 Imputer 파이프라인 내장·시각화 샘플링·연월 집계·예외처리 강화·카이제곱(지역×카테고리)

from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# 입출력 경로 (실행 위치와 무관하게 스크립트 폴더 기준)
BASE = Path(__file__).resolve().parent
CSV = BASE / "sales_100k.csv"
OUTPUT_DIR = BASE / "practice4_output"
NUM_COLS = ["quantity", "unit_price", "customer_age"]
CAT_COLS = ["region", "category", "payment_method", "customer_gender"]
TARGET = "amount"
ALPHA = 0.05
RANDOM_STATE = 42
SAMPLE_N = 100_000

# macOS 한글 폰트 설정 (실패 시 기본 폰트 유지)
try:
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


# 실습3 연계: 로딩 → 결측·타입 정리 → IQR 이상치 제거 → 정제 df 반환
def load_and_clean(file_path):
    need = set(NUM_COLS + CAT_COLS + [TARGET, "order_date"])
    df = pd.read_csv(file_path)
    missing = need - set(df.columns)
    if missing:  # 필수 컬럼 검증
        raise KeyError(f"필수 컬럼이 없습니다: {sorted(missing)}")
    if df.empty:  # 빈 파일 검증
        raise ValueError("CSV 파일에 데이터가 없습니다.")

    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")  # 날짜 파싱
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")  # 숫자 변환
    if df["amount"].notna().sum() == 0:  # amount 사용 가능 여부 검증
        raise ValueError("amount 컬럼에 사용할 수 있는 숫자가 없습니다.")
    df = df.dropna(subset=["order_date"])  # 날짜 결측 행 제거

    q1, q3 = df["amount"].quantile(0.25), df["amount"].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr  # IQR 정상범위
    before = len(df)
    df_clean = df[df["amount"].between(lo, hi)].copy()  # 이상치 제거 (null amount도 제외)

    print("\n데이터 정제 결과")
    print(f"IQR 정상범위: {lo:.2f} ~ {hi:.2f}")
    print(f"정제 전 {before:,} / 정제 후 {len(df_clean):,} / 제거 {before - len(df_clean):,}")
    return df_clean


# 유의수준 기준 검정 결과를 한 줄 해석 문자열로 반환 (t-test·카이제곱 공용)
def interpret(p_value, sig_text, non_sig_text):
    return sig_text if p_value < ALPHA else non_sig_text


# 1) EDA 시각화 4종 (2×2 서브플롯 한 figure, 대용량은 샘플로 렌더링)
def plot_eda(df, output_dir):
    sample = df.sample(n=min(len(df), SAMPLE_N), random_state=RANDOM_STATE)  # 플롯용 샘플
    monthly = (df.assign(month=df["order_date"].dt.to_period("M").astype(str))
               .groupby("month", as_index=False)["amount"].sum())  # 월별 총매출(전체 기준)
    corr = df[NUM_COLS + [TARGET]].corr()  # 수치형 상관행렬

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    sns.histplot(data=sample, x="amount", kde=True, ax=axes[0, 0])  # 히스토그램+KDE
    axes[0, 0].set_title("매출 분포")
    sns.boxplot(data=sample, x="region", y="amount", ax=axes[0, 1])  # 지역별 박스플롯
    axes[0, 1].set_title("지역별 매출"); axes[0, 1].tick_params(axis="x", rotation=45)
    sns.lineplot(data=monthly, x="month", y="amount", marker="o", ax=axes[1, 0])  # 월별 라인
    axes[1, 0].set_title("월별 총매출"); axes[1, 0].tick_params(axis="x", rotation=45)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 1])  # 상관 히트맵
    axes[1, 1].set_title("상관 히트맵")

    fig.tight_layout()
    path = output_dir / "eda_visualizations.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")  # 파일 저장 (show 미사용)
    plt.close(fig)
    print(f"\nEDA 시각화 저장: {path.name}")


# 2) 서울 vs 부산 평균 매출 Welch t-test
def run_t_test(df):
    seoul = df.loc[df["region"] == "서울", "amount"]
    busan = df.loc[df["region"] == "부산", "amount"]
    if seoul.empty or busan.empty:  # 그룹 존재 검증
        raise ValueError("서울 또는 부산 데이터가 없어 t-test를 수행할 수 없습니다.")

    t_stat, p = ttest_ind(seoul, busan, equal_var=False, nan_policy="omit")  # 등분산 미가정
    verdict = interpret(p, "평균 매출에 유의미한 차이 있음", "평균 매출에 유의미한 차이 없음")
    print("\n서울·부산 평균매출 t-test")
    print(f"서울 평균 {seoul.mean():,.2f} / 부산 평균 {busan.mean():,.2f}")
    print(f"t통계량={t_stat:.4f}, p-value={p:.6f} → {verdict}")


# 3) 지역 × 카테고리 독립성 카이제곱 검정
def run_chi_square(df):
    table = pd.crosstab(df["region"], df["category"])  # 지역×카테고리 분할표
    if table.shape[0] < 2 or table.shape[1] < 2:  # 범주 수 검증
        raise ValueError("카이제곱 검정에는 각 변수에 두 개 이상 범주가 필요합니다.")

    chi2, p, dof, _ = chi2_contingency(table)  # 독립성 검정
    verdict = interpret(p, "지역과 카테고리는 유의미한 관련 있음", "지역과 카테고리는 유의미한 관련 없음")
    print("\n지역·카테고리 카이제곱 검정")
    print(f"카이제곱={chi2:.4f}, 자유도={dof}, p-value={p:.6f} → {verdict}")


# 전처리(수치=결측대치+표준화, 범주=결측대치+원핫) + Ridge를 하나의 Pipeline으로 구성
def build_pipeline():
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())])  # 수치형 전처리
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("encoder", OneHotEncoder(handle_unknown="ignore"))])  # 범주형 전처리
    preproc = ColumnTransformer([("num", num_pipe, NUM_COLS),
                                 ("cat", cat_pipe, CAT_COLS)])  # 컬럼별 전처리 결합
    return Pipeline([("prep", preproc), ("reg", Ridge(alpha=1.0, solver="lsqr"))])


# 4) Pipeline 학습 → 평가(R2·MAE) → 저장 → 재로딩 검증
def train_and_save(df, output_dir):
    model = build_pipeline()
    X, y = df[NUM_COLS + CAT_COLS], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    model.fit(X_tr, y_tr)  # 훈련
    pred = model.predict(X_te)
    print("\nPipeline 모델 평가")
    print(f"테스트 수 {len(X_te):,} / R2 {model.score(X_te, y_te):.4f} / MAE {mean_absolute_error(y_te, pred):,.2f}")

    path = output_dir / "sales_amount_pipeline.joblib"
    joblib.dump(model, path)  # 저장
    reloaded = joblib.load(path)  # 재로딩
    print(f"모델 저장: {path.name} / 재로딩 R2 {reloaded.score(X_te, y_te):.4f}")
    print("재로딩 예측값 5개:", reloaded.predict(X_te.head()).round(0))


# 5) 지역·카테고리별 총매출 Plotly 막대차트 → HTML 저장
def plot_plotly(df, output_dir):
    summary = (df.groupby(["region", "category"], as_index=False)
               .agg(total=("amount", "sum")).sort_values("total", ascending=False))  # 총매출 집계
    fig = px.bar(summary, x="region", y="total", color="category", barmode="group",
                 title="지역·카테고리별 총매출")
    path = output_dir / "sales_by_region_category.html"
    fig.write_html(path, include_plotlyjs="cdn")  # 화면 출력 아닌 파일 저장
    print(f"\nPlotly HTML 저장: {path.name}")


# 전체 실습 4 작업 순서 실행
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # 출력 폴더 생성
    df = load_and_clean(CSV)  # 실습3 연계 정제
    plot_eda(df, OUTPUT_DIR)
    run_t_test(df)
    run_chi_square(df)
    train_and_save(df, OUTPUT_DIR)
    plot_plotly(df, OUTPUT_DIR)
    print("\n실습 4 전체 작업 완료")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:  # 파일 경로 오류
        print(f"[오류] 파일을 찾을 수 없습니다: {CSV}")
    except KeyError as e:  # 컬럼 누락
        print(f"[컬럼 오류] {e}")
    except ValueError as e:  # 데이터 오류
        print(f"[데이터 오류] {e}")
    except ImportError as e:  # 라이브러리 누락
        print(f"[라이브러리 오류] {e}")
    except Exception as e:  # 그 외 예외
        print(f"[오류] 예상하지 못한 오류: {e}")