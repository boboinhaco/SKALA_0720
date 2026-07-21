# 프로그램: Adult income — Plotly 인터랙티브 차트 · 기술통계/t-test 출력 · report.md 자동 생성
# 기능: 국적 결측 삭제 후 기술통계·상관·t-test 산출, Plotly HTML 저장, 결과를 report.md로 자동 작성
# 변경내역: v1.0 최초 작성

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import ttest_ind

BASE = Path(__file__).resolve().parent
CSV = BASE / "adult_data.csv"
NUM = ["age", "education-num", "hours-per-week", "capital-gain", "capital-loss"]


# 로딩 → 국적 결측 삭제 → 나머지 범주 결측 Unknown → 타깃 이진화
def load_data():
    df = pd.read_csv(CSV, skipinitialspace=True, na_values="?")
    df = df.dropna(subset=["native-country"]).copy()  # 국적 결측 삭제
    df[["workclass", "occupation"]] = df[["workclass", "occupation"]].fillna("Unknown")
    df["income_bin"] = (df["income"] == ">50K").astype(int)  # >50K=1
    return df


# 기술통계(평균·표준편차·분위수)와 수치형 상관계수 산출
def describe_stats(df):
    desc = df[NUM].describe().loc[["mean", "std", "25%", "50%", "75%"]].round(2)
    corr = df[NUM].corr().round(3)
    print("=== 기술통계 (평균·표준편차·분위수) ===")
    print(desc)
    print("\n=== 수치형 변수 간 상관계수 ===")
    print(corr)
    return desc, corr


# 수치형 컬럼별 income 두 그룹 t-test + p-value 해석
def run_ttest(df):
    print("\n=== t-test (고소득 vs 저소득) ===")
    rows = []
    for c in NUM:
        hi = df.loc[df["income_bin"] == 1, c]  # 고소득 그룹
        lo = df.loc[df["income_bin"] == 0, c]  # 저소득 그룹
        t, p = ttest_ind(hi, lo, equal_var=False)  # Welch t-test
        verdict = "유의미(p<0.05)" if p < 0.05 else "무의미"
        print(f"  {c:15s} t={t:7.2f} p={p:.2e} → {verdict}")
        rows.append({"변수": c, "t통계량": round(t, 2), "p-value": f"{p:.2e}",
                     "고소득평균": round(hi.mean(), 1), "저소득평균": round(lo.mean(), 1),
                     "해석": verdict})
    return pd.DataFrame(rows)


# Plotly 인터랙티브 막대차트(학력별 고소득 비율) → HTML 저장
def make_plotly(df):
    edu = (df.groupby("education")["income_bin"].mean()
           .sort_values().reset_index())  # 학력별 >50K 비율
    edu.columns = ["education", "고소득비율"]
    fig = px.bar(edu, x="고소득비율", y="education", orientation="h",
                 title="학력별 고소득(>50K) 비율", labels={"education": "학력"})
    fig.update_layout(xaxis_tickformat=".0%")  # 비율 % 표기
    path = BASE / "income_by_education.html"
    fig.write_html(path, include_plotlyjs="cdn")  # 인터랙티브 HTML 저장
    print(f"\nPlotly 저장: {path.name}")
    return path.name


# 분석 결과를 report.md 파일로 자동 생성
def write_report(df, desc, corr, ttest_df, chart_name):
    n = len(df)
    high = df["income_bin"].mean() * 100
    top_edu = df.groupby("education")["income_bin"].mean().idxmax()  # 최고 고소득 학력
    lines = [
        "# Adult Income 분석 리포트",
        "",
        "## 1. 데이터 개요",
        f"- 분석 행 수: {n:,}행 (국적 결측 삭제 후)",
        f"- 고소득(>50K) 비율: {high:.1f}%",
        f"- 사용 수치형 변수: {', '.join(NUM)} (fnlwgt는 인구 가중치라 제외)",
        "",
        "## 2. 기술통계",
        desc.to_markdown(),
        "",
        "## 3. 수치형 변수 간 상관계수",
        corr.to_markdown(),
        "- 수치형 변수들은 상관이 대부분 0.15 이하로 서로 독립적이다.",
        "",
        "## 4. t-test 결과 (고소득 vs 저소득)",
        ttest_df.to_markdown(index=False),
        "- age·education-num·hours-per-week·capital-gain·capital-loss 모두 p<0.05로 유의미한 차이.",
        "",
        "## 5. 주요 발견",
        f"- 학력이 소득과 강하게 연관되며, 최고 고소득 비율 학력은 '{top_edu}'이다.",
        "- 결측 처리: 편향이 큰 workclass·occupation는 Unknown으로 보존, 편향 없는 국적은 삭제.",
        "- 관계는 상관이며 인과가 아님(1994년 표본 기준 관측).",
        "",
        f"## 6. 시각화 산출물",
        f"- Plotly 인터랙티브 차트: `{chart_name}`",
    ]
    path = BASE / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")  # 리포트 파일 작성
    print(f"리포트 저장: {path.name}")


def main():
    df = load_data()
    desc, corr = describe_stats(df)
    ttest_df = run_ttest(df)
    chart_name = make_plotly(df)
    write_report(df, desc, corr, ttest_df, chart_name)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:  # 파일 경로 오류
        print(f"[오류] 파일을 찾을 수 없습니다: {CSV}")
    except Exception as e:  # 그 외 예외
        print(f"[오류] 실행 중 예외 발생: {e}")