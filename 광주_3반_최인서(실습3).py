# 프로그램: sales_100k.csv 집계 성능 비교 (Pandas EDA/IQR · Polars Lazy · DuckDB SQL)
# 기능: EDA→IQR 이상치 제거→region·category 집계(total·mean·count)→세 도구 timeit 비교
# 변경내역: v1.1 세 도구 동일 파이프라인 통일·중복 제거·예외처리 추가·null 그룹 일치

from pathlib import Path
import timeit
import pandas as pd
import polars as pl
import duckdb

# csv 경로 (실행 위치와 무관하게 스크립트 폴더 기준)
CSV = Path(__file__).resolve().parent / "sales_100k.csv"
# 세 도구 공통 반복 횟수 (공정 비교 위해 통일)
NUMBER = 7
GROUP_KEYS = ["region", "category"]
NEEDED = GROUP_KEYS + ["amount"]


# IQR 정상범위 계산 (Q1-1.5*IQR ~ Q3+1.5*IQR)
def iqr_bounds(s: pd.Series) -> tuple[float, float]:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


# Pandas 집계 (파일 로딩 → IQR 필터 → named aggregation → 정렬)
def pandas_agg(lo: float, hi: float) -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df = df[df["amount"].between(lo, hi)]
    # dropna=False: Polars/DuckDB처럼 null 그룹을 유지해 결과 일치
    return (
        df.groupby(GROUP_KEYS, dropna=False)
        .agg(total=("amount", "sum"), mean=("amount", "mean"), count=("amount", "count"))
        .reset_index()
        .sort_values("total", ascending=False)
    )


# Polars Lazy 집계 (scan_csv → filter → group_by → agg → sort → collect)
def polars_agg(lo: float, hi: float) -> pl.DataFrame:
    return (
        pl.scan_csv(CSV)
        .filter(pl.col("amount").is_between(lo, hi))
        .group_by(GROUP_KEYS)
        .agg(
            total=pl.col("amount").sum(),
            mean=pl.col("amount").mean(),
            count=pl.col("amount").count(),
        )
        .sort("total", descending=True)
        .collect()
    )


# DuckDB SQL 집계 (파일에 직접 SQL, 결과 DataFrame 반환)
def duckdb_agg(lo: float, hi: float) -> pd.DataFrame:
    return duckdb.sql(
        f"""
        SELECT region, category,
               SUM(amount)   AS total,
               AVG(amount)   AS mean,
               COUNT(amount) AS count
        FROM read_csv_auto('{CSV.as_posix()}')
        WHERE amount BETWEEN {lo} AND {hi}
        GROUP BY region, category
        ORDER BY total DESC
        """
    ).df()


def main() -> None:
    # 1) 로딩 + 기본 EDA
    df = pd.read_csv(CSV)
    if not set(NEEDED).issubset(df.columns):  # 필수 컬럼 검증
        raise KeyError(f"필수 컬럼 누락: {set(NEEDED) - set(df.columns)}")

    print("데이터 크기:", df.shape)
    df.info()
    print("\n기초 통계")
    print(df.describe())
    print("\n컬럼별 결측 개수")
    print(df.isna().sum())
    print("\n컬럼별 결측 비율(%)")
    print((df.isna().sum() / len(df) * 100).round(2))

    # 2) IQR 이상치 제거 (제거 전·후 행 수 출력, null amount는 필터에서 자동 제외)
    lo, hi = iqr_bounds(df["amount"])
    before = len(df)
    after = int(df["amount"].between(lo, hi).sum())
    print(f"\nIQR 정상범위: {lo:.2f} ~ {hi:.2f}")
    print(f"제거 전 행 수: {before:,} / 제거 후 행 수: {after:,} / 제거 건수: {before - after:,}")

    # 3) 세 도구 동일 집계 결과 출력
    print("\nPandas 집계")
    print(pandas_agg(lo, hi).to_string(index=False))
    print("\nPolars Lazy 집계")
    print(polars_agg(lo, hi))
    print("\nDuckDB SQL 집계")
    print(duckdb_agg(lo, hi).to_string(index=False))

    # 4) timeit 성능 비교 (세 도구 모두 파일→집계, number 통일)
    t_pd = timeit.timeit(lambda: pandas_agg(lo, hi), number=NUMBER)
    t_pl = timeit.timeit(lambda: polars_agg(lo, hi), number=NUMBER)
    t_db = timeit.timeit(lambda: duckdb_agg(lo, hi), number=NUMBER)
    print(f"\n평균 실행 시간 (number={NUMBER})")
    print(f"Pandas : {t_pd / NUMBER:.6f}초")
    print(f"Polars : {t_pl / NUMBER:.6f}초")
    print(f"DuckDB : {t_db / NUMBER:.6f}초")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError:  # 파일 경로 오류
        print(f"[오류] 파일을 찾을 수 없습니다: {CSV}")
    except KeyError as e:  # 컬럼 누락
        print(f"[오류] {e}")
    except Exception as e:  # 그 외 예외
        print(f"[오류] 실행 중 예외 발생: {e}")