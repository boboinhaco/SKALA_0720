"""
프로그램명: Practice 1
작성자: 최인서
전체 설명:
    매출 데이터를 활용해 지역별·카테고리별·월별 매출을 집계하고,
    리스트와 제너레이터의 메모리 크기를 비교한다.

변경 내역:
    - JSON 배열과 'sales = [...]' 형식 모두 지원
    - 파일 및 데이터 형식 오류 처리
    - 중복 계산 제거
    - 교재 Checkpoint 테스트 실시
"""

import ast
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


DATA_FILE = Path(__file__).with_name("Python_Practice1_Data.json")

# 정답 비교 코드
EXPECTED_REGION_TOTAL = {
    "서울": 17670,
    "부산": 4550,
    "대구": 8320,
    "인천": 11950,
    "광주": 4830,
    "대전": 6300,
    "울산": 7270,
    "세종": 5750,
}

EXPECTED_COUNT_ORDER = [
    ("서울", 14),
    ("부산", 13),
    ("대구", 13),
    ("인천", 12),
    ("광주", 12),
    ("대전", 12),
    ("울산", 12),
    ("세종", 12),
]


# 데이터 파일을 읽고 형식을 검사한다.
def load_sales(path: Path) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {path.name}"
        ) from error
    except OSError as error:
        raise OSError(f"파일을 읽을 수 없습니다: {error}") from error

    try:
        sales = json.loads(text)
    except json.JSONDecodeError:
        try:
            tree = ast.parse(text)
            assignment = tree.body[0]

            if not isinstance(assignment, ast.Assign):
                raise ValueError

            sales = ast.literal_eval(assignment.value)
        except (SyntaxError, ValueError, IndexError, AttributeError) as error:
            raise ValueError(
                "파일은 JSON 배열 또는 'sales = [...]' 형식이어야 합니다."
            ) from error

    required_keys = {"region", "category", "amount", "month"}

    if not isinstance(sales, list):
        raise TypeError("최상위 데이터는 리스트여야 합니다.")

    for index, row in enumerate(sales):
        if not isinstance(row, dict):
            raise TypeError(f"{index}번째 데이터가 딕셔너리가 아닙니다.")

        missing_keys = required_keys - row.keys()
        if missing_keys:
            raise ValueError(
                f"{index}번째 데이터에 필수 항목이 없습니다: {sorted(missing_keys)}"
            )

        if not isinstance(row["amount"], (int, float)):
            raise TypeError(f"{index}번째 amount는 숫자여야 합니다.")

    return sales


# amount가 1000보다 큰 거래를 한 건씩 반환한다.
def yield_high_amount(sales: list[dict]):
    for row in sales:
        if row["amount"] > 1000:
            yield row


# 실습 전체 과정을 실행한다.
def main() -> None:
    try:
        sales = load_sales(DATA_FILE)

        # 1. amount가 1000 이상인 거래와 지역별 총매출
        filtered_sales = [row for row in sales if row["amount"] >= 1000]
        regions = {row["region"] for row in filtered_sales}

        region_total = {
            region: sum(
                row["amount"]
                for row in filtered_sales
                if row["region"] == region
            )
            for region in regions
        }

        # 2. 지역별 거래 건수와 카테고리별 amount 목록
        region_count = Counter(row["region"] for row in sales)
        category_amounts = defaultdict(list)

        for row in sales:
            category_amounts[row["category"]].append(row["amount"])

        # 3. 리스트와 제너레이터 메모리 크기 비교
        high_amount_list = [row for row in sales if row["amount"] > 1000]
        high_amount_generator = yield_high_amount(sales)

        list_size = sys.getsizeof(high_amount_list)
        generator_size = sys.getsizeof(high_amount_generator)

        # 4. 월별·카테고리별 총매출
        month_category_total = defaultdict(lambda: defaultdict(int))

        for row in sales:
            month_category_total[row["month"]][row["category"]] += row["amount"]

        month_category_total = {
            month: dict(category_total)
            for month, category_total in sorted(month_category_total.items())
        }

        # 지역별 총매출을 한 번만 정렬해 출력과 테스트에 함께 사용한다.
        sorted_region_total = sorted(
            region_total.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        top3 = sorted_region_total[:3]

        expected_top3 = sorted(
            EXPECTED_REGION_TOTAL.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:3]

        # 교재 Checkpoint 테스트
        assert region_total == EXPECTED_REGION_TOTAL, "지역별 총매출이 다릅니다."
        assert region_count.most_common() == EXPECTED_COUNT_ORDER, (
            "지역별 거래 건수 또는 순서가 다릅니다."
        )
        assert generator_size < list_size, (
            "제너레이터 크기가 리스트보다 작지 않습니다."
        )
        assert top3 == expected_top3, "TOP 3 결과가 올바르지 않습니다."

        print("=" * 60)
        print("1. amount >= 1000 거래 및 지역별 총매출")
        print(f"필터링된 거래 수: {len(filtered_sales)}건")
        print(f"지역별 총매출: {dict(sorted_region_total)}")
        print(f"총매출 TOP 3: {top3}")

        print("\n" + "=" * 60)
        print("2. 지역별 거래 건수 및 카테고리별 amount")
        print(f"지역별 거래 건수: {region_count.most_common()}")

        for category, amounts in sorted(category_amounts.items()):
            print(f"{category}: {amounts[:5]} ... (총 {len(amounts)}건)")

        print("\n" + "=" * 60)
        print("3. 리스트와 제너레이터 메모리 크기 비교")
        print(f"리스트 크기: {list_size} bytes")
        print(f"제너레이터 크기: {generator_size} bytes")

        print("\n" + "=" * 60)
        print("4. 월별·카테고리별 총매출")
        for month, category_total in month_category_total.items():
            print(f"{month}: {category_total}")

        print("\n모든 Checkpoint를 통과했습니다.")

    except (FileNotFoundError, OSError, ValueError, TypeError) as error:
        print(f"[실행 오류] {error}", file=sys.stderr)
        sys.exit(1)
    except AssertionError as error:
        print(f"[Checkpoint 실패] {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()