"""
프로그램명: Practice 1 - 자료구조 집계·컴프리헨션·제너레이터
작성자: 최인서
작성 목적:
    Python_Practice1_Data.json의 매출 데이터를 활용하여
    1) 컴프리헨션 기반 지역별 총매출 계산
    2) Counter/defaultdict 기반 집계
    3) 리스트와 제너레이터의 메모리 크기 비교
    4) 월별·카테고리별 총매출 계산
    을 수행한다.

변경 내역:
    - 입력 파일이 일반 JSON 배열 또는 'sales = [...]' 형태여도 읽을 수 있도록 처리
    - 데이터 형식 검증 및 파일/문법 오류 예외 처리 추가
    - 평가 체크포인트 확인을 위한 assert 추가
"""


from __future__ import annotations

import ast # data.json의 scale()값을 불러오기 위함
import json
import sys # 메모리 크기 확인
from collections import Counter, defaultdict # 거래 건수 카운트, 없는 키 자동생성
from pathlib import Path
from typing import Any, Iterator #모든 자료형 허용


DATA_FILE = Path(__file__).with_name("Python_Practice1_Data.json")


def load_sales(path: Path) -> list[dict[str, Any]]:
    # 매출 데이터 파일을 읽고 기본 데이터 형식을 검증한다
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {path}"
        ) from error
    except OSError as error:
        raise OSError(f"파일을 읽는 중 오류가 발생했습니다: {error}") from error

    # 정상 JSON 배열을 먼저 시도한다.
    try:
        sales = json.loads(text)
    except json.JSONDecodeError:
        # 제공 파일처럼 'sales = [...]'인 Python 대입문 형태를 처리한다.
        try:
            tree = ast.parse(text, filename=str(path))
            if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
                raise ValueError("sales 변수에 리스트를 대입한 형식이 아닙니다.")

            assignment = tree.body[0]
            if not any(
                isinstance(target, ast.Name) and target.id == "sales"
                for target in assignment.targets
            ):
                raise ValueError("sales 변수 선언을 찾을 수 없습니다.")

            sales = ast.literal_eval(assignment.value)
            #jason 파일에서 scale()값 불러오기
        except (SyntaxError, ValueError) as error:
            raise ValueError(
                "파일이 JSON 배열 또는 'sales = [...]' 형식이 아닙니다."
            ) from error

    if not isinstance(sales, list):
        raise TypeError("최상위 데이터는 리스트여야 합니다.")

    required_keys = {"region", "category", "amount", "month"}
    for index, row in enumerate(sales):
        if not isinstance(row, dict):
            raise TypeError(f"{index}번째 데이터가 딕셔너리가 아닙니다.")
        missing_keys = required_keys - row.keys()
        if missing_keys:
            raise ValueError(
                f"{index}번째 데이터에 필수 키가 없습니다: {sorted(missing_keys)}"
            )
        if not isinstance(row["amount"], (int, float)):
            raise TypeError(f"{index}번째 amount가 숫자가 아닙니다.")

    return sales

# amount가 1000 이상인 거래를 리스트 컴프리헨션으로 필터링, 지역별 총매출을 딕셔너리 컴프리헨션으로 계산
def calculate_region_total(
    sales: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int | float]]:
    filtered_sales = [row for row in sales if row["amount"] >= 1000] # 리스트 컴프리헨션

    regions = {row["region"] for row in filtered_sales} # 집합 컴프리헨션(종복제거)

    # 지역 총매출 계산
    region_total = {
        region: sum(
            row["amount"]
            for row in filtered_sales
            if row["region"] == region
        )
        for region in regions
    }

    return filtered_sales, region_total


# ounter로 지역별 거래 건수를 계산, defaultdict로 카테고리별 amount 목록 생성
def aggregate_sales(
    sales: list[dict[str, Any]],
) -> tuple[Counter[str], defaultdict[str, list[int | float]]]:
    region_count = Counter(row["region"] for row in sales)

    category_amounts: defaultdict[str, list[int | float]] = defaultdict(list)
    for row in sales:
        category_amounts[row["category"]].append(row["amount"])

    return region_count, category_amounts


def yield_high_amount(
    sales: list[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    # amount가 1000보다 큰 행을 한 건씩 반환
    for row in sales:
        if row["amount"] > 1000:
            yield row


def calculate_month_category_total(
    sales: list[dict[str, Any]],
) -> dict[str, dict[str, int | float]]:
    # defaultdict로 month·category별 매출을 누적, 딕셔너리 컴프리헨션으로 일반 dict 형태로 변환
    grouped: defaultdict[str, defaultdict[str, int | float]] = defaultdict(
        lambda: defaultdict(int)
    )

    for row in sales:
        grouped[row["month"]][row["category"]] += row["amount"]

    return {
        month: dict(category_total)
        for month, category_total in sorted(grouped.items())
    }


def print_result(
    filtered_sales: list[dict[str, Any]],
    region_total: dict[str, int | float],
    region_count: Counter[str],
    category_amounts: defaultdict[str, list[int | float]],
    list_size: int,
    generator_size: int,
    month_category_total: dict[str, dict[str, int | float]],
) -> None:
    """실습 결과를 읽기 쉬운 형태로 출력한다."""
    sorted_region_total = dict(
        sorted(region_total.items(), key=lambda item: item[1], reverse=True)
    )
    top3 = list(sorted_region_total.items())[:3]

    print("=" * 60)
    print("1. amount >= 1000 거래 및 지역별 총매출")
    print("=" * 60)
    print(f"필터링된 거래 수: {len(filtered_sales)}건")
    print(f"지역별 총매출: {sorted_region_total}")
    print(f"총매출 TOP 3: {top3}")

    print("\n" + "=" * 60)
    print("2. 지역별 거래 건수 및 카테고리별 amount 리스트")
    print("=" * 60)
    print(f"지역별 거래 건수: {region_count.most_common()}")
    for category, amounts in sorted(category_amounts.items()):
        print(f"{category}: {amounts}")

    print("\n" + "=" * 60)
    print("3. 리스트와 제너레이터 메모리 크기 비교")
    print("=" * 60)
    print(f"리스트 크기: {list_size} bytes")
    print(f"제너레이터 크기: {generator_size} bytes")
    print(f"제너레이터가 더 작은가? {generator_size < list_size}")

    print("\n" + "=" * 60)
    print("4. 월별·카테고리별 총매출")
    print("=" * 60)
    for month, category_total in month_category_total.items():
        print(f"{month}: {category_total}")


def run_checkpoints(
    region_total: dict[str, int | float],
    region_count: Counter[str],
    list_size: int,
    generator_size: int,
) -> None:
    """교재의 Checkpoint 항목을 assert로 확인한다."""
    expected_region_total = {
        "서울": 17670,
        "부산": 4550,
        "대구": 8320,
        "인천": 11950,
        "광주": 4830,
        "대전": 6300,
        "울산": 7270,
        "세종": 5750,
    }
    expected_count_order = [
        ("서울", 14),
        ("부산", 13),
        ("대구", 13),
        ("인천", 12),
        ("광주", 12),
        ("대전", 12),
        ("울산", 12),
        ("세종", 12),
    ]
    expected_top3 = [
        ("서울", 17670),
        ("인천", 11950),
        ("대구", 8320),
    ]

    top3 = sorted(
        region_total.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:3]

    assert region_total == expected_region_total, "region_total 값이 다릅니다."
    assert region_count.most_common() == expected_count_order, (
        "Counter.most_common() 순서 또는 값이 다릅니다."
    )
    assert generator_size < list_size, (
        "제너레이터의 메모리 크기가 리스트보다 작지 않습니다."
    )
    assert top3 == expected_top3, "TOP 3가 금액 내림차순으로 정렬되지 않았습니다."

    print("\n모든 Checkpoint를 통과했습니다.")


def main() -> None:
    """전체 실습 과정을 순서대로 실행한다."""
    try:
        sales = load_sales(DATA_FILE)

        filtered_sales, region_total = calculate_region_total(sales)
        region_count, category_amounts = aggregate_sales(sales)

        # 제너레이터를 list로 변환하지 않고 객체 자체의 크기를 비교한다.
        high_amount_list = [
            row for row in sales if row["amount"] > 1000
        ]
        high_amount_generator = yield_high_amount(sales)

        list_size = sys.getsizeof(high_amount_list)
        generator_size = sys.getsizeof(high_amount_generator)

        month_category_total = calculate_month_category_total(sales)

        print_result(
            filtered_sales,
            region_total,
            region_count,
            category_amounts,
            list_size,
            generator_size,
            month_category_total,
        )
        run_checkpoints(
            region_total,
            region_count,
            list_size,
            generator_size,
        )

    except (FileNotFoundError, OSError, ValueError, TypeError) as error:
        print(f"[실행 오류] {error}", file=sys.stderr)
        sys.exit(1)
    except AssertionError as error:
        print(f"[Checkpoint 실패] {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
