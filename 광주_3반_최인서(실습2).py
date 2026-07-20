"""
프로그램명: Practice 2
작성자 : 광주_3반_최인서
전체설명:
    CSV 데이터를 안전하게 읽고 Pydantic v2로 검증한 뒤,
    정상 데이터는 CSV로, 오류 데이터는 JSON으로 저장하고 재검증한다.

변경 내역:
    - safe_load_csv()에 try-except-finally와 logging 적용
    - SalesRecord 모델에 필수값·양수 조건 적용
    - ValidationError를 정상 데이터와 오류 데이터로 분리
    - 중복 저장 코드를 함수로 분리하고 Checkpoint assert 추가
"""

import csv
import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "practice2_input.csv"
VALID_FILE = BASE_DIR / "practice2_valid.csv"
ERROR_FILE = BASE_DIR / "practice2_errors.json"

logger = logging.getLogger("practice2")
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)

# Checkpoint 확인용 원본 데이터: 정상 4건, 오류 3건
RAW_DATA = [
    {"month": "2024-01", "region": "서울", "amount": 1500, "category": "전자"},
    {"month": "2024-02", "region": "부산", "amount": 800, "category": "의류"},
    {"month": "2024-03", "region": "대구", "amount": 950},
    {"month": "2024-04", "region": "인천", "amount": 1200, "category": "식품"},
    {"month": "", "region": "광주", "amount": 700, "category": "의류"},
    {"month": "2024-02", "region": "", "amount": 900, "category": "전자"},
    {"month": "2024-03", "region": "대전", "amount": 0, "category": "식품"},
]


# 월·지역·금액을 검증하는 매출 데이터 모델 (Pydantic v2 스키마 정의)
class SalesRecord(BaseModel):

    month: str
    region: str
    amount: float = Field(gt=0)
    category: str | None = None

    # month와 region은 공백 문자열도 허용x
    @field_validator("month", "region")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("비어 있을 수 없습니다.")
        return value

    # CSV의 빈 category를 None으로 변환
    @field_validator("category", mode="before")
    @classmethod
    def empty_category_to_none(cls, value: object) -> object:
        return None if value == "" else value


# 실습용 raw_data를 CSV 파일로 저장
def create_input_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["month", "region", "amount", "category"]

    try:
        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("입력 CSV 생성 완료: %s", path.name)
    except OSError as error:
        raise OSError(f"입력 CSV 저장 실패: {error}") from error


# CSV 파일을 안전하게 읽고 dict 리스트로 반환
def safe_load_csv(path: Path) -> list[dict[str, str]] | None:
    try:
        with path.open(newline="", encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
    except FileNotFoundError:
        logger.error("파일을 찾을 수 없습니다: %s", path.name)
        return None
    except (OSError, csv.Error) as error:
        logger.error("CSV 로딩 실패: %s", error)
        return None
    else:
        logger.info("CSV 로딩 성공: %d건", len(rows))
        return rows
    finally:
        print("로딩 종료")


# Pydantic 검증 결과를 valid와 errors로 분리
def validate_records(
    raw_data: list[dict[str, str]],
) -> tuple[list[SalesRecord], list[dict[str, object]]]:
    valid: list[SalesRecord] = []
    errors: list[dict[str, object]] = []

    for row_number, row in enumerate(raw_data, start=1):
        try:
            valid.append(SalesRecord.model_validate(row))
        except ValidationError as error:
            error_info = {"row": row_number, "error": str(error)}
            errors.append(error_info)
            logger.warning("%d행 검증 실패: %s", row_number, error)

    return valid, errors


# 정상 데이터는 CSV, 오류 데이터는 JSON으로 저장
def save_results(
    valid: list[SalesRecord],
    errors: list[dict[str, object]],
) -> None:
    valid_rows = [record.model_dump() for record in valid]

    try:
        with VALID_FILE.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["month", "region", "amount", "category"],
            )
            writer.writeheader()
            writer.writerows(valid_rows)

        with ERROR_FILE.open("w", encoding="utf-8") as file:
            json.dump(errors, file, ensure_ascii=False, indent=2)

        logger.info("결과 저장 완료: %s, %s", VALID_FILE.name, ERROR_FILE.name)
    except OSError as error:
        raise OSError(f"결과 파일 저장 실패: {error}") from error


def main() -> None:
    try:
        create_input_csv(INPUT_FILE, RAW_DATA)

        # 존재하지 않는 파일은 None을 반환
        assert safe_load_csv(BASE_DIR / "missing.csv") is None

        raw_data = safe_load_csv(INPUT_FILE)
        if raw_data is None:
            raise RuntimeError("입력 데이터를 불러오지 못했습니다.")

        valid, errors = validate_records(raw_data)
        save_results(valid, errors)

        reloaded = safe_load_csv(VALID_FILE)
        if reloaded is None:
            raise RuntimeError("저장한 정상 데이터를 다시 읽지 못했습니다.")

        # 테스트 데이터와 출력 결과 비교 확인
        assert len(valid) == 4, "valid 데이터는 4건이어야 합니다."
        assert len(errors) == 3, "errors 데이터는 3건이어야 합니다."
        assert len(reloaded) == 4, "재로딩된 데이터는 4건이어야 합니다."

        print("\n" + "=" * 50)
        print(f"검증 성공: {len(valid)}건")
        print(f"검증 실패: {len(errors)}건")
        print(f"재로딩 확인: {len(reloaded)}건")
        print("모든 Checkpoint를 통과했습니다.")

    except (OSError, RuntimeError, AssertionError) as error:
        logger.error("파이프라인 실행 실패: %s", error)


if __name__ == "__main__":
    main()
