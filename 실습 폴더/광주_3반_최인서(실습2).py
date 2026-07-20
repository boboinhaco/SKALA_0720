"""
프로그램명: Practice 2
작성자 : 광주_3반_최인서
전체설명:
    CSV 데이터를 안전하게 읽고 Pydantic v2로 검증한 뒤,
    정상 데이터는 CSV로, 오류 데이터는 JSON으로 저장하고 재검증한다.

변경 내역:
    - 제공 JSON 파일을 실제 입력 데이터로 사용
    - 원본 데이터는 변경하지 않고 검증 오류 3건을 별도로 구성
    - 파일 I/O, Pydantic 검증, 결과 저장의 중복 코드 최소화
    - 교재 Checkpoint assert 적용
"""

import csv
import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator


BASE_DIR = Path(__file__).parent
SOURCE_FILE = BASE_DIR / "Python_Practice2_Data.json"
INPUT_FILE = BASE_DIR / "practice2_input.csv"
VALID_FILE = BASE_DIR / "practice2_valid.csv"
ERROR_FILE = BASE_DIR / "practice2_errors.json"
FIELDS = ["month", "region", "amount", "category"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("practice2")


# 월·지역·금액을 검증하는 Pydantic v2 스키마
class SalesRecord(BaseModel):
    month: str
    region: str
    amount: float = Field(gt=0)
    category: str | None = None

    # month와 region은 빈 문자열이나 공백만 있는 값을 허용x
    @field_validator("month", "region")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("비어 있을 수 없습니다.")
        return value

    # CSV의 빈 category는 선택값인 None으로 변환
    @field_validator("category", mode="before")
    @classmethod
    def empty_category_to_none(cls, value: object) -> object:
        return None if value == "" else value


# 제공 JSON 파일을 읽어 딕셔너리 리스트로 반환
def load_json(path: Path) -> list[dict]:
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"JSON 파일을 찾을 수 없습니다: {path.name}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON 형식이 올바르지 않습니다: {error}") from error
    except OSError as error:
        raise OSError(f"JSON 파일을 읽을 수 없습니다: {error}") from error

    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError("JSON의 최상위 데이터는 딕셔너리 리스트여야 합니다.")

    return data


# 원본 JSON에서 정상 4건과 오류 검증용 3건을 구성
def build_practice_data(source: list[dict]) -> list[dict]:
    if len(source) < 7:
        raise ValueError("실습 데이터를 만들기 위해 JSON 데이터가 최소 7건 필요합니다.")

    valid_rows = [row.copy() for row in source[:4]]
    error_rows = [
        {**source[4], "month": ""},
        {**source[5], "region": ""},
        {**source[6], "amount": 0},
    ]
    return valid_rows + error_rows


# 딕셔너리 리스트를 CSV 파일로 저장
def save_csv(path: Path, rows: list[dict]) -> None:
    try:
        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    except OSError as error:
        raise OSError(f"CSV 저장 실패({path.name}): {error}") from error


# CSV 파일을 읽고, 실패하면 None을 반환
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


# 각 행을 SalesRecord로 검증해 정상과 오류 데이터로 분리
def validate_records(
    raw_data: list[dict[str, str]],
) -> tuple[list[SalesRecord], list[dict[str, object]]]:
    valid: list[SalesRecord] = []
    errors: list[dict[str, object]] = []

    for row_number, row in enumerate(raw_data, start=1):
        try:
            valid.append(SalesRecord.model_validate(row))
        except ValidationError as error:
            errors.append({"row": row_number, "error": str(error)})
            logger.warning("%d행 검증 실패: %s", row_number, error)

    return valid, errors


# 정상 데이터는 CSV, 오류 데이터는 JSON
def save_results(
    valid: list[SalesRecord],
    errors: list[dict[str, object]],
) -> None:
    save_csv(VALID_FILE, [record.model_dump() for record in valid])

    try:
        with ERROR_FILE.open("w", encoding="utf-8") as file:
            json.dump(errors, file, ensure_ascii=False, indent=2)
    except OSError as error:
        raise OSError(f"JSON 저장 실패({ERROR_FILE.name}): {error}") from error

    logger.info("결과 저장 완료: %s, %s", VALID_FILE.name, ERROR_FILE.name)


def main() -> None:
    try:
        source_data = load_json(SOURCE_FILE)
        practice_data = build_practice_data(source_data)
        save_csv(INPUT_FILE, practice_data)

        # 존재하지 않는 파일은 None을 반환
        assert safe_load_csv(BASE_DIR / "missing.csv") is None

        raw_data = safe_load_csv(INPUT_FILE)
        if raw_data is None:
            raise RuntimeError("입력 CSV를 불러오지 못했습니다.")

        valid, errors = validate_records(raw_data)
        save_results(valid, errors)

        reloaded = safe_load_csv(VALID_FILE)
        if reloaded is None:
            raise RuntimeError("정상 데이터 CSV를 다시 읽지 못했습니다.")

        # 체크포인트와 출력 값이 같아야 함
        assert len(valid) == 4, "valid 데이터는 4건이어야 합니다."
        assert len(errors) == 3, "errors 데이터는 3건이어야 합니다."
        assert len(reloaded) == 4, "재로딩된 데이터는 4건이어야 합니다."

        print("\n" + "=" * 50)
        print(f"원본 JSON 데이터: {len(source_data)}건")
        print(f"실습 검증 데이터: {len(practice_data)}건")
        print(f"검증 성공: {len(valid)}건")
        print(f"검증 실패: {len(errors)}건")
        print(f"재로딩 확인: {len(reloaded)}건")
        print("모든 Checkpoint를 통과했습니다.")

    except (FileNotFoundError, OSError, ValueError, RuntimeError, AssertionError) as error:
        logger.error("파이프라인 실행 실패: %s", error)


if __name__ == "__main__":
    main()