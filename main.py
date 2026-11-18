import json
import math
import os
import time

EPSILON = 1e-9
DEFAULT_REPEAT = 10
BENCHMARK_SIZES = [3, 5, 13, 25]


class Matrix2D:
    """2차원 정사각 행렬을 저장하고 위치별 접근을 지원하는 자료구조."""

    def __init__(self, size, rows=None):
        self.size = size
        self.data = self._build_data(size, rows)

    def _build_data(self, size, rows):
        if rows is not None:
            return self._copy_float_rows(rows)
        empty_rows = []
        for _ in range(size):
            empty_rows.append([0.0] * size)
        return empty_rows

    def _copy_float_rows(self, rows):
        copied = []
        for row in rows:
            float_row = []
            for value in row:
                float_row.append(float(value))
            copied.append(float_row)
        return copied

    def get_value(self, row, col):
        return self.data[row][col]

    def set_value(self, row, col, value):
        self.data[row][col] = float(value)

    def to_flat_list(self):
        flat = []
        for row in self.data:
            for value in row:
                flat.append(value)
        return flat


def normalize_label(raw_label):
    """입력 라벨('+', 'cross', 'x')을 표준 라벨('Cross', 'X')로 정규화한다."""
    if not isinstance(raw_label, str):
        return None
    cleaned = raw_label.strip().lower()
    if cleaned in ("+", "cross"):
        return "Cross"
    if cleaned == "x":
        return "X"
    return None


def is_number(value):
    """bool과 NaN/Inf를 제외한 유한한 int 또는 float 숫자인지 판별한다."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


def is_valid_square_matrix(rows, expected_size):
    """2차원 리스트가 expected_size x expected_size 숫자 행렬인지 검증한다."""
    if not isinstance(rows, list) or len(rows) != expected_size:
        return False
    for row in rows:
        if not isinstance(row, list) or len(row) != expected_size:
            return False
        for value in row:
            if not is_number(value):
                return False
    return True


def parse_row_numbers(line, expected_len):
    """공백으로 구분된 한 줄의 문자열을 expected_len개의 유한 실수 리스트로 파싱한다."""
    tokens = line.strip().split()
    if len(tokens) != expected_len:
        return None
    numbers = []
    for token in tokens:
        try:
            value = float(token)
        except ValueError:
            return None
        if not math.isfinite(value):
            return None
        numbers.append(value)
    return numbers


def extract_pattern_size(pattern_key):
    """'size_{N}_{idx}' 형식의 키에서 정수 N을 추출한다."""
    parts = pattern_key.split("_")
    if len(parts) != 3 or parts[0] != "size" or not parts[2]:
        return None
    try:
        size = int(parts[1])
    except ValueError:
        return None
    if size <= 0:
        return None
    return size


def extract_filter_size(filter_key):
    """'size_{N}' 형식의 필터 키에서 정수 N을 추출한다."""
    parts = filter_key.split("_")
    if len(parts) != 2 or parts[0] != "size":
        return None
    try:
        size = int(parts[1])
    except ValueError:
        return None
    if size <= 0:
        return None
    return size


def compute_mac_2d(pattern, filter_matrix):
    """2차원 패턴과 필터의 위치별 곱셈 누적(MAC) 점수를 계산한다."""
    total = 0.0
    size = pattern.size
    for row in range(size):
        for col in range(size):
            total += pattern.get_value(row, col) * filter_matrix.get_value(row, col)
    return total


def compute_mac_1d(flat_pattern, flat_filter):
    """1차원으로 평탄화된 패턴과 필터의 MAC 점수를 계산한다."""
    total = 0.0
    length = len(flat_pattern)
    for idx in range(length):
        total += flat_pattern[idx] * flat_filter[idx]
    return total


def judge_scores(score_a, score_b, label_a, label_b, tie_label, epsilon=EPSILON):
    """허용오차(epsilon) 기준으로 두 점수를 비교해 승자 또는 동점 라벨을 반환한다."""
    if abs(score_a - score_b) < epsilon:
        return tie_label
    if score_a > score_b:
        return label_a
    return label_b


def measure_mac_2d_ms(pattern, filter_matrix, repeat=DEFAULT_REPEAT):
    """2차원 MAC 연산을 repeat회 반복 측정하여 평균 소요 시간(ms)을 반환한다."""
    start_time = time.perf_counter()
    for _ in range(repeat):
        compute_mac_2d(pattern, filter_matrix)
    elapsed_sec = time.perf_counter() - start_time
    return (elapsed_sec / repeat) * 1000.0


def measure_mac_1d_ms(flat_pattern, flat_filter, repeat=DEFAULT_REPEAT):
    """1차원 평탄화 MAC 연산을 repeat회 반복 측정하여 평균 소요 시간(ms)을 반환한다."""
    start_time = time.perf_counter()
    for _ in range(repeat):
        compute_mac_1d(flat_pattern, flat_filter)
    elapsed_sec = time.perf_counter() - start_time
    return (elapsed_sec / repeat) * 1000.0


def create_synthetic_matrix(size, pattern_type):
    """크기 size x size의 십자가('Cross') 또는 'X' 패턴 행렬을 생성한다."""
    matrix = Matrix2D(size)
    center = size // 2
    for idx in range(size):
        if pattern_type == "Cross":
            matrix.set_value(center, idx, 1.0)
            matrix.set_value(idx, center, 1.0)
        elif pattern_type == "X":
            matrix.set_value(idx, idx, 1.0)
            matrix.set_value(idx, size - 1 - idx, 1.0)
    return matrix


def read_single_row(row_index, size):
    """유효한 숫자 size개가 입력될 때까지 한 줄을 반복 입력받는다."""
    while True:
        raw_line = input(f"  {row_index + 1}행: ")
        parsed = parse_row_numbers(raw_line, size)
        if parsed is not None:
            return parsed
        print(f"입력 형식 오류: 각 줄에 {size}개의 숫자를 공백으로 구분해 입력하세요.")


def read_matrix_from_console(title, size):
    """콘솔에서 size x size 크기의 행렬을 한 줄씩 입력받아 Matrix2D로 반환한다."""
    print(f"{title} ({size}줄 입력, 공백 구분)")
    rows = []
    for row_index in range(size):
        rows.append(read_single_row(row_index, size))
    return Matrix2D(size, rows)


def print_performance_table(sample_pairs, sizes, repeat=DEFAULT_REPEAT):
    """크기별 평균 MAC 연산 시간(ms)과 연산 횟수(N^2) 표를 출력한다."""
    print(f"\n#---------------------------------------\n# 성능 분석 (평균/{repeat}회)\n#---------------------------------------")
    print(f"{'크기':<10} {'평균 시간(ms)':>14} {'연산 횟수':>12}")
    print("-" * 39)
    for size in sizes:
        pattern, filter_matrix = get_benchmark_pair(sample_pairs, size)
        avg_ms = measure_mac_2d_ms(pattern, filter_matrix, repeat)
        size_label = f"{size}x{size}"
        print(f"{size_label:<10} {avg_ms:>14.4f} {size * size:>12}")


def get_benchmark_pair(sample_pairs, size):
    """벤치마크에 사용할 (패턴, 필터) 쌍을 반환한다."""
    if size in sample_pairs:
        return sample_pairs[size]
    pattern = create_synthetic_matrix(size, "Cross")
    filter_matrix = create_synthetic_matrix(size, "Cross")
    return pattern, filter_matrix


def run_manual_mode():
    """모드 1: 사용자 입력(3x3)을 받아 MAC 점수와 판정 결과, 성능을 출력한다."""
    print("\n#---------------------------------------\n# [1] 필터 입력\n#---------------------------------------")
    filter_a = read_matrix_from_console("필터 A", 3)
    filter_b = read_matrix_from_console("필터 B", 3)
    print("✓ 필터 A, B 저장 완료 (3x3)")
    print("\n#---------------------------------------\n# [2] 패턴 입력\n#---------------------------------------")
    pattern = read_matrix_from_console("패턴", 3)
    print_manual_result(pattern, filter_a, filter_b)
    print_performance_table({3: (pattern, filter_a)}, [3])


def print_manual_result(pattern, filter_a, filter_b):
    """모드 1의 MAC 점수, 연산 시간, 판정 결과를 출력한다."""
    score_a = compute_mac_2d(pattern, filter_a)
    score_b = compute_mac_2d(pattern, filter_b)
    avg_ms = measure_mac_2d_ms(pattern, filter_a, DEFAULT_REPEAT)
    verdict = judge_scores(score_a, score_b, "A", "B", "판정 불가")
    print("\n#---------------------------------------\n# [3] MAC 결과\n#---------------------------------------")
    print(f"A 점수: {score_a}\nB 점수: {score_b}")
    print(f"연산 시간(평균/{DEFAULT_REPEAT}회): {avg_ms:.4f} ms")
    verdict_text = f"판정 불가 (|A-B| < {EPSILON})" if verdict == "판정 불가" else verdict
    print(f"판정: {verdict_text}")


def resolve_data_path(filename):
    """현재 작업 디렉터리 또는 스크립트 디렉터리에서 데이터 파일 경로를 찾는다."""
    if os.path.exists(filename):
        return filename
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, filename)


def load_json_file(file_path):
    """JSON 파일을 읽어 딕셔너리로 반환하며 실패 시 None을 반환한다."""
    resolved = resolve_data_path(file_path)
    try:
        with open(resolved, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as err:
        print(f"JSON 로드 오류: {err}")
        return None
    if not isinstance(data, dict):
        print("JSON 스키마 오류: 루트 객체가 딕셔너리가 아닙니다.")
        return None
    return data


def load_single_size_filters(raw_group, size):
    """단일 크기의 필터 그룹('cross', 'x')을 정규화 및 검증하여 딕셔너리로 반환한다."""
    if not isinstance(raw_group, dict):
        return None
    parsed = {}
    for raw_key, matrix_rows in raw_group.items():
        std_label = normalize_label(raw_key)
        if std_label is None or std_label in parsed or not is_valid_square_matrix(matrix_rows, size):
            return None
        parsed[std_label] = Matrix2D(size, matrix_rows)
    if "Cross" not in parsed or "X" not in parsed:
        return None
    return parsed


def load_filters_by_size(raw_filters):
    """전체 필터 딕셔너리를 크기별로 파싱해 반환한다."""
    print("\n#---------------------------------------\n# [1] 필터 로드\n#---------------------------------------")
    if not isinstance(raw_filters, dict):
        print("✗ filters 항목이 올바르지 않습니다.")
        return {}
    filters_by_size = {}
    for filter_key, raw_group in raw_filters.items():
        register_filter_group(filter_key, raw_group, filters_by_size)
    return filters_by_size


def register_filter_group(filter_key, raw_group, filters_by_size):
    """단일 크기 필터 그룹을 검증하고 filters_by_size에 등록한다."""
    size = extract_filter_size(filter_key)
    parsed = load_single_size_filters(raw_group, size) if size else None
    if parsed is None:
        print(f"✗ {filter_key:<7} 필터 스키마 오류로 로드 건너뜀")
        return
    filters_by_size[size] = parsed
    print(f"✓ {filter_key:<7} 필터 로드 완료 (Cross, X)")


def validate_pattern_payload(pattern_item, size):
    """패턴 항목의 input 행렬과 expected 라벨을 검증하고 (오류사유, 행렬, 기대라벨)을 반환한다."""
    if not isinstance(pattern_item, dict):
        return ("패턴 항목이 딕셔너리가 아님", None, None)
    rows = pattern_item.get("input")
    if not is_valid_square_matrix(rows, size):
        return (f"패턴과 필터({size}x{size}) 크기 불일치 또는 데이터 오류", None, None)
    expected_label = normalize_label(pattern_item.get("expected"))
    if expected_label is None:
        return (f"알 수 없는 expected 라벨: {pattern_item.get('expected')}", None, None)
    return (None, Matrix2D(size, rows), expected_label)


def validate_pattern_case(pattern_key, pattern_item, filters_by_size):
    """패턴 케이스의 키, 필터 존재 여부, payload를 검증해 반환한다."""
    size = extract_pattern_size(pattern_key)
    if size is None:
        return ("키 형식 오류(size_N_idx 아님)", None, None, None)
    if size not in filters_by_size:
        return (f"크기 {size}x{size}에 해당하는 필터 없음", size, None, None)
    error_reason, pattern_matrix, expected = validate_pattern_payload(pattern_item, size)
    return (error_reason, size, pattern_matrix, expected)


def evaluate_single_pattern(pattern_key, pattern_item, filters_by_size):
    """단일 패턴 케이스를 검증하고 MAC 판정 및 PASS/FAIL 결과를 반환한다."""
    error_reason, size, pattern_matrix, expected = validate_pattern_case(pattern_key, pattern_item, filters_by_size)
    if error_reason is not None:
        return {"key": pattern_key, "passed": False, "reason": error_reason, "size": size}
    score_cross = compute_mac_2d(pattern_matrix, filters_by_size[size]["Cross"])
    score_x = compute_mac_2d(pattern_matrix, filters_by_size[size]["X"])
    predicted = judge_scores(score_cross, score_x, "Cross", "X", "UNDECIDED")
    passed = predicted == expected
    reason = "" if passed else build_fail_reason(predicted)
    return build_case_result(pattern_key, passed, reason, size, pattern_matrix, score_cross, score_x, predicted, expected)


def build_fail_reason(predicted):
    """예측값에 따른 실패 사유 문자열을 생성한다."""
    if predicted == "UNDECIDED":
        return "동점(UNDECIDED) 처리 규칙에 따라 FAIL"
    return "예측 라벨과 expected 불일치"


def build_case_result(key, passed, reason, size, pattern, score_cross, score_x, predicted, expected):
    """평가 완료된 패턴 케이스의 결과 딕셔너리를 구성한다."""
    return {
        "key": key,
        "passed": passed,
        "reason": reason,
        "size": size,
        "pattern": pattern,
        "score_cross": score_cross,
        "score_x": score_x,
        "predicted": predicted,
        "expected": expected,
    }


def print_pattern_result(result):
    """단일 패턴 케이스의 판정 결과를 콘솔에 출력한다."""
    print(f"--- {result['key']} ---")
    if "score_cross" not in result:
        print(f"판정: ERROR | FAIL ({result['reason']})")
        return
    status = "PASS" if result["passed"] else f"FAIL ({result['reason']})"
    print(f"Cross 점수: {result['score_cross']}")
    print(f"X 점수: {result['score_x']}")
    print(f"판정: {result['predicted']} | expected: {result['expected']} | {status}")


def collect_benchmark_pairs(results, filters_by_size):
    """로드된 필터와 패턴을 크기별 벤치마크 입력 쌍으로 수집한다."""
    pairs = {}
    for result in results:
        size = result.get("size")
        pattern = result.get("pattern")
        if size is not None and pattern is not None and size in filters_by_size:
            if size not in pairs:
                pairs[size] = (pattern, filters_by_size[size]["Cross"])
    return pairs


def collect_failed_cases(results):
    """평가 결과 리스트에서 실패한 케이스만 추출한다."""
    failed_list = []
    for result in results:
        if not result["passed"]:
            failed_list.append(result)
    return failed_list


def print_summary_report(results):
    """전체 테스트 수, 통과 수, 실패 수 및 실패 케이스 목록을 요약 출력한다."""
    failed_list = collect_failed_cases(results)
    passed_count = len(results) - len(failed_list)
    print("\n#---------------------------------------\n# [4] 결과 요약\n#---------------------------------------")
    print(f"총 테스트: {len(results)}개\n통과: {passed_count}개\n실패: {len(failed_list)}개")
    if not failed_list:
        return
    print("\n실패 케이스:")
    for item in failed_list:
        print(f"- {item['key']}: {item['reason']}")


def run_json_mode(file_path="data.json"):
    """모드 2: data.json을 로드해 전체 패턴을 분석하고 성능 및 요약을 출력한다."""
    data = load_json_file(file_path)
    if data is None:
        return
    filters_by_size = load_filters_by_size(data.get("filters"))
    raw_patterns = data.get("patterns")
    if not isinstance(raw_patterns, dict):
        print("JSON 스키마 오류: patterns 항목이 올바르지 않습니다.")
        return
    results = evaluate_all_patterns(raw_patterns, filters_by_size)
    sample_pairs = collect_benchmark_pairs(results, filters_by_size)
    print_performance_table(sample_pairs, BENCHMARK_SIZES)
    print_summary_report(results)


def evaluate_all_patterns(raw_patterns, filters_by_size):
    """모든 패턴 항목을 순회하며 평가하고 결과를 출력한다."""
    print("\n#---------------------------------------\n# [2] 패턴 분석 (라벨 정규화 적용)\n#---------------------------------------")
    results = []
    for pattern_key, pattern_item in raw_patterns.items():
        result = evaluate_single_pattern(pattern_key, pattern_item, filters_by_size)
        print_pattern_result(result)
        results.append(result)
    return results


def read_pattern_size():
    """보너스 모드용 행렬 크기 N(3 이상 정수)을 입력받는다."""
    while True:
        raw = input("패턴 크기 N 입력 (예: 5, 13, 25): ").strip()
        try:
            size = int(raw)
        except ValueError:
            print("입력 오류: 정수를 입력하세요.")
            continue
        if size >= 3:
            return size
        print("입력 오류: 3 이상의 정수를 입력하세요.")


def print_matrix_preview(title, matrix):
    """생성된 행렬을 문자로 시각화하여 출력한다."""
    print(f"\n[{title} ({matrix.size}x{matrix.size})]")
    for row in range(matrix.size):
        line_tokens = []
        for col in range(matrix.size):
            mark = "■" if matrix.get_value(row, col) > 0 else "□"
            line_tokens.append(mark)
        print(" ".join(line_tokens))


def run_bonus_mode():
    """모드 3(보너스): 패턴 자동 생성 및 2차원 vs 1차원 평탄화 MAC 성능을 비교한다."""
    size = read_pattern_size()
    cross_matrix = create_synthetic_matrix(size, "Cross")
    x_matrix = create_synthetic_matrix(size, "X")
    if size <= 15:
        print_matrix_preview("자동 생성 Cross 패턴", cross_matrix)
        print_matrix_preview("자동 생성 X 패턴", x_matrix)
    repeat_count = 100
    avg_2d = measure_mac_2d_ms(cross_matrix, cross_matrix, repeat_count)
    avg_1d = measure_mac_1d_ms(cross_matrix.to_flat_list(), cross_matrix.to_flat_list(), repeat_count)
    print(f"\n[최적화 전/후 비교 ({size}x{size}, {repeat_count}회 평균)]")
    print(f"- 2차원 접근 MAC: {avg_2d:.5f} ms\n- 1차원 평탄화 MAC: {avg_1d:.5f} ms")


def select_mode():
    """사용자로부터 실행할 시뮬레이터 모드를 선택받는다."""
    print("=== Mini NPU Simulator ===\n[모드 선택]")
    print("1. 사용자 입력 (3x3)\n2. data.json 분석\n3. 보너스: 패턴 생성 및 1차원 최적화 비교")
    while True:
        choice = input("선택: ").strip()
        if choice in ("1", "2", "3"):
            return choice
        print("입력 오류: 1, 2, 3 중에서 선택하세요.")


def main():
    """프로그램 진입점: 모드를 선택받아 해당 시뮬레이션 흐름을 실행한다."""
    try:
        mode = select_mode()
        if mode == "1":
            run_manual_mode()
        elif mode == "2":
            run_json_mode()
        else:
            run_bonus_mode()
    except (EOFError, KeyboardInterrupt):
        print("\n프로그램을 종료합니다.")


if __name__ == "__main__":
    main()
