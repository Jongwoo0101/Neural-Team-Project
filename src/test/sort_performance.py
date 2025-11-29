import os
import re
import math
from typing import List, Dict

# 보조 함수: 과학적 표기(1e-08 등)를 안전하게 float으로 변환
def parse_float(x: str):
    try:
        return float(x)
    except:
        # 혹시 모를 예외 대응
        return float(f"{x}")

# 로그 한 줄을 파싱하는 함수
# loss_acc_log.txt의 각 라인에서 하이퍼파라미터, Loss, ACC를 추출한다.
def parse_log_line(line: str) -> Dict:
    """
    예시 입력:

    Optimizer: Adam   , LR: 0.1    , Batch:  128, Iters:  500, Depth: 1,
    Act/Init: relu/relu   , L2: 1e-08   | FINAL LOSS: 2.186918 | Test ACC: 0.1542 |
    """

    # 하이퍼파라미터 정보 정규식 패턴
    pattern_hp = (
        r"Optimizer:\s*([A-Za-z0-9_]+)\s*,\s*"
        r"LR:\s*([0-9.eE+-]+)\s*,\s*"
        r"Batch:\s*([0-9]+)\s*,\s*"
        r"Iters:\s*([0-9]+)\s*,\s*"
        r"Depth:\s*([0-9]+)\s*,\s*"
        r"Act/Init:\s*([A-Za-z0-9_/]+)\s*,\s*"
        r"L2:\s*([0-9.eE+-]+)"
    )
    match_hp = re.search(pattern_hp, line)

    # 정상적으로 매칭되지 않는 경우는 무시
    if not match_hp:
        return None

    optimizer, lr, batch, iters, depth, act_init, l2 = match_hp.groups()

    # Loss / ACC 추출 패턴
    pattern_metrics = r"FINAL LOSS:\s*([0-9.eE+-]+)\s*\|\s*Test ACC:\s*([0-9.eE+-]+)"
    match_metrics = re.search(pattern_metrics, line)

    if not match_metrics:
        return None

    loss, acc = match_metrics.groups()

    # 파싱한 결과 반환
    return {
        "line": line.strip(),       # 원본 라인을 그대로 저장
        "optimizer": optimizer,
        "lr": lr,                   # 문자열 그대로 저장 (1e-08 유지 목적)
        "batch": int(batch),
        "iters": int(iters),
        "depth": int(depth),
        "act_init": act_init,
        "l2": l2,                   # 문자열 그대로 저장
        "loss": parse_float(loss),  # 숫자 변환
        "acc": parse_float(acc)     # 숫자 변환
    }

# 정렬된 로그 리스트를 파일로 출력
def write_sorted(filename: str, items: List[Dict]):
    with open(filename, "w") as f:
        for x in items:
            f.write(x["line"] + "\n")

# 상위 Percentile 분석 함수
# Top 1%, 3%, 6%, 12%에 대해 평균 Loss/ACC 및
# LR/L2 등장 빈도 등을 분석한다.
def analyze_top_percentiles(items: List[Dict], filename: str):
    # 퍼센타일 비율
    percentiles = [0.01, 0.03, 0.06, 0.12]

    with open(filename, "w") as f:
        f.write("=== Top Percentiles Analysis ===\n")

        n = len(items)

        for p in percentiles:
            # 최소 1개는 포함
            count = max(1, int(n * p))
            top_items = items[:count]  # ACC 기준 상위 모델만 사용

            # 평균 Loss/ACC 계산
            avg_loss = sum(x["loss"] for x in top_items) / count
            avg_acc = sum(x["acc"] for x in top_items) / count

            f.write(f"\n--- Top {int(p*100)}% ({count} models) ---\n")
            f.write(f"Avg Loss: {avg_loss:.6f}\n")
            f.write(f"Avg ACC : {avg_acc:.6f}\n")

            # 하이퍼파라미터 요약
            f.write("Hyperparameter summary:\n")

            lr_counts = {}
            l2_counts = {}

            for x in top_items:
                lr_counts[x["lr"]] = lr_counts.get(x["lr"], 0) + 1
                l2_counts[x["l2"]] = l2_counts.get(x["l2"], 0) + 1

            # LR 빈도수 출력
            f.write(" Most common LR:\n")
            for k, v in sorted(lr_counts.items(), key=lambda x: -x[1]):
                f.write(f"   LR {k} -> {v} times\n")

            # L2 빈도수 출력
            f.write(" Most common L2:\n")
            for k, v in sorted(l2_counts.items(), key=lambda x: -x[1]):
                f.write(f"   L2 {k} -> {v} times\n")

# 메인 함수
# ACC/Loss 정렬 및 퍼센타일 분석을 수행
def main():
    input_file = "../log/loss_acc_log.txt"  # 입력 로그 파일
    log_dir = "../log"                      # 출력 디렉토리

    os.makedirs(log_dir, exist_ok=True)

    # 로그 파일 읽기
    lines = open(input_file, "r").readlines()

    entries = []
    for line in lines:
        parsed = parse_log_line(line)
        if parsed:
            entries.append(parsed)

    # ACC 기준 내림차순 정렬
    sorted_acc = sorted(entries, key=lambda x: x["acc"], reverse=True)
    write_sorted(os.path.join(log_dir, "sort_acc.txt"), sorted_acc)

    # LOSS 기준 오름차순 정렬
    sorted_loss = sorted(entries, key=lambda x: x["loss"])
    write_sorted(os.path.join(log_dir, "sort_loss.txt"), sorted_loss)

    # 퍼센타일 분석 (ACC 기준 상위 모델 사용)
    analyze_top_percentiles(sorted_acc, os.path.join(log_dir, "sort_percentiles.txt"))

    print("Done: sort_acc.txt, sort_loss.txt, sort_percentiles.txt created.")


if __name__ == "__main__":
    main()
