'''
하이퍼파라미터 탐색 결과로 생성된 CSV 로그 파일
(loss_acc_log.csv)을 분석하여 다음의 결과물을 자동 생성한다.

[1] 정확도 기준 정렬 (sort_acc.csv)
    - final_acc 기준 내림차순 정렬
    - 가장 성능이 좋은 모델 조합을 빠르게 파악하기 위함

[2] 손실 기준 정렬 (sort_loss.csv)
    - final_loss 기준 오름차순 정렬
    - 최적 수렴 특성을 보인 모델 조합을 식별하기 위함

[3] 상위 퍼센타일 분석 (sort_percentiles.txt)
    - Top 1%, 3%, 6%, 12% 모델을 기준으로 평균 Loss, 평균 ACC 계산
    - 최상위 모델들이 공통적으로 가지는 하이퍼파라미터 경향 분석
        • 가장 자주 등장한 Learning Rate
        • 가장 자주 등장한 L2(Weight Decay)
        • 전체 모델 상세 정보 출력
'''

import os
import pandas as pd

NUMERIC_COLS = [
    "lr", "batch_size", "max_iterations", "max_epochs",
    "weight_decay_lambda", "dropout_ration",
    "final_loss", "final_acc"
]

# 퍼센타일 분석 함수
def analyze_top_percentiles(df, filename):
    percentiles = [0.01, 0.03, 0.06, 0.12]

    with open(filename, "w") as f:
        f.write("=== Top Percentiles Analysis ===\n")

        n = len(df)

        for p in percentiles:
            count = max(1, int(n * p))
            top_df = df.head(count)

            avg_loss = top_df["final_loss"].mean()
            avg_acc = top_df["final_acc"].mean()

            f.write(f"\n--- Top {int(p * 100)}% ({count} models) ---\n")
            f.write(f"Avg Loss: {avg_loss:.6f}\n")
            f.write(f"Avg ACC : {avg_acc:.6f}\n")

            f.write("Hyperparameter summary:\n")

            # LR 빈도
            f.write(" Most common LR:\n")
            for lr, cnt in top_df["lr"].value_counts().items():
                f.write(f"   LR {lr} -> {cnt} times\n")

            # L2 빈도
            f.write(" Most common L2 (weight_decay_lambda):\n")
            for l2, cnt in top_df["weight_decay_lambda"].value_counts().items():
                f.write(f"   L2 {l2} -> {cnt} times\n")

            # 모델 상세 정보 전체 출력
            f.write("\nModel details:\n")
            for i, row in top_df.iterrows():
                f.write(f"  {i+1}) "
                        f"optimizer={row['optimizer']} | "
                        f"lr={row['lr']} | "
                        f"batch_size={row['batch_size']} | "
                        f"max_iterations={row['max_iterations']} | "
                        f"max_epochs={row['max_epochs']} | "
                        f"hidden_size_list={row['hidden_size_list']} | "
                        f"activation={row['activation']} | "
                        f"weight_init_std={row['weight_init_std']} | "
                        f"weight_decay_lambda={row['weight_decay_lambda']} | "
                        f"use_batchnorm={row['use_batchnorm']} | "
                        f"dropout_ration={row['dropout_ration']} | "
                        f"final_loss={row['final_loss']} | "
                        f"final_acc={row['final_acc']}\n"
                        )


def main():
    input_csv = "hyperparam_results.csv"
    log_dir = "../log"
    os.makedirs(log_dir, exist_ok=True)

    df = pd.read_csv(input_csv)

    # 숫자 변환
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ACC 기준 정렬
    df_acc = df.sort_values(by="final_acc", ascending=False)
    df_acc.to_csv(os.path.join(log_dir, "sort_acc.csv"), index=False)

    # LOSS 기준 정렬
    df_loss = df.sort_values(by="final_loss", ascending=True)
    df_loss.to_csv(os.path.join(log_dir, "sort_loss.csv"), index=False)

    # 퍼센타일 분석
    analyze_top_percentiles(df_acc, os.path.join(log_dir, "sort_percentiles.txt"))

    print("Done: sort_acc.csv, sort_loss.csv, sort_percentiles.txt created.")


if __name__ == "__main__":
    main()
