"""
LLM推論用の設定ファイル
"""

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# 生成設定
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50
REPETITION_PENALTY = 1.1

# モデル読み込み設定
TRUST_REMOTE_CODE = True

# CUDA環境ではfloat16を使用
USE_FP16 = True

# CUDA利用のGPU番号
CUDA_ID = 0