from pathlib import Path


# ============================================================
# Project Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

DOCUMENT_DIR = DATA_DIR / "documents"
INDEX_DIR = OUTPUT_DIR / "faiss_index"

FAISS_INDEX_PATH = INDEX_DIR / "index.faiss"
METADATA_PATH = INDEX_DIR / "metadata.json"


# ============================================================
# GPU Settings
# ============================================================

# 物理GPU ID
# RTX4070Ti     : 0
# RTX4000 Ada   : 1
PHYSICAL_CUDA_ID = 0

# CUDA_VISIBLE_DEVICES を使って、
# PyTorchから見えるGPUを1枚に制限する
USE_CUDA_VISIBLE_DEVICES = True

# PyTorchから見える論理GPU ID
# CUDA_VISIBLE_DEVICES使用時は基本的に cuda:0
CUDA_ID = 0


# ============================================================
# Embedding Model Settings
# ============================================================

# 日本語も扱いやすい多言語Embeddingモデル
EMBEDDING_MODEL_NAME = (
    "intfloat/multilingual-e5-small"
)

# E5系モデルではquery/passsage prefixを付ける
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

EMBEDDING_BATCH_SIZE = 16
EMBEDDING_MAX_LENGTH = 512

NORMALIZE_EMBEDDINGS = True


# ============================================================
# FAISS Settings
# ============================================================

# cosine類似度相当で検索するため、
# 正規化済みベクトル + Inner Product を使う
FAISS_INDEX_TYPE = "IndexFlatIP"

TOP_K = 5


# ============================================================
# Text Split Settings
# ============================================================

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

SUPPORTED_EXTENSIONS = [
    ".txt",
    ".md",
]


# ============================================================
# RAG Generation Settings
# ============================================================

GENERATION_MODEL_NAME = (
    "Qwen/Qwen2.5-1.5B-Instruct"
)

MAX_CONTEXT_CHARS = 3000
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True


# ============================================================
# Random Seed
# ============================================================

SEED = 42


# ============================================================
# Utility
# ============================================================

def create_dirs():
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DOCUMENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )