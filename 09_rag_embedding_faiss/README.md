# 09 RAG + Embedding + FAISS

## 概要

本セクションでは、RAG（Retrieval-Augmented Generation）を用いた検索拡張生成システムを実装する。

RAGは、LLM単体では持たない外部知識を検索し、その検索結果をプロンプトに含めることで、より正確な回答を生成する技術である。

本セクションでは以下を学習する。

* Embedding（文章のベクトル化）
* FAISSによる高速類似検索
* 文書チャンク分割
* RAGパイプライン構築
* LLMへのコンテキスト注入

---

## 前セクションとの違い

### 08 Diffusers LoRA Train

* 画像生成モデルを追加学習した
* モデルの重みを更新した

### 09 RAG

* モデルは学習しない
* 外部知識を検索して回答する
* 知識更新時に再学習が不要

RAGは「ファインチューニングの代替手段」として広く利用されている。

---

## RAGの全体構成

```text
文書
 ↓
チャンク分割
 ↓
Embedding
 ↓
FAISS Index作成
 ↓
質問
 ↓
Embedding
 ↓
類似検索
 ↓
検索結果をLLMへ入力
 ↓
回答生成
```

---

## フォルダ構成

```text
09_rag_embedding_faiss/
├─ README.md
├─ data/
│   └─ documents/
├─ models/
├─ outputs/
│   └─ faiss_index/
├─ checkpoints/
└─ src/
   ├─ config.py
   ├─ device.py
   ├─ utils.py
   ├─ build_index.py
   ├─ search_index.py
   └─ rag_chat.py
```

---

## 使用環境

### 共通環境

* Python 3.11 / 3.12
* Anaconda
* 仮想環境: ml-env

### Windows

* RTX4070Ti
* RTX4000 Ada
* CUDA 12.x

### Mac

* Apple M4 Pro
* MPS対応

---

## インストール

### PyTorch

CUDA環境：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 必要ライブラリ

```bash
pip install transformers accelerate sentencepiece
pip install numpy tqdm
pip install scikit-learn scipy
```

### FAISS

Windows / macOS：

```bash
conda install -c conda-forge faiss-cpu
```

---

## GPU設定

複数GPU環境では、意図しないGPU利用を防ぐため、`CUDA_VISIBLE_DEVICES` を利用する。

```python
PHYSICAL_CUDA_ID = 0
USE_CUDA_VISIBLE_DEVICES = True
CUDA_ID = 0
```

使用例：

```python
import os

from config import (
    USE_CUDA_VISIBLE_DEVICES,
    PHYSICAL_CUDA_ID,
)

if USE_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        PHYSICAL_CUDA_ID
    )
```

---

## Embeddingモデル

使用モデル：

```text
intfloat/multilingual-e5-small
```

特徴：

* 日本語対応
* 多言語対応
* 軽量
* 高速

E5系モデルではprefixを付与する。

```text
query: 質問文
passage: 文書
```

---

## LLMモデル

使用モデル：

```text
Qwen/Qwen2.5-1.5B-Instruct
```

用途：

* RAG回答生成
* 日本語対応
* GPUメモリ消費が比較的小さい

---

## 実行手順

### 1. 文書配置

```text
data/documents/
└─ sample.md
```

例：

```md
RAGはRetrieval-Augmented Generationの略です。
外部文書を検索し、その内容をLLMに与えて回答を生成します。
```

---

### 2. FAISS Index作成

```bash
cd src
python build_index.py
```

生成：

```text
outputs/faiss_index/
├─ index.faiss
└─ metadata.json
```

---

### 3. 類似検索

```bash
python search_index.py
```

質問例：

```text
RAGとは何ですか？
```

---

### 4. RAGチャット

```bash
python rag_chat.py
```

終了：

```text
q
```

---

## 各ファイルの役割

### config.py

* パラメータ管理
* パス管理
* GPU設定

### device.py

* CUDA/MPS/CPU管理
* GPU情報表示
* Seed固定

### utils.py

* 文書読込
* チャンク分割
* JSON保存

### build_index.py

文書をEmbedding化し、FAISS Indexを作成する。

```text
文書 → Embedding → Index
```

### search_index.py

質問文から関連文書を検索する。

```text
質問 → 検索
```

### rag_chat.py

検索結果をLLMに渡して回答を生成する。

```text
質問 → 検索 → 生成
```

---

## 学習ポイント

本セクションで重要なのは、

「LLMは知識を覚えさせるだけではない」

という考え方である。

RAGでは、

* モデルを再学習しなくても知識更新できる
* 社内文書や独自データを利用できる
* 幻覚（Hallucination）を低減できる

という利点がある。

現在の生成AIシステムでは、RAGは極めて重要な技術となっている。

---

## 次セクション

### 10 LLM Agent + Tool Calling

次のセクションでは、

* ツール呼び出し
* 関数呼び出し
* Agent
* 自律的なLLM

を学習し、LLMを「行動できるAI」へ発展させる。
