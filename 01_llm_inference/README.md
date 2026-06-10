# 01_llm_inference

## 概要

本章では Hugging Face Transformers を用いて大規模言語モデル（LLM）の推論を行う。

対象タスク：

* テキスト生成
* チャット形式推論
* トークナイズ
* モデルロード
* GPU推論

使用ライブラリ：

```python
transformers
torch
```

---

## 目的

LLM推論の基本的な流れを理解する。

```txt
テキスト
↓
Tokenizer
↓
Token ID
↓
Model
↓
Token ID
↓
Tokenizer Decode
↓
テキスト
```

---

## ディレクトリ構成

```txt
01_llm_inference/
├─ README.md
├─ logs/
└─ src/
    ├─ config.py
    ├─ device.py
    ├─ load_model.py
    ├─ inference.py
    └─ utils.py
```

---

## 使用モデル

例：

```python
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
```

その他利用可能なモデル：

```txt
Qwen/Qwen3-0.6B
Qwen/Qwen3-1.7B
microsoft/Phi-3-mini-4k-instruct
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

---

## LLM推論の流れ

### 1. Tokenizer読込

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)
```

---

### 2. Model読込

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)
```

---

### 3. 入力変換

```python
inputs = tokenizer(
    prompt,
    return_tensors="pt"
)
```

---

### 4. 推論

```python
outputs = model.generate(
    **inputs
)
```

---

### 5. デコード

```python
response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)
```

---

## 推論実行

```powershell
python .\src\inference.py
```

---

## generate()

代表的な引数：

```python
model.generate(
    max_new_tokens=256,
    temperature=0.7,
    do_sample=True,
    top_p=0.9
)
```

---

### max_new_tokens

生成する最大トークン数

```python
max_new_tokens=256
```

---

### temperature

ランダム性

```txt
低い
↓
決定的

高い
↓
創造的
```

例：

```python
temperature=0.7
```

---

### top_p

確率上位候補のみ使用

例：

```python
top_p=0.9
```

---

### do_sample

サンプリング有効化

```python
do_sample=True
```

---

## GPU利用

CUDA利用確認：

```python
torch.cuda.is_available()
```

GPUへ転送：

```python
model.to(device)
```

---

## 出力例

入力：

```txt
What is machine learning?
```

出力：

```txt
Machine learning is a branch of artificial intelligence
that enables computers to learn patterns from data
without being explicitly programmed.
```

---

## 主なクラス

### AutoTokenizer

役割：

```txt
文字列
↓
トークンID
```

---

### AutoModelForCausalLM

役割：

```txt
次トークン予測
```

---

### generate()

役割：

```txt
文章生成
```

---

## 学んだこと

* Hugging Face Hub
* AutoTokenizer
* AutoModelForCausalLM
* Tokenization
* GPU推論
* Text Generation
* Chat Model
* Generation Parameters

---

## 次章

```txt
02_peft_lora
```

LoRA を利用した LLM の軽量ファインチューニングを学習する。
