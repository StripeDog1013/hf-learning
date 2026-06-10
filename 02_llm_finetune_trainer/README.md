# 02_llm_finetune_trainer

## 概要

本章では Hugging Face Trainer を利用して LLM のファインチューニングを行う。

前章：

```txt
01_llm_inference
```

では推論のみを行った。

本章では、

```txt
事前学習済みLLM
↓
独自データ
↓
ファインチューニング
```

を学習する。

---

## 目的

以下を理解する。

* Dataset
* Tokenizer
* Trainer
* TrainingArguments
* Fine-Tuning
* Checkpoint
* Resume Training
* Evaluation

---

## ディレクトリ構成

```txt
02_llm_finetune_trainer/
├─ checkpoints/
├─ data/
├─ logs/
├─ README.md
└─ src/
    ├─ config.py
    ├─ device.py
    ├─ dataset.py
    ├─ tokenizer.py
    ├─ load_model.py
    ├─ utils.py
    ├─ train.py
    ├─ resume_train.py
    ├─ continue_train.py
    ├─ evaluate.py
    └─ inference.py
```

---

## 使用モデル

例：

```python
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
```

学習時は比較的小型のモデルを利用する。

---

## 使用データセット

例：

```python
DATASET_NAME = "Abirate/english_quotes"
```

構造：

```txt
quote
author
tags
```

サンプル：

```txt
Quote:
Be yourself; everyone else is already taken.

Author:
Oscar Wilde
```

---

## 学習の流れ

```txt
Dataset
↓
Tokenizer
↓
Token IDs
↓
Trainer
↓
Fine-Tuning
↓
Checkpoint
↓
Evaluation
```

---

## Dataset読込

```python
from datasets import load_dataset

dataset = load_dataset(
    DATASET_NAME
)
```

---

## Tokenization

```python
tokenizer(
    text,
    truncation=True,
    padding="max_length",
    max_length=128,
)
```

出力：

```txt
input_ids
attention_mask
```

---

## Data Collator

役割：

```txt
複数サンプル
↓
バッチ化
```

例：

```python
DataCollatorForLanguageModeling
```

---

## Trainer

Trainerは学習ループを自動化する。

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
```

---

## TrainingArguments

代表的な設定：

```python
TrainingArguments(
    output_dir="checkpoints",
    num_train_epochs=5,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
)
```

---

### num_train_epochs

学習回数

例：

```python
num_train_epochs=5
```

---

### learning_rate

学習率

例：

```python
learning_rate=5e-5
```

---

### per_device_train_batch_size

バッチサイズ

例：

```python
per_device_train_batch_size=4
```

---

### save_steps

チェックポイント保存間隔

例：

```python
save_steps=100
```

---

## 学習実行

```powershell
python .\src\train.py
```

出力：

```txt
loss
learning_rate
epoch
```

---

## Checkpoint

保存先：

```txt
checkpoints/
├─ checkpoint-100
├─ checkpoint-200
└─ checkpoint-300
```

内容：

```txt
model
optimizer
scheduler
trainer_state
```

---

## Resume Training

途中から再開する。

```powershell
python .\src\resume_train.py
```

内部：

```python
trainer.train(
    resume_from_checkpoint=True
)
```

---

## Continue Training

学習済みモデルを読み込み、
新たな学習を開始する。

```powershell
python .\src\continue_train.py
```

入力：

```txt
checkpoints/final_model
```

出力：

```txt
checkpoints/continued_model
```

---

## Evaluation

評価実行：

```powershell
python .\src\evaluate.py
```

出力例：

```txt
eval_loss: 1.82
```

---

## Inference

学習済みモデルによる推論。

```powershell
python .\src\inference.py
```

入力：

```txt
Machine learning is
```

出力：

```txt
Machine learning is a field of artificial
intelligence that enables systems to learn
from data.
```

---

## 損失関数

LLMでは通常、

```txt
Cross Entropy Loss
```

が使用される。

目的：

```txt
正しい次トークン
↓
高確率
```

---

## 評価指標

### Loss

```txt
小さいほど良い
```

例：

```txt
5.0 → 2.0 → 1.0
```

---

### Perplexity

```txt
PPL = exp(loss)
```

例：

```txt
Loss=2.0
↓
PPL≈7.39
```

---

## 学習済みモデル保存

```python
trainer.save_model(
    "checkpoints/final_model"
)
```

Tokenizerも保存：

```python
tokenizer.save_pretrained(
    "checkpoints/final_model"
)
```

---

## 学んだこと

* Dataset
* Tokenizer
* Trainer
* TrainingArguments
* Checkpoint
* Resume Training
* Continue Training
* Evaluation
* Fine-Tuning
* Language Modeling

---

## 前章との違い

01_llm_inference

```txt
推論のみ
```

02_llm_finetune_trainer

```txt
学習
↓
評価
↓
推論
```

---

## 次章

```txt
03_peft_lora
```

LoRA を利用した軽量ファインチューニングを学習する。
