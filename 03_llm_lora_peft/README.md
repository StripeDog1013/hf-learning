# 03_llm_lora_peft

## 概要

本章では PEFT (Parameter-Efficient Fine-Tuning) を利用した LoRA 学習を行う。

前章：

```txt
02_llm_finetune_trainer
```

ではモデル全体を学習した。

本章では、

```txt
事前学習済みLLM
↓
LoRA Adapter
↓
軽量学習
```

を学習する。

---

## LoRAとは

LoRA
(Low-Rank Adaptation)

大規模言語モデルの重み全体を学習せず、

```txt
追加の小さな行列
```

だけを学習する手法。

---

## 従来のFine-Tuning

```txt
Base Model
↓
全パラメータ更新
```

例：

```txt
1B Parameters
↓
全部更新
```

---

## LoRA

```txt
Base Model
↓
固定
↓
LoRA Adapterのみ更新
```

例：

```txt
1B Parameters
↓
数百万パラメータのみ更新
```

---

## メリット

### VRAM削減

通常Fine-Tuning

```txt
大
```

LoRA

```txt
小
```

---

### 学習高速化

```txt
学習対象が少ない
↓
高速
```

---

### 配布が容易

通常：

```txt
数GB
```

LoRA：

```txt
数MB〜数十MB
```

---

## 使用ライブラリ

```python
transformers
peft
torch
datasets
```

---

## ディレクトリ構成

```txt
03_llm_lora_peft/
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

---

## 使用データセット

例：

```python
DATASET_NAME = "Abirate/english_quotes"
```

---

## 学習の流れ

```txt
Dataset
↓
Tokenizer
↓
Base Model
↓
LoRA Adapter追加
↓
Trainer
↓
学習
↓
保存
```

---

## PEFT

PEFT：

```txt
Parameter-Efficient Fine-Tuning
```

読込：

```python
from peft import (
    LoraConfig,
    get_peft_model,
)
```

---

## LoRA設定

```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
)
```

---

### r

LoRA行列サイズ

例：

```python
r=8
```

一般的：

```txt
4
8
16
32
```

---

### lora_alpha

LoRAのスケール

例：

```python
lora_alpha=16
```

---

### lora_dropout

過学習抑制

例：

```python
lora_dropout=0.1
```

---

## モデル作成

```python
model = get_peft_model(
    base_model,
    lora_config,
)
```

---

## 学習対象確認

```python
model.print_trainable_parameters()
```

出力例：

```txt
trainable params: 2,097,152
all params: 1,300,000,000
trainable%: 0.16%
```

---

## Trainer学習

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
```

---

## 学習実行

```powershell
python .\src\train.py
```

---

## Checkpoint

保存内容：

```txt
adapter_model.safetensors
adapter_config.json
```

---

## Resume Training

学習再開

```powershell
python .\src\resume_train.py
```

---

## Continue Training

既存LoRAを読み込み、
追加学習を行う。

```powershell
python .\src\continue_train.py
```

---

## LoRA保存

```python
model.save_pretrained(
    OUTPUT_DIR
)
```

保存例：

```txt
checkpoints/
└─ final_model/
    ├─ adapter_config.json
    ├─ adapter_model.safetensors
    └─ README.md
```

---

## LoRA読込

```python
from peft import PeftModel

model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
)
```

---

## 推論

```powershell
python .\src\inference.py
```

内部：

```txt
Base Model
+
LoRA Adapter
```

を結合して推論する。

---

## 評価

```powershell
python .\src\evaluate.py
```

出力例：

```txt
eval_loss: 1.23
```

---

## LoRAと通常学習の比較

### 通常Fine-Tuning

```txt
学習速度     遅い
VRAM使用量   多い
保存サイズ   大きい
```

---

### LoRA

```txt
学習速度     速い
VRAM使用量   少ない
保存サイズ   小さい
```

---

## 学習済みLoRAの利用

```txt
Base Model
+
LoRA Adapter
↓
推論
```

---

## 実運用

現在の多くのOSS LLMは、

```txt
LoRA
QLoRA
```

で学習されている。

例：

```txt
日本語特化モデル
企業内チャットボット
RAG向けモデル
```

---

## 学んだこと

* PEFT
* LoRA
* Adapter
* Fine-Tuning
* Parameter-Efficient Learning
* Checkpoint
* Resume Training
* Continue Training
* LoRA Inference

---

## 前章との違い

02_llm_finetune_trainer

```txt
全パラメータ学習
```

03_llm_lora_peft

```txt
LoRAのみ学習
```

---

## LoRAの特徴

```txt
軽量
高速
低VRAM
高効率
```

---

## 次章

```txt
04_vit_image_classification
```

Vision Transformer を利用した画像分類を学習する。
