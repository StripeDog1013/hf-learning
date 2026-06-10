# 04_vit_image_classification

## 概要

本章では Vision Transformer (ViT) を利用した画像分類を学習する。

前章までは自然言語処理（NLP）を扱っていたが、本章から Vision タスクへ進む。

対象タスク：

```txt
画像
↓
クラス分類
```

例：

```txt
犬画像
↓
dog

猫画像
↓
cat
```

---

## Vision Transformer (ViT)とは

Vision Transformer は Google が提案した画像向け Transformer モデルである。

従来：

```txt
CNN
↓
ResNet
↓
EfficientNet
```

ViT：

```txt
画像
↓
Patch分割
↓
Transformer
↓
分類
```

---

## 使用モデル

例：

```python
MODEL_NAME = "google/vit-base-patch16-224"
```

---

## 使用データセット

例：

```python
DATASET_NAME = "beans"
```

クラス：

```txt
0 : angular_leaf_spot
1 : bean_rust
2 : healthy
```

---

## ディレクトリ構成

```txt
04_vit_image_classification/
├─ checkpoints/
├─ data/
├─ logs/
├─ README.md
└─ src/
    ├─ config.py
    ├─ device.py
    ├─ dataset.py
    ├─ save_dataset_images.py
    ├─ image_processor.py
    ├─ load_model.py
    ├─ utils.py
    ├─ train.py
    ├─ resume_train.py
    ├─ continue_train.py
    ├─ evaluate.py
    ├─ inference.py
    └─ visualize_images.py
```

---

## 画像分類とは

入力：

```txt
画像
```

出力：

```txt
クラス
```

例：

```txt
Bean Rust画像
↓
bean_rust
```

---

## ViTの仕組み

### 1. Patch分割

画像：

```txt
224 × 224
```

↓

```txt
16 × 16
```

単位で分割

---

### 2. Flatten

各Patchをベクトル化

```txt
Patch
↓
Vector
```

---

### 3. Positional Encoding

位置情報を付与

```txt
Patch
+
Position
```

---

### 4. Transformer Encoder

Self-Attention により特徴抽出

```txt
Patch Sequence
↓
Transformer
```

---

### 5. Classification Head

最終分類

```txt
Hidden State
↓
Linear Layer
↓
Class Probability
```

---

## データセット確認

```powershell
python .\src\dataset.py
```

出力例：

```txt
Rows : 1034

Columns :
[
    image,
    labels
]
```

---

## データセット保存

```powershell
python .\src\save_dataset_images.py
```

保存先：

```txt
data/
└─ beans/
   ├─ train/
   ├─ validation/
   └─ test/
```

---

## Image Processor

読込：

```python
from transformers import AutoImageProcessor
```

例：

```python
image_processor =
AutoImageProcessor.from_pretrained(
    MODEL_NAME
)
```

役割：

```txt
画像
↓
Resize
↓
Normalize
↓
Tensor化
```

---

## モデル読込

```python
from transformers import ViTForImageClassification
```

例：

```python
model =
ViTForImageClassification.from_pretrained(
    MODEL_NAME
)
```

---

## 学習の流れ

```txt
Image
↓
Image Processor
↓
Pixel Values
↓
ViT
↓
Logits
↓
Loss
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

出力例：

```txt
loss
eval_loss
learning_rate
epoch
```

---

## Resume Training

学習再開

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

既存モデルから再学習

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

## 評価

```powershell
python .\src\evaluate.py
```

出力例：

```txt
eval_loss: 0.31
eval_accuracy: 0.94
```

---

## Accuracy

分類問題で最も重要な指標

計算：

```txt
正解数
───────
全データ数
```

例：

```txt
95 / 100
↓
95%
```

---

## 推論

```powershell
python .\src\inference.py
```

入力：

```txt
bean.jpg
```

出力：

```txt
healthy
score=0.98
```

---

## Softmax

出力例：

```txt
healthy            0.98
bean_rust          0.01
angular_leaf_spot  0.01
```

最も高い確率が予測結果となる。

---

## 画像可視化

```powershell
python .\src\visualize_images.py
```

確認内容：

```txt
画像
ラベル
クラス名
```

---

## 学習済みモデル保存

```python
trainer.save_model(
    OUTPUT_DIR
)
```

Image Processor も保存する。

```python
image_processor.save_pretrained(
    OUTPUT_DIR
)
```

---

## ViTの特徴

### メリット

```txt
Transformerベース
高精度
実装がシンプル
```

---

### デメリット

```txt
学習コスト高
データ量が必要
```

---

## CNNとの比較

CNN

```txt
局所特徴抽出
```

ViT

```txt
Global Attention
```

---

## 学んだこと

* Vision Transformer
* Patch
* Positional Encoding
* Image Classification
* AutoImageProcessor
* ViTForImageClassification
* Trainer
* Evaluation
* Accuracy
* Inference

---

## これまでとの違い

01～03

```txt
自然言語処理
```

04

```txt
画像処理
```

---

## 次章

```txt
05_segformer_segmentation
```

画像分類から発展し、

```txt
画像全体
↓
1クラス
```

ではなく、

```txt
画像内の全ピクセル
↓
クラス分類
```

を行う Semantic Segmentation を学習する。
