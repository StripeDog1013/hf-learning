# 06_detr_object_detection

## 概要

本章では Hugging Face Transformers の DETR (DEtection TRansformer) を用いて物体検出モデルを学習する。

使用モデル：

```python
facebook/detr-resnet-50
```

使用データセット：

```python
cppe-5
```

CPPE-5 は医療用保護具を検出する物体検出データセットであり、以下の5クラスを含む。

| ID | Class       |
| -- | ----------- |
| 0  | coverall    |
| 1  | face_shield |
| 2  | gloves      |
| 3  | goggles     |
| 4  | mask        |

---

## DETRとは

DETR (DEtection TRansformer) は Facebook Research が開発した Transformer ベースの物体検出モデルである。

従来の物体検出モデル：

```txt
Faster R-CNN
YOLO
SSD
```

では Anchor Box や NMS が必要だったが、DETR は Transformer と Hungarian Matching を用いて End-to-End な物体検出を実現している。

入力：

```txt
画像
```

出力：

```txt
Bounding Box
Class Label
Confidence Score
```

例：

```txt
mask         0.92
bbox=(120, 80, 240, 260)

face_shield  0.88
bbox=(320, 60, 420, 250)
```

---

## ディレクトリ構成

```txt
06_detr_object_detection/
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
    ├─ evaluate_map.py
    ├─ inference.py
    ├─ visualize_ground_truth.py
    └─ visualize_boxes.py
```

---

## データセット確認

```powershell
python .\src\dataset.py
```

出力例：

```txt
Rows : 1000
Columns :
[
    image_id,
    image,
    width,
    height,
    objects
]
```

---

## データセット保存

Hugging Face Dataset をローカルへ保存する。

```powershell
python .\src\save_dataset_images.py
```

保存先：

```txt
data/
└─ cppe-5/
   ├─ train/
   │  ├─ images/
   │  └─ annotations/
   │
   └─ validation/
      ├─ images/
      └─ annotations/
```

---

## 学習

```powershell
python .\src\train.py
```

モデル保存先：

```txt
checkpoints/final_model
```

---

## 学習再開

途中で中断された学習を再開する。

```powershell
python .\src\resume_train.py
```

使用対象：

```txt
checkpoints/checkpoint-xxxx
```

---

## 継続学習

既存モデルを読み込み、新しい学習を開始する。

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

## Ground Truth 可視化

アノテーションが正しいか確認する。

```powershell
python .\src\visualize_ground_truth.py
```

出力：

```txt
logs/ground_truth/
```

---

## 推論

1枚画像に対して推論を実行する。

```powershell
python .\src\inference.py
```

出力：

```txt
logs/inference_result.jpg
```

---

## Bounding Box 可視化

検出結果を画像へ描画する。

```powershell
python .\src\visualize_boxes.py
```

出力：

```txt
logs/predicted_boxes/
```

---

## 評価

### Eval Loss

```powershell
python .\src\evaluate.py
```

出力例：

```txt
eval_loss: 2.13
```

---

### mAP

```powershell
python .\src\evaluate_map.py
```

出力例：

```txt
map: 0.094
map_50: 0.175
map_75: 0.092
```

---

## mAP指標

### mAP

IoU 0.50～0.95 の平均精度。

```txt
0.0 ～ 1.0
```

目安：

```txt
0.05以下  学習失敗
0.10前後 学習成功
0.20以上 良好
0.50以上 実用レベル
```

---

### mAP@50

IoU 0.50 以上を正解とした場合の精度。

```txt
map_50
```

---

### Recall

見逃し率を含めた検出性能。

```txt
mar_100
```

---

## 学習結果

今回の学習結果：

```txt
mAP     : 0.094
mAP@50  : 0.175
```

評価：

```txt
学習成功
Bounding Box 学習成功
クラス分類は改善余地あり
```

---

## 今回学んだこと

* Hugging Face Object Detection
* DETR
* Bounding Box
* COCO Annotation Format
* Hungarian Matching
* mAP
* Recall
* Object Detection Inference
* Bounding Box Visualization

---

## 次章

```txt
07_diffusers_inference
```

Stable Diffusion を用いた画像生成を学習する。
