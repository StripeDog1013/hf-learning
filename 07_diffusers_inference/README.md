# 07_diffusers_inference

## 概要

本章では Hugging Face Diffusers を利用した画像生成を学習する。

対象タスク：

```txt
テキスト
↓
画像生成
```

利用モデル：

```txt
Stable Diffusion
```

本章では主に以下を学習する。

* Text-to-Image
* Image-to-Image
* Scheduler
* Seed固定
* ベンチマーク
* GPU推論

---

## Stable Diffusionとは

Stable Diffusion は潜在空間上で画像生成を行う拡散モデルである。

生成の流れ：

```txt
ノイズ画像
↓
UNet
↓
ノイズ除去
↓
画像生成
```

---

## 使用ライブラリ

```python
diffusers
transformers
torch
accelerate
```

---

## ディレクトリ構成

```txt
07_diffusers_inference/
├─ outputs/
├─ README.md
└─ src/
    ├─ config.py
    ├─ device.py
    ├─ load_pipeline.py
    ├─ utils.py
    ├─ text_to_image.py
    ├─ image_to_image.py
    ├─ scheduler_compare.py
    └─ benchmark.py
```

---

## 使用モデル

### SD-Turbo

```python
MODEL_NAME = "stabilityai/sd-turbo"
```

特徴：

```txt
高速
軽量
少ないstepで生成可能
```

推奨設定：

```python
NUM_INFERENCE_STEPS = 4
GUIDANCE_SCALE = 0.0
```

---

### Stable Diffusion v1.5

```python
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
```

特徴：

```txt
高品質
低速
一般的なSDモデル
```

推奨設定：

```python
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5
```

---

## Device選択

優先順位：

```txt
CUDA
↓
MPS
↓
CPU
```

確認：

```python
device = get_device()
```

---

## Pipeline読込

```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_NAME,
)
```

---

## Text-to-Image

実行：

```powershell
python .\src\text_to_image.py
```

入力：

```python
PROMPT = (
    "a cute shiba inu wearing a space suit"
)
```

出力：

```txt
outputs/generated.png
```

---

## Prompt

生成したい内容を指定する。

例：

```txt
a futuristic city at night
```

---

## Negative Prompt

生成したくない内容を指定する。

例：

```txt
low quality, blurry, distorted
```

---

## Seed固定

同じ画像を再現する。

```python
generator = torch.Generator(
    device=device
).manual_seed(42)
```

同じ

```txt
Prompt
Seed
Model
Scheduler
```

なら基本的に同じ画像が生成される。

---

## Scheduler

Scheduler はノイズ除去アルゴリズムである。

生成品質や速度に影響する。

---

## Scheduler比較

実行：

```powershell
python .\src\scheduler_compare.py
```

出力：

```txt
outputs/
├─ scheduler_euler.png
├─ scheduler_euler_a.png
└─ scheduler_dpm_solver.png
```

利用Scheduler：

```txt
Euler
Euler Ancestral
DPM Solver
```

---

## Schedulerの特徴

### Euler

```txt
高速
安定
```

---

### Euler A

```txt
多様性が高い
創造的
```

---

### DPM Solver

```txt
高品質
高速
```

---

## Benchmark

画像生成速度を計測する。

実行：

```powershell
python .\src\benchmark.py
```

出力例：

```txt
Average : 1.25 sec/image
Min     : 1.18 sec/image
Max     : 1.33 sec/image
```

---

## 推論パラメータ

### num_inference_steps

ノイズ除去回数。

```txt
少ない
↓
高速
低品質

多い
↓
低速
高品質
```

例：

```python
NUM_INFERENCE_STEPS = 30
```

---

### guidance_scale

Promptへの忠実度。

```txt
低い
↓
自由

高い
↓
Prompt重視
```

例：

```python
GUIDANCE_SCALE = 7.5
```

---

## 画像サイズ

```python
WIDTH = 512
HEIGHT = 512
```

一般的な設定：

```txt
512×512
768×768
1024×1024
```

大きいほどVRAMを消費する。

---

## VRAM使用量の目安

### SD-Turbo

```txt
512×512
約4～6GB
```

---

### Stable Diffusion v1.5

```txt
512×512
約6～8GB
```

---

## 出力画像保存

```python
image.save(output_path)
```

保存先：

```txt
outputs/
```

---

## 学んだこと

* Diffusers
* Stable Diffusion
* Text-to-Image
* Scheduler
* Seed
* GPU推論
* Benchmark
* Prompt Engineering

---

## 前章との違い

06_detr_object_detection

```txt
画像
↓
物体検出
```

07_diffusers_inference

```txt
テキスト
↓
画像生成
```

---

## 次章

```txt
08_diffusers_lora_train
```

LoRA を用いた画像生成モデルの追加学習を行う。
