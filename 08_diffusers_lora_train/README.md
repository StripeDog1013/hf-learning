# 08_diffusers_lora_train

## 概要

本章では Hugging Face Diffusers を利用して Stable Diffusion の LoRA (Low-Rank Adaptation) 学習を行う。

対象タスク：

```txt
画像 + Prompt
↓
LoRA学習
↓
オリジナル画像生成
```

例：

* ペット（大福）
* キャラクター
* イラストスタイル
* 顔や服装

---

## LoRAとは

LoRA は既存の大規模モデル全体を再学習するのではなく、一部の重みだけを追加学習する手法である。

特徴：

* 軽量
* 高速学習
* 低VRAM
* モデル本体を変更しない

---

## DreamBooth LoRA の流れ

```txt
画像
↓
VAE（固定）
↓
潜在空間
↓
UNet（LoRA学習）
↓
ノイズ予測
↓
画像生成
```

---

## ディレクトリ構成

```txt
08_diffusers_lora_train/
├─ README.md
├─ data/
│  └─ daifuku/
│      ├─ 001.jpg
│      ├─ 002.jpg
│      └─ ...
├─ checkpoints/
│  └─ lora/
│      └─ pytorch_lora_weights.safetensors
├─ outputs/
├─ logs/
└─ src/
    ├─ config.py
    ├─ device.py
    ├─ dataset.py
    ├─ load_pipeline.py
    ├─ utils.py
    ├─ train_lora.py
    ├─ resume_train_lora.py
    ├─ inference_lora.py
    └─ benchmark.py
```

---

## 使用モデル

標準：

```python
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
```

軽量確認用：

```python
MODEL_NAME = "stabilityai/sd-turbo"
```

---

## 学習画像

推奨枚数：

```txt
最低 : 10枚
推奨 : 20～50枚
```

今回：

```txt
大福画像：33枚
```

十分な枚数である。

---

## トリガーワード

学習時：

```python
INSTANCE_PROMPT = "a photo of sks cat"
```

推論時：

```python
"a photo of sks cat wearing a wizard hat"
```

実在しない固有トークンの利用も有効：

```python
INSTANCE_PROMPT = "a photo of daifuku_cat"
```

---

## 学習対象

固定：

```txt
VAE
Text Encoder
```

学習：

```txt
UNet + LoRA
```

コード：

```python
freeze_parameters(pipe.vae)
freeze_parameters(pipe.text_encoder)
```

理由：

* VRAM削減
* 学習高速化
* 過学習抑制

---

## LoRA設定

例：

```python
LORA_RANK = 4
LORA_ALPHA = 4
LORA_DROPOUT = 0.0
```

### Rank

LoRA容量を決める。

```txt
小さい → 軽量
大きい → 表現力向上
```

推奨：

```txt
4 ～ 16
```

---

## 学習設定

例：

```python
NUM_EPOCHS = 30
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
GRADIENT_ACCUMULATION_STEPS = 4
```

---

## float16 と float32

LoRA学習では float16 が不安定になることがある。

現象：

```txt
loss = NaN
```

対策：

```python
torch.float32
```

推論時のみ float16 を使う構成も一般的。

---

## 学習

実行：

```powershell
python .\src\train_lora.py
```

出力：

```txt
checkpoints/lora/
└─ pytorch_lora_weights.safetensors
```

---

## 継続学習

実行：

```powershell
python .\src\resume_train_lora.py
```

既存 LoRA を読み込んで追加学習する。

---

## 推論

実行：

```powershell
python .\src\inference_lora.py
```

出力：

```txt
outputs/lora_inference.png
```

---

## LoRA強度

推論時：

```python
pipe.fuse_lora(
    lora_scale=1.2
)
```

目安：

```txt
1.0 : 標準
1.2 : おすすめ
1.5 : 強め
2.0 : 強すぎ
```

---

## Prompt例（大福）

```txt
a photo of sks cat wearing a wizard hat,
blue eyes,
cream white fur,
gray tabby markings on face,
striped tail,
realistic,
highly detailed,
natural lighting
```

---

## Negative Prompt

```txt
cartoon,
anime,
painting,
illustration,
blurry,
low quality,
deformed,
extra limbs,
duplicate,
bad anatomy
```

---

## ベンチマーク

実行：

```powershell
python .\src\benchmark.py
```

比較：

```txt
Base Model
vs
Base Model + LoRA
```

---

## LoRA保存形式

保存：

```python
StableDiffusionPipeline.save_lora_weights()
```

読込：

```python
pipe.load_lora_weights()
```

保存形式が一致しないと：

```txt
Invalid LoRA checkpoint
```

エラーが発生する。

---

## 学んだこと

* Stable Diffusion
* Diffusers
* DreamBooth
* LoRA
* PEFT
* VAE
* UNet
* Prompt Engineering
* Image Generation

---

## 前章との違い

07_diffusers_inference

```txt
Text
↓
画像生成
```

08_diffusers_lora_train

```txt
画像
+
Prompt
↓
LoRA学習
↓
画像生成
```

---

## 次の発展テーマ

* SDXL LoRA
* ControlNet
* IP-Adapter
* Flux LoRA
* ComfyUI
* Kohya SS
* LoRAマージ
* 複数LoRA同時利用

これで Hugging Face の主要タスク

* NLP
* Vision
* Detection
* Diffusion

を一通り学習完了。
