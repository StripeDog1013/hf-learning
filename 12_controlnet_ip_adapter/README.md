# 12 ControlNet / IP-Adapter

## 概要

本セクションでは **Stable Diffusion XL (SDXL)** を利用し、画像生成AIをより細かく制御する方法を学習する。

これまでの画像生成は **Promptのみ** を条件としていたが、本章では

* エッジ画像
* 奥行き画像
* 人物ポーズ
* 参照画像

などを条件として画像生成を行う。

これらを実現する代表的な技術が

* ControlNet
* IP-Adapter

である。

---

# 学習目標

本章では以下を理解する。

* SDXL
* ControlNet
* IP-Adapter
* Conditioning
* Canny Edge
* Depth Map
* OpenPose
* Reference Image
* Image Prompt

---

# 前章との違い

## 11 Multimodal

```text
画像
      +
質問

↓

Vision Language Model

↓

画像理解
```

---

## 12 ControlNet

```text
Prompt
      +
Control Image

↓

ControlNet

↓

SDXL

↓

画像生成
```

---

## 12 IP-Adapter

```text
Prompt
      +
Reference Image

↓

Image Encoder

↓

SDXL

↓

画像生成
```

11章は

**画像を理解するAI**

だったが、

本章は

**画像を参考に画像を生成するAI**

である。

---

# フォルダ構成

```text
12_controlnet_ip_adapter/
├── README.md
├── data/
│   ├── images/
│   ├── control/
│   └── reference/
├── models/
├── outputs/
├── checkpoints/
└── src/
    ├── config.py
    ├── device.py
    ├── utils.py
    ├── image_utils.py
    ├── sdxl_txt2img.py
    ├── controlnet_canny.py
    ├── controlnet_depth.py
    ├── controlnet_openpose.py
    ├── ip_adapter.py
    ├── comparison.py
    └── batch_generate.py
```

---

# 各ファイル

## config.py

各種設定。

* モデル
* Prompt
* GPU
* Seed
* 出力先
* Image Size

---

## device.py

09章以降共通。

* CUDA
* Apple MPS
* CPU

をサポート。

---

## utils.py

共通処理。

* Seed固定
* Prompt保存
* JSON保存
* Generation Log保存

---

## image_utils.py

画像処理。

* PIL読込
* Resize
* Canny生成
* 保存

---

## sdxl_txt2img.py

最も基本となるSDXL生成。

```text
Prompt

↓

SDXL

↓

画像
```

ControlNet無し。

---

## controlnet_canny.py

Canny Edgeを利用。

```text
画像

↓

Canny

↓

ControlNet

↓

画像生成
```

輪郭を維持した画像生成を学習する。

---

## controlnet_depth.py

Depth Mapを利用。

```text
画像

↓

Depth

↓

ControlNet

↓

画像生成
```

奥行きを維持した画像生成を学習する。

---

## controlnet_openpose.py

人物ポーズを利用。

```text
人物

↓

OpenPose

↓

ControlNet

↓

画像生成
```

人物ポーズを固定した画像生成を行う。

---

## ip_adapter.py

参照画像を利用。

```text
Reference Image

↓

IP-Adapter

↓

SDXL

↓

画像生成
```

雰囲気・色味・特徴を反映できる。

---

## comparison.py

各生成結果を横並び比較。

```text
SDXL

ControlNet

IP-Adapter

↓

比較画像
```

---

## batch_generate.py

複数Promptを一括生成。

今後の13章でも利用する。

---

# 実行順

推奨。

```text
① sdxl_txt2img.py

↓

② controlnet_canny.py

↓

③ controlnet_depth.py

↓

④ controlnet_openpose.py

↓

⑤ ip_adapter.py

↓

⑥ comparison.py

↓

⑦ batch_generate.py
```

---

# ControlNetとは

ControlNetとは

**画像を条件としてSDXLを制御する技術**

である。

```text
Prompt

+

Control Image

↓

ControlNet

↓

SDXL
```

Promptだけでは制御できない

* 輪郭
* 奥行き
* ポーズ

などを維持できる。

---

# IP-Adapterとは

IP-Adapterは

**参照画像の特徴を利用する技術**

である。

```text
Reference Image

↓

Image Encoder

↓

UNet
```

ControlNetほど構図を固定せず、

* 雰囲気
* 配色
* デザイン
* キャラクター性

を反映する用途に向いている。

---

# ControlNetとIP-Adapterの違い

| 項目    | ControlNet    | IP-Adapter      |
| ----- | ------------- | --------------- |
| 入力    | Control Image | Reference Image |
| 輪郭維持  | ◎             | △               |
| 奥行き維持 | ◎             | ×               |
| ポーズ維持 | ◎             | ×               |
| 色・雰囲気 | △             | ◎               |
| 構図維持  | ◎             | △               |

---

# 本章で重要な概念

## Conditioning

画像生成では

```text
Promptだけ
```

ではなく

```text
Prompt

+

画像条件
```

を与えることで、

より思い通りの画像生成が可能になる。

---

# 実行環境

## Windows

* RTX4070Ti
* RTX4000 Ada

推奨。

---

## macOS

* Apple M4 Pro

動作可能だが、

ControlNetはVRAM・メモリ使用量が大きく、

Windows GPU環境を推奨する。

---

# 開発中に遭遇した主な注意点

## xFormers / Triton

Windowsでは

```text
triton not found
```

という警告が表示される場合がある。

これは高速化機能が利用できないだけであり、

画像生成自体には影響しない。

---

## ControlNet OpenPose

OpenPoseモデルによっては

```text
diffusion_pytorch_model.fp16.safetensors
```

が存在しない場合がある。

その際は

* variant指定を外す
* fp16対応モデルへ変更する

などの対応が必要になる。

---

## Device不一致

Depth推定モデルを追加した際、

```text
CPU

↓

Depth

GPU

↓

SDXL
```

となり、

```text
Expected all tensors to be on the same device
```

が発生した。

主要コンポーネント

* text_encoder
* text_encoder_2
* UNet
* VAE
* ControlNet

を明示的に同じDeviceへ配置することで解決した。

---

# 次章

## 13 SDXL / FLUX LoRA

次章では

* SDXL LoRA学習
* FLUX LoRA学習
* DreamBoothとの違い
* 推論
* LoRA Merge

などを学習し、画像生成AIのファインチューニングへ進む。
