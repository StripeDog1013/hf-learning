# 11 Multimodal (LLaVA, Qwen-VL)

## 概要

本セクションでは **Vision Language Model (VLM)** を学習する。

これまで扱ってきたLLMは文章のみを入力としていたが、VLMでは画像と文章を同時に入力し、画像内容を理解した上で回答を生成できる。

代表的な用途は以下の通り。

* 画像説明
* 画像に関する質問応答 (Visual Question Answering)
* OCR
* 画像比較
* マルチモーダルチャット

本章では Hugging Face の **LLaVA** と **Qwen2.5-VL** を利用し、画像理解AIの基礎を学習する。

---

# 学習目標

本セクションでは以下を理解する。

* Vision Language Model (VLM)
* Processor の役割
* 画像とテキストの同時入力
* LLaVA
* Qwen2.5-VL
* Chat Template
* マルチ画像入力
* Webcam入力

---

# 前章との違い

## 10 LLM Agent + Tool Calling

```text
文章
↓

LLM

↓

回答
```

---

## 11 Multimodal

```text
画像
        +
文章
↓

Processor

↓

VLM

↓

回答
```

画像が入力として追加される点が最大の違いである。

---

# Vision Language Modelとは

Vision Language Model (VLM) は

* Vision Encoder
* Large Language Model

を組み合わせたモデルである。

画像を特徴量へ変換し、その特徴量をLLMへ入力することで画像理解を実現する。

---

# Processorとは

これまでのLLMでは

```python
tokenizer(...)
```

のみを利用していた。

VLMでは

```python
processor(...)
```

を利用する。

Processorは

* Tokenizer
* Image Processor

をまとめたクラスである。

役割

```text
画像
↓

画像テンソル

+

テキスト

↓

LLM入力
```

---

# フォルダ構成

```text
11_multimodal_vlm/
├── README.md
├── data/
│   ├── images/
│   └── samples/
├── models/
├── outputs/
├── checkpoints/
└── src/
    ├── config.py
    ├── device.py
    ├── utils.py
    ├── image_utils.py
    ├── llava_inference.py
    ├── qwen_vl_inference.py
    ├── multi_image_chat.py
    └── webcam_vlm.py
```

---

# 各ファイル

## config.py

各種設定。

* モデル名
* GPU設定
* 推論設定
* ディレクトリ設定
* 画像サイズ

---

## device.py

09章・10章と同じ。

* CUDA
* Apple MPS
* CPU

をサポート。

---

## utils.py

共通処理。

* JSON保存
* チャットログ保存
* テキスト保存
* ヘッダ表示

---

## image_utils.py

画像専用ユーティリティ。

* 画像読込
* RGB変換
* リサイズ
* 保存
* 情報表示

---

## llava_inference.py

LLaVAを利用した画像説明。

```text
画像

↓

LLaVA

↓

回答
```

入力例

```text
この画像を説明してください
```

---

## qwen_vl_inference.py

Qwen2.5-VLによる画像理解。

LLaVAとの違い

* Chat Template
* AutoProcessor

を利用する。

---

## multi_image_chat.py

複数画像入力。

例

```text
画像A

画像B

↓

違いを説明してください
```

---

## webcam_vlm.py

Webカメラ画像を利用。

```text
Webcam

↓

Qwen2.5-VL

↓

画像説明
```

---

# 実行方法

## LLaVA

```bash
python llava_inference.py
```

---

## Qwen2.5-VL

```bash
python qwen_vl_inference.py
```

---

## Multi Image Chat

```bash
python multi_image_chat.py
```

---

## Webcam

```bash
python webcam_vlm.py
```

---

# Processorの流れ

```text
PIL Image
        +
Question

↓

Processor

↓

Pixel Values
Input IDs
Attention Mask

↓

VLM

↓

Generate

↓

Answer
```

---

# LLaVAとQwen2.5-VLの違い

| 項目            | LLaVA | Qwen2.5-VL |
| ------------- | ----- | ---------- |
| Processor     | 〇     | 〇          |
| Chat Template | ×     | 〇          |
| 日本語性能         | 普通    | 高い         |
| 画像比較          | 可能    | 得意         |
| 最新性           | △     | ◎          |

---

# 動作確認環境

## Windows

* RTX4070Ti (12GB)
* RTX4000 Ada (20GB)

推奨GPU。

---

## macOS

* Apple M4 Pro
* RAM 24GB

LLaVA 7B は動作するものの推論時間が長くなる場合がある。

Windows GPUでの実行を推奨。

---

# 本章で重要なポイント

## TokenizerからProcessorへ

これまで

```python
tokenizer(...)
```

だった処理が

```python
processor(...)
```

へ変わる。

Processorは

* 画像
* テキスト

を同時に扱える。

---

## Chat Template

Qwen2.5-VLでは

```python
processor.apply_chat_template(...)
```

を利用してチャット形式を構築する。

これは今後のMultimodalモデルでも広く利用される。

---

## 画像サイズ

高解像度画像は推論時間・VRAM使用量が大きく増加する。

本プロジェクトでは

```python
resize_keep_aspect(...)
```

を利用して適切なサイズへ縮小してから推論を行う。

---

# 次章

## 12 ControlNet / IP-Adapter

次章では画像生成AIをさらに発展させ、

* ControlNet
* IP-Adapter

を利用して

* ポーズ指定
* 線画指定
* 画像参照生成

など、画像生成を細かく制御する方法を学習する。
