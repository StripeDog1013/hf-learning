# 05 SegFormer Semantic Segmentation

## Overview

This project demonstrates semantic segmentation using SegFormer and Hugging Face Transformers.

The ADE20K dataset (scene_parse_150) is used for training and evaluation.

The goal is to predict a semantic class for every pixel in an image.

---

## Model

Model:

```text
nvidia/segformer-b0-finetuned-ade-512-512
```

Task:

```text
Semantic Segmentation
```

Classes:

```text
150
```

Input Image Size:

```text
512 x 512
```

---

## Project Structure

```text
05_segformer_segmentation/
│
├─ README.md
│
├─ checkpoints/
├─ logs/
├─ data/
│
└─ src/
    ├─ config.py
    ├─ device.py
    ├─ dataset.py
    ├─ image_processor.py
    ├─ load_model.py
    ├─ utils.py
    │
    ├─ train.py
    ├─ resume_train.py
    ├─ continue_train.py
    │
    ├─ evaluate.py
    ├─ inference.py
    └─ visualize_mask.py
```

---

## Dataset

### Hugging Face Dataset

```text
merve/scene_parse_150
```

### Local Dataset Structure

```text
data/scene_parse_150/
├─ train/
│  ├─ images/
│  └─ masks/
│
└─ validation/
   ├─ images/
   └─ masks/
```

Enable local dataset:

```python
USE_LOCAL_IMAGE_FOLDER = True
```

Use Hugging Face dataset:

```python
USE_LOCAL_IMAGE_FOLDER = False
```

---

## Training

Run training:

```bash
python src/train.py
```

Resume training:

```bash
python src/resume_train.py
```

Continue training from final_model:

```bash
python src/continue_train.py
```

---

## Evaluation

```bash
python src/evaluate.py
```

Metrics:

```text
Pixel Accuracy
```

---

## Inference

```bash
python src/inference.py
```

Output:

```text
logs/pred_mask.png
```

---

## Visualization

Visualize prediction result:

```bash
python src/visualize_mask.py
```

Output:

```text
logs/segmentation_visualize.png
```

Layout:

```text
Original Image
Ground Truth Mask
Predicted Mask
```

---

## GPU

Single GPU execution:

```python
USE_CUDA_VISIBLE_DEVICES = True
PHYSICAL_CUDA_ID = 0
```

Example:

```text
0 = RTX 4070 Ti
1 = RTX 4000 Ada
```

---

## Notes

ADE20K labels must satisfy:

```text
0 ~ 149
255 = Ignore Label
```

Label value 150 should be converted to 255 before training.

---

## Learning Objectives

* Understand semantic segmentation
* Understand SegFormer architecture
* Learn Hugging Face image processors
* Learn segmentation mask handling
* Learn pixel-level prediction
* Learn Transformer-based computer vision models

```
```
