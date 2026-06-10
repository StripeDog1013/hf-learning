# 00_env_check

## 概要

Hugging Face学習環境の動作確認を行う。

確認項目：

* Python
* PyTorch
* CUDA
* Transformers
* Datasets
* Accelerate

---

## 目的

以降の章で利用するライブラリが正常に動作することを確認する。

---

## 実行

```powershell
python .\src\env_check.py
```

---

## 確認項目

### Python

```python
import sys
print(sys.version)
```

---

### PyTorch

```python
import torch
print(torch.__version__)
```

---

### CUDA

```python
torch.cuda.is_available()
```

---

### GPU名

```python
torch.cuda.get_device_name(0)
```

---

### Transformers

```python
import transformers
```

---

### Datasets

```python
from datasets import load_dataset
```

---

## 完了条件

以下が確認できること。

```txt
CUDA Available : True
Transformers Import : OK
Datasets Import : OK
```

---

## 学んだこと

* Python環境確認
* CUDA確認
* Hugging Face環境確認
* PyTorch動作確認
