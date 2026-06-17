# 10 LLM Agent + Tool Calling

## 概要

本セクションでは、LLMに外部ツールを利用させる **Agent** の基礎を学習する。

通常のLLMは文章生成しか行えないが、Tool Callingを利用することで以下のような処理が可能になる。

* 現在時刻の取得
* 数式計算
* 文字数・単語数の計算
* 外部API呼び出し
* RAG検索

本章では、LLMが「何をするか」を判断し、Pythonが「実際の処理」を実行する構成を実装する。

---

## 学習目標

本セクションで習得する内容：

* Tool Callingの基本概念
* Agent Loop
* 複数ツールの利用
* ルールベースAgent
* LLMとPythonの役割分担
* Native Tool Callingの必要性理解

---

## 09章との違い

### 09 RAG + Embedding + FAISS

```text
質問
↓
Embedding
↓
FAISS検索
↓
LLM回答
```

RAGでは、LLMは検索結果を利用して回答する。

---

### 10 LLM Agent + Tool Calling

```text
質問
↓
LLMがツールを選択
↓
Pythonがツール実行
↓
LLMが最終回答
```

Agentでは、LLMが外部ツールを利用して行動できる。

---

## Agentとは

Agentとは、

> LLMが状況に応じてツールを選択し、結果を利用して回答する仕組み

である。

例：

```text
ユーザー:
今何時？

LLM:
get_current_time()

Python:
2026-06-17 20:00:00

LLM:
現在時刻は2026年6月17日20時00分00秒です。
```

---

## Tool Callingの役割分担

```text
LLM
↓
何をするか決める

Python
↓
実際に処理する
```

例えば計算：

```text
1 + 2 * 3
```

LLMに計算させると誤る場合がある。

```text
誤: 9
正: 7
```

Tool CallingではPythonの関数を実行するため、正しい結果が得られる。

---

## フォルダ構成

```text
10_llm_agent_tool_calling/
├─ README.md
├─ data/
├─ models/
├─ outputs/
├─ checkpoints/
└─ src/
   ├─ config.py
   ├─ device.py
   ├─ utils.py
   ├─ tools.py
   ├─ tool_calling_chat.py
   ├─ simple_agent.py
   └─ multi_tool_agent.py
```

---

## 各ファイルの説明

### config.py

各種パラメータ管理。

* モデル名
* GPU設定
* Agent設定
* 出力フォルダ

GPU固定用設定：

```python
PHYSICAL_CUDA_ID = 0
USE_CUDA_VISIBLE_DEVICES = True
CUDA_ID = 0
```

使用例：

```python
if USE_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        PHYSICAL_CUDA_ID
    )
```

---

### device.py

デバイス管理。

対応：

* CUDA
* Apple MPS
* CPU

主な機能：

* GPU情報表示
* メモリ表示
* Seed固定

---

### utils.py

共通処理。

* JSON保存
* ログ保存
* テキスト保存
* ディレクトリ作成

---

### tools.py

Agentが利用するツール群。

実装済み：

* calculator
* get_current_time
* count_text_length
* count_words
* sum_numbers
* average_numbers

---

### tool_calling_chat.py

最初のTool Calling実装。

```text
LLM
↓
JSON生成
↓
Python実行
↓
最終回答
```

プロンプトのみでTool Callingを模倣するため、不安定さを体験する教材。

---

### simple_agent.py

単一ツールAgent。

```text
質問
↓
ツール選択
↓
ツール実行
↓
最終回答
```

---

### multi_tool_agent.py

複数ツールAgent。

例：

```text
今何時？
あと 1 + 2 * 3 も計算して
```

実行：

```text
get_current_time
↓
calculator
↓
最終回答
```

---

## 実行方法

### Tool Calling

```bash
python tool_calling_chat.py
```

### Simple Agent

```bash
python simple_agent.py
```

### Multi Tool Agent

```bash
python multi_tool_agent.py
```

---

## 入力例

```text
今何時？
```

```text
1 + 2 * 3 を計算して
```

```text
"hello world" の文字数を数えて
```

```text
10, 20, 30 の平均を教えて
```

```text
今何時？ あと 100 * 20 も計算して
```

---

## 本章の重要ポイント

### LLMは万能ではない

LLMは確率モデルであり、

* 計算ミス
* JSON破損
* 不要なツール呼び出し

を起こす。

---

### Agentは制約が重要

安定したAgentには、

* ルールベース
* JSON Schema
* Native Tool Calling
* Structured Output

が重要である。

---

## 次章

### 11 Multimodal (LLaVA, Qwen-VL)

次章では、

```text
画像入力
↓
LLM理解
↓
回答生成
```

を学習する。

例：

* 画像説明
* OCR
* 物体認識
* 画像QA

マルチモーダルAIの基礎を実装する。
