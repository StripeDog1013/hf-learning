import os

from config import (
    CUDA_ID,
    DO_SAMPLE,
    EMBEDDING_MAX_LENGTH,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    GENERATION_MODEL_NAME,
    MAX_CONTEXT_CHARS,
    MAX_NEW_TOKENS,
    METADATA_PATH,
    NORMALIZE_EMBEDDINGS,
    PHYSICAL_CUDA_ID,
    QUERY_PREFIX,
    SEED,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    USE_CUDA_VISIBLE_DEVICES,
)

if USE_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        PHYSICAL_CUDA_ID
    )

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from device import (
    clear_cuda_cache,
    get_torch_device,
    print_device_info,
    set_seed,
)
from utils import (
    load_json,
    print_header,
)


def mean_pooling(
    model_output,
    attention_mask,
):
    token_embeddings = model_output.last_hidden_state

    input_mask_expanded = (
        attention_mask
        .unsqueeze(-1)
        .expand(token_embeddings.size())
        .float()
    )

    embeddings = torch.sum(
        token_embeddings * input_mask_expanded,
        dim=1,
    ) / torch.clamp(
        input_mask_expanded.sum(dim=1),
        min=1e-9,
    )

    return embeddings


def load_embedding_model(
    device: torch.device,
):
    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        EMBEDDING_MODEL_NAME
    )

    model.to(device)
    model.eval()

    return tokenizer, model


def load_generation_model(
    device: torch.device,
):
    tokenizer = AutoTokenizer.from_pretrained(
        GENERATION_MODEL_NAME,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        GENERATION_MODEL_NAME,
        torch_dtype=(
            torch.float16
            if device.type == "cuda"
            else torch.float32
        ),
        trust_remote_code=True,
    )

    model.to(device)
    model.eval()

    return tokenizer, model


@torch.no_grad()
def encode_query(
    query: str,
    tokenizer,
    model,
    device: torch.device,
) -> np.ndarray:

    query_text = QUERY_PREFIX + query

    inputs = tokenizer(
        [query_text],
        padding=True,
        truncation=True,
        max_length=EMBEDDING_MAX_LENGTH,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    outputs = model(**inputs)

    embedding = mean_pooling(
        outputs,
        inputs["attention_mask"],
    )

    if NORMALIZE_EMBEDDINGS:
        embedding = F.normalize(
            embedding,
            p=2,
            dim=1,
        )

    return embedding.cpu().numpy().astype(
        "float32"
    )


def load_faiss_index():
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: "
            f"{FAISS_INDEX_PATH}\n"
            "Please run build_index.py first."
        )

    return faiss.read_index(
        str(FAISS_INDEX_PATH)
    )


def retrieve_contexts(
    query: str,
    index,
    metadata: list[dict],
    embedding_tokenizer,
    embedding_model,
    device: torch.device,
    top_k: int = TOP_K,
) -> list[dict]:

    query_embedding = encode_query(
        query=query,
        tokenizer=embedding_tokenizer,
        model=embedding_model,
        device=device,
    )

    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    results = []

    for rank, idx in enumerate(indices[0]):

        if idx < 0:
            continue

        item = metadata[int(idx)].copy()

        item["rank"] = rank + 1
        item["score"] = float(scores[0][rank])

        results.append(item)

    return results


def build_context_text(
    contexts: list[dict],
) -> str:

    context_blocks = []

    total_length = 0

    for item in contexts:

        block = (
            f"[Source {item['rank']}]\n"
            f"file: {item['source']}\n"
            f"text:\n{item['text']}\n"
        )

        if total_length + len(block) > MAX_CONTEXT_CHARS:
            break

        context_blocks.append(block)
        total_length += len(block)

    return "\n".join(context_blocks)


def build_prompt(
    query: str,
    context_text: str,
) -> str:

    prompt = f"""あなたはRAGシステムの回答アシスタントです。
以下の参考情報だけを根拠に、ユーザーの質問に日本語で答えてください。

参考情報に答えがない場合は、
「参考情報だけでは分かりません」と答えてください。

# 参考情報
{context_text}

# 質問
{query}

# 回答
"""

    return prompt


@torch.no_grad()
def generate_answer(
    prompt: str,
    tokenizer,
    model,
    device: torch.device,
) -> str:

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:,
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return answer.strip()


def print_contexts(
    contexts: list[dict],
):

    print_header("Retrieved Contexts")

    for item in contexts:

        print("\n" + "-" * 60)
        print(
            f"Rank : {item['rank']}"
        )
        print(
            f"Score: {item['score']:.4f}"
        )
        print(
            f"Source: {item['source']}"
        )
        print("\nText:")
        print(item["text"])


def main():

    print_header("RAG Chat")

    set_seed(SEED)

    device = get_torch_device(
        cuda_id=CUDA_ID
    )

    print_device_info(
        cuda_id=CUDA_ID
    )

    index = load_faiss_index()

    metadata = load_json(
        METADATA_PATH
    )

    print(
        f"\nLoaded index vectors: "
        f"{index.ntotal}"
    )

    print(
        f"Loaded metadata: "
        f"{len(metadata)}"
    )

    print(
        f"\nEmbedding model: "
        f"{EMBEDDING_MODEL_NAME}"
    )

    embedding_tokenizer, embedding_model = (
        load_embedding_model(
            device=device
        )
    )

    print(
        f"Generation model: "
        f"{GENERATION_MODEL_NAME}"
    )

    generation_tokenizer, generation_model = (
        load_generation_model(
            device=device
        )
    )

    while True:

        query = input(
            "\n質問を入力してください "
            "(終了: q): "
        ).strip()

        if query.lower() in ["q", "quit", "exit"]:
            print("\n終了します。")
            break

        if not query:
            continue

        contexts = retrieve_contexts(
            query=query,
            index=index,
            metadata=metadata,
            embedding_tokenizer=embedding_tokenizer,
            embedding_model=embedding_model,
            device=device,
            top_k=TOP_K,
        )

        print_contexts(
            contexts
        )

        context_text = build_context_text(
            contexts
        )

        prompt = build_prompt(
            query=query,
            context_text=context_text,
        )

        answer = generate_answer(
            prompt=prompt,
            tokenizer=generation_tokenizer,
            model=generation_model,
            device=device,
        )

        print_header("RAG Answer")
        print(answer)

        clear_cuda_cache()


if __name__ == "__main__":
    main()