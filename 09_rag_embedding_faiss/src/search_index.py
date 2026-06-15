import os

from config import (
    CUDA_ID,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_LENGTH,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    NORMALIZE_EMBEDDINGS,
    PHYSICAL_CUDA_ID,
    QUERY_PREFIX,
    SEED,
    TOP_K,
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
from transformers import AutoModel, AutoTokenizer

from device import (
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


def search(
    query: str,
    index,
    metadata: list[dict],
    tokenizer,
    model,
    device: torch.device,
    top_k: int = TOP_K,
) -> list[dict]:

    query_embedding = encode_query(
        query=query,
        tokenizer=tokenizer,
        model=model,
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


def print_results(
    query: str,
    results: list[dict],
):

    print_header("Search Results")

    print(f"Query: {query}")

    for item in results:
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
        print(
            f"Chunk ID: {item['chunk_id']}"
        )
        print("\nText:")
        print(item["text"])


def main():

    print_header("FAISS Search")

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

    tokenizer, model = load_embedding_model(
        device=device
    )

    query = input(
        "\n質問を入力してください: "
    ).strip()

    if not query:
        raise ValueError(
            "Query is empty."
        )

    results = search(
        query=query,
        index=index,
        metadata=metadata,
        tokenizer=tokenizer,
        model=model,
        device=device,
        top_k=TOP_K,
    )

    print_results(
        query=query,
        results=results,
    )


if __name__ == "__main__":
    main()