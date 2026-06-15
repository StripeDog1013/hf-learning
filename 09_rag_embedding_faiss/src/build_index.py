import os

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CUDA_ID,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_LENGTH,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    INDEX_DIR,
    METADATA_PATH,
    NORMALIZE_EMBEDDINGS,
    PASSAGE_PREFIX,
    PHYSICAL_CUDA_ID,
    SEED,
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
    create_dirs,
    load_documents,
    print_header,
    save_json,
    split_text,
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

    return torch.sum(
        token_embeddings * input_mask_expanded,
        dim=1,
    ) / torch.clamp(
        input_mask_expanded.sum(dim=1),
        min=1e-9,
    )


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
def encode_texts(
    texts: list[str],
    tokenizer,
    model,
    device: torch.device,
) -> np.ndarray:

    embeddings = []

    for start in range(
        0,
        len(texts),
        EMBEDDING_BATCH_SIZE,
    ):
        batch_texts = texts[
            start:start + EMBEDDING_BATCH_SIZE
        ]

        inputs = tokenizer(
            batch_texts,
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

        batch_embeddings = mean_pooling(
            outputs,
            inputs["attention_mask"],
        )

        if NORMALIZE_EMBEDDINGS:
            batch_embeddings = F.normalize(
                batch_embeddings,
                p=2,
                dim=1,
            )

        embeddings.append(
            batch_embeddings.cpu().numpy()
        )

    return np.vstack(embeddings).astype("float32")


def build_chunks(
    documents: list[dict],
) -> tuple[list[str], list[dict]]:

    texts = []
    metadata = []

    for doc_id, document in enumerate(documents):

        source = document["source"]
        text = document["text"]

        chunks = split_text(
            text=text,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        for chunk_id, chunk in enumerate(chunks):

            texts.append(
                PASSAGE_PREFIX + chunk
            )

            metadata.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "source": source,
                    "text": chunk,
                }
            )

    return texts, metadata


def build_faiss_index(
    embeddings: np.ndarray,
):

    embedding_dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        embedding_dim
    )

    index.add(embeddings)

    return index


def main():

    print_header("Build FAISS Index")

    create_dirs()
    set_seed(SEED)

    device = get_torch_device(
        cuda_id=CUDA_ID
    )

    print_device_info(
        cuda_id=CUDA_ID
    )

    documents = load_documents()

    if len(documents) == 0:
        raise RuntimeError(
            "No documents found. "
            "Please add .txt or .md files "
            "to data/documents/"
        )

    print(
        f"\nLoaded documents: "
        f"{len(documents)}"
    )

    texts, metadata = build_chunks(
        documents
    )

    if len(texts) == 0:
        raise RuntimeError(
            "No text chunks were created."
        )

    print(
        f"Created chunks: "
        f"{len(texts)}"
    )

    tokenizer, model = load_embedding_model(
        device=device
    )

    print(
        f"\nEmbedding model: "
        f"{EMBEDDING_MODEL_NAME}"
    )

    embeddings = encode_texts(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )

    print(
        f"Embeddings shape: "
        f"{embeddings.shape}"
    )

    index = build_faiss_index(
        embeddings
    )

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(FAISS_INDEX_PATH),
    )

    save_json(
        metadata,
        METADATA_PATH,
    )

    print(
        f"\nSaved FAISS index: "
        f"{FAISS_INDEX_PATH}"
    )

    print(
        f"Saved metadata: "
        f"{METADATA_PATH}"
    )

    print("\nBuild finished.")


if __name__ == "__main__":
    main()