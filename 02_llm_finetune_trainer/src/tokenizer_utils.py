from config import MAX_LENGTH


def format_prompt(example: dict) -> str:
    instruction = example["instruction"]
    response = example["response"]

    text = (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
        f"{response}"
    )

    return text


def tokenize_example(example: dict, tokenizer) -> dict:
    text = format_prompt(example)

    tokenized = tokenizer(
        text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
    )

    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


def tokenize_dataset(dataset, tokenizer):
    tokenized_dataset = dataset.map(
        lambda example: tokenize_example(example, tokenizer),
        remove_columns=dataset.column_names,
    )

    return tokenized_dataset


def print_tokenized_sample(dataset, tokenizer, index: int = 0) -> None:
    sample = dataset[index]

    print("\n=== Tokenized Sample ===")
    print(f"input_ids length      : {len(sample['input_ids'])}")
    print(f"attention_mask length : {len(sample['attention_mask'])}")
    print(f"labels length         : {len(sample['labels'])}")

    print("\n=== Decoded Text ===")
    print(
        tokenizer.decode(
            sample["input_ids"],
            skip_special_tokens=False,
        )
    )