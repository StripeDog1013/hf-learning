from peft import LoraConfig, TaskType

from config import (
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
)


def build_lora_config() -> LoraConfig:
    """
    PEFT用のLoRA設定を作成する
    """

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    return lora_config


def print_lora_config(lora_config: LoraConfig) -> None:
    """
    LoRA設定を表示する
    """

    print("\n=== LoRA Config ===")
    print(f"r           : {lora_config.r}")
    print(f"alpha       : {lora_config.lora_alpha}")
    print(f"dropout     : {lora_config.lora_dropout}")
    print(f"bias        : {lora_config.bias}")
    print(f"task_type   : {lora_config.task_type}")
    print(f"targets     : {lora_config.target_modules}")


def main() -> None:
    lora_config = build_lora_config()

    print_lora_config(lora_config)


if __name__ == "__main__":
    main()