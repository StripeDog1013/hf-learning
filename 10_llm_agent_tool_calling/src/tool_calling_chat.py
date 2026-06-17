import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re

from config import (
    CUDA_ID,
    DO_SAMPLE,
    MAX_NEW_TOKENS,
    MODEL_NAME,
    PHYSICAL_CUDA_ID,
    SEED,
    TEMPERATURE,
    TOP_P,
    USE_CUDA_VISIBLE_DEVICES,
)

if USE_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        PHYSICAL_CUDA_ID
    )

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from device import (
    clear_cuda_cache,
    get_torch_device,
    print_device_info,
    set_seed,
)
from tools import (
    get_tool_descriptions,
    run_tool,
)
from utils import (
    create_dirs,
    print_header,
    save_chat_log,
)


def load_llm(
    device: torch.device,
):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
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


def build_tool_selection_prompt(
    user_input: str,
) -> str:

    tool_descriptions = get_tool_descriptions()

    prompt = f"""
あなたはTool Callingエンジンです。

重要:
- JSON以外は一切出力しない
- 説明を書かない
- Markdownを書かない
- Pythonコードを書かない
- JSONは1個だけ出力する

利用可能なツール:
{tool_descriptions}

ユーザー入力:
{user_input}

出力:
"""

    return prompt


def build_final_answer_prompt(
    user_input: str,
    tool_name: str,
    tool_args: dict,
    tool_result: str,
) -> str:

    prompt = f"""あなたは親切な日本語アシスタントです。
以下のツール実行結果を使って、ユーザーに自然に回答してください。
数式計算を求められた場合、あなた自身で計算してはいけません。
必ず calculator ツールを選択してください。

ユーザーの依頼:
{user_input}

使用したツール:
{tool_name}

ツール引数:
{json.dumps(tool_args, ensure_ascii=False)}

ツール実行結果:
{tool_result}

最終回答:
"""

    return prompt


@torch.no_grad()
def generate_text(
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

    text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return text.strip()


def extract_json(
    text: str,
) -> dict:

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL,
    )

    if match is None:
        raise ValueError(
            f"JSON not found in model output:\n{text}"
        )

    return json.loads(
        match.group(0)
    )


def main():

    print_header("Tool Calling Chat")

    create_dirs()
    set_seed(SEED)

    device = get_torch_device(
        cuda_id=CUDA_ID
    )

    print_device_info(
        cuda_id=CUDA_ID
    )

    tokenizer, model = load_llm(
        device=device
    )

    messages = []

    while True:

        user_input = input(
            "\n入力してください "
            "(終了: q): "
        ).strip()

        if user_input.lower() in [
            "q",
            "quit",
            "exit",
        ]:
            print("\n終了します。")
            break

        if not user_input:
            continue

        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        selection_prompt = build_tool_selection_prompt(
            user_input=user_input
        )

        raw_output = generate_text(
            prompt=selection_prompt,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

        print_header("Model Tool Selection")
        print(raw_output)

        try:
            tool_decision = extract_json(
                raw_output
            )

        except Exception as e:
            print_header("Parse Error")
            print(e)

            messages.append(
                {
                    "role": "assistant",
                    "content": raw_output,
                }
            )
            continue

        use_tool = tool_decision.get(
            "use_tool",
            False,
        )

        if not use_tool:
            answer = tool_decision.get(
                "answer",
                raw_output,
            )

            print_header("Assistant Answer")
            print(answer)

            messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            continue

        tool_name = tool_decision.get(
            "tool_name",
            "",
        )

        tool_args = tool_decision.get(
            "arguments",
            {},
        )

        tool_result = run_tool(
            tool_name=tool_name,
            tool_args=tool_args,
        )

        print_header("Tool Result")
        print(
            f"Tool: {tool_name}"
        )
        print(
            f"Args: {tool_args}"
        )
        print(
            f"Result: {tool_result}"
        )

        final_prompt = build_final_answer_prompt(
            user_input=user_input,
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
        )

        final_answer = generate_text(
            prompt=final_prompt,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

        print_header("Assistant Answer")
        print(final_answer)

        messages.append(
            {
                "role": "assistant",
                "content": final_answer,
            }
        )

        clear_cuda_cache()

    log_path = save_chat_log(
        messages,
        prefix="tool_calling_chat",
    )

    print(f"\nSaved chat log: {log_path}")


if __name__ == "__main__":
    main()