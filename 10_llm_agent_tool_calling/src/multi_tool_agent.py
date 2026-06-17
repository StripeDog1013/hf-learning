import json
import os
import re
from typing import Any

from config import (
    CUDA_ID,
    DO_SAMPLE,
    MAX_AGENT_STEPS,
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

# MacのOpenMP競合回避用
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
from tools import run_tool
from utils import (
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


@torch.no_grad()
def generate_text(
    prompt: str,
    tokenizer,
    model,
    device: torch.device,
    max_new_tokens: int | None = None,
) -> str:

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    generate_kwargs = {
        "max_new_tokens": (
            max_new_tokens
            if max_new_tokens is not None
            else MAX_NEW_TOKENS
        ),
        "do_sample": DO_SAMPLE,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if DO_SAMPLE:
        generate_kwargs["temperature"] = TEMPERATURE
        generate_kwargs["top_p"] = TOP_P

    outputs = model.generate(
        **inputs,
        **generate_kwargs,
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


def extract_quoted_text(
    user_input: str,
) -> str | None:

    patterns = [
        r'"([^"]+)"',
        r"'([^']+)'",
        r"「([^」]+)」",
        r"『([^』]+)』",
    ]

    for pattern in patterns:
        match = re.search(pattern, user_input)

        if match:
            return match.group(1)

    return None


def extract_numbers(
    user_input: str,
) -> list[float]:

    values = re.findall(
        r"-?\d+(?:\.\d+)?",
        user_input,
    )

    return [
        float(value)
        for value in values
    ]


def extract_expression(
    user_input: str,
) -> str | None:

    match = re.search(
        r"[-+*/().\d\s]+",
        user_input,
    )

    if match is None:
        return None

    expr = match.group(0).strip()

    if not any(
        op in expr
        for op in ["+", "-", "*", "/"]
    ):
        return None

    return expr


def build_tool_plan_by_rule(
    user_input: str,
) -> list[dict[str, Any]]:

    tool_plan = []

    quoted_text = extract_quoted_text(
        user_input
    )

    numbers = extract_numbers(
        user_input
    )

    expression = extract_expression(
        user_input
    )

    if (
        "何時" in user_input
        or "時刻" in user_input
        or "現在時刻" in user_input
    ):
        tool_plan.append(
            {
                "tool_name": "get_current_time",
                "arguments": {},
            }
        )

    if "文字数" in user_input:

        text = (
            quoted_text
            if quoted_text is not None
            else user_input
        )

        tool_plan.append(
            {
                "tool_name": "count_text_length",
                "arguments": {
                    "text": text,
                },
            }
        )

    if "単語数" in user_input:

        text = (
            quoted_text
            if quoted_text is not None
            else user_input
        )

        tool_plan.append(
            {
                "tool_name": "count_words",
                "arguments": {
                    "text": text,
                },
            }
        )

    if "平均" in user_input:

        tool_plan.append(
            {
                "tool_name": "average_numbers",
                "arguments": {
                    "numbers": numbers,
                },
            }
        )

    if (
        "合計" in user_input
        or "総和" in user_input
    ):

        tool_plan.append(
            {
                "tool_name": "sum_numbers",
                "arguments": {
                    "numbers": numbers,
                },
            }
        )

    if (
        expression is not None
        and "平均" not in user_input
        and "合計" not in user_input
        and "総和" not in user_input
    ):

        tool_plan.append(
            {
                "tool_name": "calculator",
                "arguments": {
                    "expression": expression,
                },
            }
        )

    return tool_plan[
        :MAX_AGENT_STEPS
    ]


def execute_tool_plan(
    tool_plan: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    results = []

    for i, tool_call in enumerate(
        tool_plan,
        start=1,
    ):

        tool_name = tool_call.get(
            "tool_name",
            "",
        )

        arguments = tool_call.get(
            "arguments",
            {},
        )

        result = run_tool(
            tool_name=tool_name,
            tool_args=arguments,
        )

        item = {
            "step": i,
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
        }

        results.append(item)

        print_header(
            f"Tool Result {i}"
        )

        print(f"Tool : {tool_name}")
        print(f"Args : {arguments}")
        print(f"Result : {result}")

    return results


def build_final_answer_prompt(
    user_input: str,
    tool_results: list[dict[str, Any]],
) -> str:

    prompt = f"""
あなたは日本語アシスタントです。

以下のツール実行結果だけを使って、1文または2文で簡潔に回答してください。

禁止:
- 注意点を書かない
- 箇条書きにしない
- 同じ内容を繰り返さない
- 追加質問を書かない
- JSONを書かない

ユーザーの依頼:
{user_input}

ツール実行結果:
{json.dumps(tool_results, ensure_ascii=False)}

回答:
"""

    return prompt


def run_multi_tool_agent(
    user_input: str,
    tokenizer,
    model,
    device: torch.device,
) -> str:

    tool_plan = build_tool_plan_by_rule(
        user_input
    )

    print_header("Tool Plan")

    print(
        json.dumps(
            tool_plan,
            ensure_ascii=False,
            indent=2,
        )
    )

    if len(tool_plan) == 0:

        return (
            "この依頼では対応するツールを"
            "判定できませんでした。"
        )

    tool_results = execute_tool_plan(
        tool_plan
    )

    final_prompt = build_final_answer_prompt(
        user_input=user_input,
        tool_results=tool_results,
    )

    final_answer = generate_text(
        prompt=final_prompt,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=80,
    )

    final_answer = "\n".join(
        final_answer.strip().splitlines()[:2]
    )

    return final_answer


def main():

    print_header("Multi Tool Agent")

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

        answer = run_multi_tool_agent(
            user_input=user_input,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

        print_header("Final Answer")
        print(answer)

        messages.append(
            {
                "user": user_input,
                "assistant": answer,
            }
        )

        clear_cuda_cache()

    log_path = save_chat_log(
        messages,
        prefix="multi_tool_agent",
    )

    print(f"\nSaved chat log: {log_path}")


if __name__ == "__main__":
    main()