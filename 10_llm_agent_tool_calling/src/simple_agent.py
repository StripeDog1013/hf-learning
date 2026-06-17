import json
import os
import re

from config import (
    CUDA_ID,
    DO_SAMPLE,
    MAX_AGENT_STEPS,
    MAX_NEW_TOKENS,
    MODEL_NAME,
    OBSERVATION_END,
    OBSERVATION_START,
    PHYSICAL_CUDA_ID,
    SEED,
    TEMPERATURE,
    TOP_P,
    TOOL_CALL_END,
    TOOL_CALL_START,
    USE_CUDA_VISIBLE_DEVICES,
)

if USE_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        PHYSICAL_CUDA_ID
    )

# OpenMP競合回避（Mac用）
os.environ[
    "KMP_DUPLICATE_LIB_OK"
] = "TRUE"

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
    print_header,
    save_chat_log,
)


def load_llm(
    device: torch.device,
):

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_NAME,
            torch_dtype=(
                torch.float16
                if device.type == "cuda"
                else torch.float32
            ),
            trust_remote_code=True,
        )
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
) -> str:

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        min_new_tokens=10,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        pad_token_id=(
            tokenizer.eos_token_id
        ),
    )

    generated = outputs[
        0,
        inputs["input_ids"].shape[1]:,
    ]

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )

    if not text.strip():
        print("[DEBUG] Empty generation")

    return text.strip()


def build_prompt(
    user_input: str,
    history: str,
) -> str:

    tool_desc = get_tool_descriptions()

    if history.strip():

        prompt = f"""
あなたはAgentです。

以下はツール実行結果です。
この結果を使って、ユーザーに最終回答してください。

重要:
- もうツールは呼び出さない
- JSONを出力しない
- tool_name を出力しない
- 自然な日本語だけで答える

ユーザー:
{user_input}

ツール実行結果:
{history}

最終回答:
"""
        return prompt

    prompt = f"""
あなたはAgentです。

利用可能なツール:
{tool_desc}

必要な場合だけ、次のJSON形式でツールを1つ呼び出してください。

{{
  "tool_name": "get_current_time",
  "arguments": {{}}
}}

ツールが不要なら、自然な日本語で回答してください。

重要:
- JSONは1個だけ
- Markdownは使わない
- Pythonコードは書かない
- 説明文を混ぜない

ユーザー:
{user_input}
"""

    return prompt


def extract_tool_call(
    text: str,
):

    pattern = (
        rf"{TOOL_CALL_START}"
        r"(.*?)"
        rf"{TOOL_CALL_END}"
    )

    match = re.search(
        pattern,
        text,
        re.DOTALL,
    )

    if match is not None:
        json_text = match.group(1).strip()

        try:
            return json.loads(json_text)
        except Exception:
            return None

    json_match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL,
    )

    if json_match is None:
        return None

    try:
        data = json.loads(
            json_match.group(0)
        )
    except Exception:
        return None

    if "tool_name" not in data:
        return None

    if "arguments" not in data:
        data["arguments"] = {}

    return data


def append_observation(
    history: str,
    observation: str,
):

    history += (
        f"\n"
        f"{OBSERVATION_START}\n"
        f"{observation}\n"
        f"{OBSERVATION_END}\n"
    )

    return history


def run_agent(
    user_input: str,
    tokenizer,
    model,
    device: torch.device,
):

    history = ""

    prompt = build_prompt(
        user_input,
        history,
    )

    response = generate_text(
        prompt,
        tokenizer,
        model,
        device,
    )

    print_header("Agent Step 1")
    print(response)

    tool_call = extract_tool_call(response)

    if tool_call is None:
        return response

    tool_name = tool_call.get(
        "tool_name",
        "",
    )

    tool_args = tool_call.get(
        "arguments",
        {},
    )

    result = run_tool(
        tool_name,
        tool_args,
    )

    print_header("Tool Result")
    print(f"Tool : {tool_name}")
    print(f"Args : {tool_args}")
    print(f"Result : {result}")

    history = append_observation(
        history,
        f"{tool_name} => {result}",
    )

    final_prompt = build_prompt(
        user_input,
        history,
    )

    final_answer = generate_text(
        final_prompt,
        tokenizer,
        model,
        device,
    )

    print_header("Agent Step 2")
    print(final_answer)

    return final_answer


def main():

    print_header(
        "Simple Agent"
    )

    set_seed(SEED)

    device = (
        get_torch_device(
            CUDA_ID
        )
    )

    print_device_info(
        CUDA_ID
    )

    tokenizer, model = (
        load_llm(device)
    )

    messages = []

    while True:

        user_input = input(
            "\n入力 "
            "(終了:q): "
        ).strip()

        if user_input.lower() in (
            "q",
            "quit",
            "exit",
        ):
            break

        if not user_input:
            continue

        answer = run_agent(
            user_input,
            tokenizer,
            model,
            device,
        )

        print_header(
            "Final Answer"
        )

        print(answer)

        messages.append(
            {
                "user": user_input,
                "assistant": answer,
            }
        )

        clear_cuda_cache()

    save_chat_log(
        messages,
        prefix="simple_agent",
    )


if __name__ == "__main__":
    main()