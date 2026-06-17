import ast
import operator
from datetime import datetime


# ============================================================
# Calculator
# ============================================================

ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def safe_eval_expr(
    expr: str,
) -> float:

    def _eval(node):

        if isinstance(node, ast.Expression):
            return _eval(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, int | float):
                return node.value

            raise ValueError(
                "Only int and float are allowed."
            )

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)

            if op_type not in ALLOWED_OPERATORS:
                raise ValueError(
                    f"Unsupported operator: {op_type}"
                )

            return ALLOWED_OPERATORS[op_type](
                _eval(node.left),
                _eval(node.right),
            )

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)

            if op_type not in ALLOWED_OPERATORS:
                raise ValueError(
                    f"Unsupported operator: {op_type}"
                )

            return ALLOWED_OPERATORS[op_type](
                _eval(node.operand)
            )

        raise ValueError(
            f"Unsupported expression: {type(node)}"
        )

    tree = ast.parse(
        expr,
        mode="eval",
    )

    return _eval(tree)


def calculator(
    expression: str,
) -> str:

    try:
        result = safe_eval_expr(
            expression
        )

        return str(result)

    except Exception as e:
        return (
            "Calculation error: "
            f"{e}"
        )


# ============================================================
# Time Tool
# ============================================================

def get_current_time() -> str:

    now = datetime.now()

    return now.strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# ============================================================
# Text Tool
# ============================================================

def count_text_length(
    text: str,
) -> str:

    return str(len(text))


def count_words(
    text: str,
) -> str:

    words = text.split()

    return str(len(words))


# ============================================================
# List Tool
# ============================================================

def sum_numbers(
    numbers: list[int | float],
) -> str:

    return str(sum(numbers))


def average_numbers(
    numbers: list[int | float],
) -> str:

    if len(numbers) == 0:
        return "Average error: empty list"

    return str(
        sum(numbers) / len(numbers)
    )


# ============================================================
# Tool Registry
# ============================================================

TOOLS = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "count_text_length": count_text_length,
    "count_words": count_words,
    "sum_numbers": sum_numbers,
    "average_numbers": average_numbers,
}


TOOL_DESCRIPTIONS = {
    "calculator": (
        "数式を計算します。"
        "入力例: {'expression': '1 + 2 * 3'}"
    ),
    "get_current_time": (
        "現在時刻を取得します。"
        "入力例: {}"
    ),
    "count_text_length": (
        "文字数を数えます。"
        "入力例: {'text': 'hello'}"
    ),
    "count_words": (
        "単語数を数えます。"
        "入力例: {'text': 'hello world'}"
    ),
    "sum_numbers": (
        "数値リストの合計を計算します。"
        "入力例: {'numbers': [1, 2, 3]}"
    ),
    "average_numbers": (
        "数値リストの平均を計算します。"
        "入力例: {'numbers': [1, 2, 3]}"
    ),
}


def get_tool_names() -> list[str]:

    return list(
        TOOLS.keys()
    )


def get_tool_descriptions() -> str:

    lines = []

    for name, description in TOOL_DESCRIPTIONS.items():

        lines.append(
            f"- {name}: {description}"
        )

    return "\n".join(lines)


def run_tool(
    tool_name: str,
    tool_args: dict,
) -> str:

    if tool_name not in TOOLS:
        return (
            "Tool error: "
            f"unknown tool '{tool_name}'"
        )

    tool_func = TOOLS[tool_name]

    try:
        result = tool_func(
            **tool_args
        )

        return str(result)

    except Exception as e:
        return (
            "Tool execution error: "
            f"{e}"
        )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print("\n===== tools.py Test =====")

    print("\nAvailable tools:")
    print(get_tool_descriptions())

    print("\nCalculator:")
    print(
        run_tool(
            "calculator",
            {
                "expression": "1 + 2 * 3",
            },
        )
    )

    print("\nCurrent time:")
    print(
        run_tool(
            "get_current_time",
            {},
        )
    )

    print("\nText length:")
    print(
        run_tool(
            "count_text_length",
            {
                "text": "hello world",
            },
        )
    )

    print("\nSum:")
    print(
        run_tool(
            "sum_numbers",
            {
                "numbers": [1, 2, 3],
            },
        )
    )

    print("\nAverage:")
    print(
        run_tool(
            "average_numbers",
            {
                "numbers": [1, 2, 3],
            },
        )
    )

    print("\n===== Test Finished =====")