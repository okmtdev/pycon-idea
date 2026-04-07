"""
Pythonコードのトークン化を可視化するツール

tiktokenを使用して、Pythonコードがどのようにトークン化されるかを
色分け表示し、トークン数を分析します。
"""

import tiktoken
from rich.console import Console
from rich.table import Table
from rich.text import Text

# トークンごとの色分け用カラーパレット
COLORS = [
    "red",
    "green",
    "blue",
    "yellow",
    "magenta",
    "cyan",
    "bright_red",
    "bright_green",
    "bright_blue",
    "bright_yellow",
    "bright_magenta",
    "bright_cyan",
]

console = Console()


def get_encoder(model: str = "gpt-4o") -> tiktoken.Encoding:
    """指定モデルのトークナイザを取得する。"""
    return tiktoken.encoding_for_model(model)


def tokenize(code: str, encoding: tiktoken.Encoding) -> list[int]:
    """コードをトークン化してトークンIDのリストを返す。"""
    return encoding.encode(code)


def decode_tokens(token_ids: list[int], encoding: tiktoken.Encoding) -> list[str]:
    """トークンIDのリストを個々の文字列に復元する。"""
    return [encoding.decode([tid]) for tid in token_ids]


def visualize_tokens(code: str, encoding: tiktoken.Encoding, label: str = "") -> int:
    """コードのトークン化を色分けして表示し、トークン数を返す。"""
    token_ids = tokenize(code, encoding)
    token_strings = decode_tokens(token_ids, encoding)

    if label:
        console.print(f"\n[bold underline]{label}[/bold underline]")

    console.print(f"[dim]Original code:[/dim]")
    console.print(code.strip())
    console.print()

    # トークンを色分けして表示
    colored = Text()
    for i, token_str in enumerate(token_strings):
        color = COLORS[i % len(COLORS)]
        display = token_str.replace("\n", "↵\n").replace("\t", "→")
        colored.append(display, style=f"bold {color}")
        colored.append("|", style="dim")

    console.print("[dim]Tokenized (each color = 1 token):[/dim]")
    console.print(colored)
    console.print(f"\n[bold]Token count: {len(token_ids)}[/bold]")

    return len(token_ids)


def compare_snippets(
    snippets: list[tuple[str, str]],
    encoding_name: str = "o200k_base",
) -> None:
    """複数のコードスニペットのトークン数を比較する。"""
    encoding = tiktoken.get_encoding(encoding_name)

    results = []
    for label, code in snippets:
        count = visualize_tokens(code, encoding, label)
        results.append((label, count))
        console.print("─" * 60)

    # 比較テーブルを表示
    table = Table(title=f"Token Count Comparison ({encoding_name})")
    table.add_column("Snippet", style="cyan")
    table.add_column("Tokens", justify="right", style="bold")
    table.add_column("Ratio", justify="right", style="dim")

    min_count = min(r[1] for r in results)
    for label, count in results:
        ratio = f"{count / min_count:.2f}x"
        table.add_row(label, str(count), ratio)

    console.print()
    console.print(table)


def analyze_token_breakdown(code: str, encoding_name: str = "o200k_base") -> None:
    """トークンをカテゴリ別に分類して表示する。"""
    encoding = tiktoken.get_encoding(encoding_name)
    token_ids = tokenize(code, encoding)
    token_strings = decode_tokens(token_ids, encoding)

    categories: dict[str, list[str]] = {
        "keywords": [],
        "identifiers": [],
        "operators": [],
        "whitespace": [],
        "strings": [],
        "other": [],
    }

    python_keywords = {
        "def",
        "class",
        "import",
        "from",
        "return",
        "if",
        "else",
        "elif",
        "for",
        "while",
        "try",
        "except",
        "finally",
        "with",
        "as",
        "yield",
        "lambda",
        "pass",
        "break",
        "continue",
        "and",
        "or",
        "not",
        "in",
        "is",
        "None",
        "True",
        "False",
        "async",
        "await",
    }

    operators = {"=", "+", "-", "*", "/", ":", "(", ")", "[", "]", "{", "}", ",", "."}

    for token_str in token_strings:
        stripped = token_str.strip()
        if stripped in python_keywords:
            categories["keywords"].append(token_str)
        elif stripped in operators or all(c in "=+-*/<>!&|^~%" for c in stripped if c):
            categories["operators"].append(token_str)
        elif token_str.isspace() or token_str in ("\n", "\t", "    "):
            categories["whitespace"].append(token_str)
        elif stripped.startswith(("'", '"')) or stripped.startswith(("#",)):
            categories["strings"].append(token_str)
        elif stripped.isidentifier():
            categories["identifiers"].append(token_str)
        else:
            categories["other"].append(token_str)

    table = Table(title="Token Breakdown by Category")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right", style="bold")
    table.add_column("Percentage", justify="right")
    table.add_column("Examples", style="dim")

    total = len(token_ids)
    for cat, tokens in categories.items():
        if tokens:
            pct = f"{len(tokens) / total * 100:.1f}%"
            examples = ", ".join(repr(t) for t in tokens[:5])
            if len(tokens) > 5:
                examples += ", ..."
            table.add_row(cat, str(len(tokens)), pct, examples)

    console.print(f"\n[bold]Total tokens: {total}[/bold]")
    console.print(table)


# --- デモ用コードスニペット ---

DEMO_VARIABLE_NAMES = [
    (
        "Short names",
        """\
def f(x, y):
    r = x + y
    return r
""",
    ),
    (
        "Moderate names",
        """\
def add(a, b):
    result = a + b
    return result
""",
    ),
    (
        "Descriptive names",
        """\
def add_two_numbers(first_number, second_number):
    calculated_result = first_number + second_number
    return calculated_result
""",
    ),
]

DEMO_COMMENTS = [
    (
        "No comments",
        """\
def process(data):
    return [x for x in data if x > 0]
""",
    ),
    (
        "Japanese comment",
        """\
def process(data):
    # 正の値のみをフィルタリングする
    return [x for x in data if x > 0]
""",
    ),
    (
        "English comment",
        """\
def process(data):
    # Filter only positive values
    return [x for x in data if x > 0]
""",
    ),
    (
        "Docstring",
        """\
def process(data):
    \"\"\"Filter and return only positive values from the input data.\"\"\"
    return [x for x in data if x > 0]
""",
    ),
]

DEMO_TYPE_HINTS = [
    (
        "Without type hints",
        """\
def get_user_names(users):
    return [user.name for user in users]
""",
    ),
    (
        "With type hints",
        """\
def get_user_names(users: list) -> list:
    return [user.name for user in users]
""",
    ),
    (
        "With detailed type hints",
        """\
from typing import List

def get_user_names(users: List["User"]) -> List[str]:
    return [user.name for user in users]
""",
    ),
]

DEMO_NAMING_CONVENTION = [
    (
        "snake_case",
        """\
def calculate_total_price(item_list):
    total_price = 0
    for item in item_list:
        total_price += item.unit_price * item.quantity
    return total_price
""",
    ),
    (
        "camelCase",
        """\
def calculateTotalPrice(itemList):
    totalPrice = 0
    for item in itemList:
        totalPrice += item.unitPrice * item.quantity
    return totalPrice
""",
    ),
]

DEMO_JAPANESE_VS_ENGLISH = [
    (
        "English identifiers",
        """\
def get_user_names(user_list):
    result = []
    for user in user_list:
        result.append(user.name)
    return result
""",
    ),
    (
        "Japanese identifiers",
        """\
def ゲットユーザーネーム(ユーザーリスト):
    結果 = []
    for ユーザー in ユーザーリスト:
        結果.append(ユーザー.名前)
    return 結果
""",
    ),
]


def run_all_demos() -> None:
    """全てのデモを実行する。"""
    demos = [
        ("Variable Name Length", DEMO_VARIABLE_NAMES),
        ("Comment Style", DEMO_COMMENTS),
        ("Type Hints", DEMO_TYPE_HINTS),
        ("Naming Convention (snake_case vs camelCase)", DEMO_NAMING_CONVENTION),
        ("Japanese vs English Identifiers", DEMO_JAPANESE_VS_ENGLISH),
    ]

    for title, snippets in demos:
        console.print(f"\n{'=' * 60}")
        console.print(f"[bold cyan] Experiment: {title}[/bold cyan]")
        console.print("=" * 60)
        compare_snippets(snippets)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize Python code tokenization"
    )
    parser.add_argument(
        "--demo",
        choices=["variables", "comments", "types", "naming", "japanese", "all"],
        default="all",
        help="Which demo to run (default: all)",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Tokenize a Python file instead of running demos",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="o200k_base",
        help="Encoding to use (default: o200k_base)",
    )

    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            code = f.read()
        encoding = tiktoken.get_encoding(args.encoding)
        visualize_tokens(code, encoding, label=args.file)
        console.print()
        analyze_token_breakdown(code, args.encoding)
    else:
        demo_map = {
            "variables": [("Variable Name Length", DEMO_VARIABLE_NAMES)],
            "comments": [("Comment Style", DEMO_COMMENTS)],
            "types": [("Type Hints", DEMO_TYPE_HINTS)],
            "naming": [("Naming Convention", DEMO_NAMING_CONVENTION)],
            "japanese": [("Japanese vs English", DEMO_JAPANESE_VS_ENGLISH)],
        }

        if args.demo == "all":
            run_all_demos()
        else:
            for title, snippets in demo_map[args.demo]:
                console.print(f"\n{'=' * 60}")
                console.print(f"[bold cyan] Experiment: {title}[/bold cyan]")
                console.print("=" * 60)
                compare_snippets(snippets, args.encoding)
