"""
Pythonコードのトークン化を可視化するツール

tiktokenを使用して、Pythonコードがどのようにトークン化されるかを
色分け表示し、トークン数を分析します。
"""

from pathlib import Path

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


# ============================================================
# グラフィカル可視化（-G / --graph オプション）
# ============================================================

# matplotlib用のカラーパレット（トークン色分け用）
MPL_COLORS = [
    "#E74C3C", "#2ECC71", "#3498DB", "#F39C12", "#9B59B6", "#1ABC9C",
    "#E67E22", "#27AE60", "#2980B9", "#F1C40F", "#8E44AD", "#16A085",
    "#D35400", "#C0392B", "#7F8C8D", "#2C3E50", "#E84393", "#00CEC9",
]


def plot_token_comparison(
    snippets: list[tuple[str, str]],
    encoding_name: str = "o200k_base",
    output_path: str = "token_comparison.png",
    title: str = "Token Count Comparison",
) -> None:
    """スニペットのトークン数比較を棒グラフとして保存する。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    encoding = tiktoken.get_encoding(encoding_name)

    labels = []
    counts = []
    for label, code in snippets:
        token_ids = tokenize(code, encoding)
        labels.append(label)
        counts.append(len(token_ids))

    min_count = min(counts)
    bar_colors = ["#2ECC71" if c == min_count else "#3498DB" for c in counts]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    bars = ax.bar(range(len(labels)), counts, color=bar_colors, edgecolor="white", linewidth=0.5)

    for bar, count in zip(bars, counts):
        ratio = count / min_count
        label_text = f"{count}"
        if ratio > 1.0:
            label_text += f"\n({ratio:.2f}x)"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.02,
            label_text,
            ha="center", va="bottom", fontweight="bold", fontsize=10,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Token Count", fontsize=12)
    ax.set_title(f"{title} ({encoding_name})", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]Saved:[/green] {output_path}")


def plot_token_breakdown_pie(
    code: str,
    encoding_name: str = "o200k_base",
    output_path: str = "token_breakdown.png",
    title: str = "Token Breakdown",
) -> None:
    """トークンカテゴリの内訳を円グラフとして保存する。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    encoding = tiktoken.get_encoding(encoding_name)
    token_ids = tokenize(code, encoding)
    token_strings = decode_tokens(token_ids, encoding)

    python_keywords = {
        "def", "class", "import", "from", "return", "if", "else", "elif",
        "for", "while", "try", "except", "finally", "with", "as", "yield",
        "lambda", "pass", "break", "continue", "and", "or", "not", "in",
        "is", "None", "True", "False", "async", "await",
    }
    operators = {"=", "+", "-", "*", "/", ":", "(", ")", "[", "]", "{", "}", ",", "."}

    categories = {"keywords": 0, "identifiers": 0, "operators": 0, "whitespace": 0, "strings": 0, "other": 0}

    for token_str in token_strings:
        stripped = token_str.strip()
        if stripped in python_keywords:
            categories["keywords"] += 1
        elif stripped in operators or all(c in "=+-*/<>!&|^~%" for c in stripped if c):
            categories["operators"] += 1
        elif token_str.isspace() or token_str in ("\n", "\t", "    "):
            categories["whitespace"] += 1
        elif stripped.startswith(("'", '"')) or stripped.startswith(("#",)):
            categories["strings"] += 1
        elif stripped.isidentifier():
            categories["identifiers"] += 1
        else:
            categories["other"] += 1

    # ゼロのカテゴリを除外
    filtered = {k: v for k, v in categories.items() if v > 0}
    pie_colors = ["#E74C3C", "#2ECC71", "#3498DB", "#95A5A6", "#F39C12", "#9B59B6"]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        filtered.values(),
        labels=filtered.keys(),
        colors=pie_colors[:len(filtered)],
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct / 100 * len(token_ids)))})",
        startangle=90,
        textprops={"fontsize": 11},
    )
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight("bold")
    ax.set_title(f"{title} (total: {len(token_ids)} tokens, {encoding_name})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]Saved:[/green] {output_path}")


def plot_token_heatmap(
    code: str,
    encoding_name: str = "o200k_base",
    output_path: str = "token_heatmap.png",
    title: str = "Token Boundaries",
) -> None:
    """コードをトークン境界で色分けした画像を保存する。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    encoding = tiktoken.get_encoding(encoding_name)
    token_ids = tokenize(code, encoding)
    token_strings = decode_tokens(token_ids, encoding)

    # テキストを行ごとに処理
    lines: list[list[tuple[str, str]]] = [[]]  # (text, color) のリスト
    for i, token_str in enumerate(token_strings):
        color = MPL_COLORS[i % len(MPL_COLORS)]
        parts = token_str.split("\n")
        for j, part in enumerate(parts):
            if j > 0:
                lines.append([])
            if part:
                lines[-1].append((part, color))

    char_width = 0.55
    line_height = 1.4
    fig_width = max(12, max((sum(len(t) for t, _ in line) for line in lines), default=40) * char_width * 0.14 + 1)
    fig_height = max(3, len(lines) * line_height * 0.14 + 1.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, fig_width / 0.14)
    ax.set_ylim(-len(lines) * line_height - 0.5, 1.5)
    ax.axis("off")

    ax.text(0, 1.0, f"{title} ({len(token_ids)} tokens, {encoding_name})",
            fontsize=13, fontweight="bold", fontfamily="monospace",
            transform=ax.transAxes, va="bottom")

    for row_idx, line_tokens in enumerate(lines):
        x = 0.5
        y = -row_idx * line_height
        for text, color in line_tokens:
            w = len(text) * char_width
            rect = patches.FancyBboxPatch(
                (x - 0.1, y - 0.5), w + 0.15, line_height * 0.85,
                boxstyle="round,pad=0.05", facecolor=color, alpha=0.25,
                edgecolor=color, linewidth=1.2,
            )
            ax.add_patch(rect)
            display = text.replace("\t", "→")
            ax.text(x, y, display, fontsize=10, fontfamily="monospace",
                    va="center", color="#2C3E50")
            x += w + 0.3

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]Saved:[/green] {output_path}")


def run_graph_demos(
    demo_key: str,
    encoding_name: str = "o200k_base",
    output_dir: str = "figures",
) -> None:
    """デモスニペットのグラフを生成する。"""
    Path(output_dir).mkdir(exist_ok=True)

    demo_map = {
        "variables": ("Variable Name Length", DEMO_VARIABLE_NAMES),
        "comments": ("Comment Style", DEMO_COMMENTS),
        "types": ("Type Hints", DEMO_TYPE_HINTS),
        "naming": ("Naming Convention", DEMO_NAMING_CONVENTION),
        "japanese": ("Japanese vs English", DEMO_JAPANESE_VS_ENGLISH),
        "comprehension": ("Comprehension vs Loop", DEMO_COMPREHENSION_VS_LOOP),
        "formatting": ("String Formatting", DEMO_STRING_FORMATTING),
        "class": ("Class Definition Style", DEMO_CLASS_DEFINITION),
        "context": ("Context Manager", DEMO_CONTEXT_MANAGER),
    }

    if demo_key == "all":
        targets = list(demo_map.items())
    else:
        targets = [(demo_key, demo_map[demo_key])]

    for key, (title, snippets) in targets:
        # 比較棒グラフ
        plot_token_comparison(
            snippets,
            encoding_name=encoding_name,
            output_path=f"{output_dir}/{key}_comparison.png",
            title=title,
        )
        # 最初のスニペットのヒートマップ（代表例）
        plot_token_heatmap(
            snippets[0][1],
            encoding_name=encoding_name,
            output_path=f"{output_dir}/{key}_heatmap.png",
            title=f"{title}: {snippets[0][0]}",
        )

    console.print(f"\n[bold green]All graphs saved to {output_dir}/[/bold green]")


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


DEMO_COMPREHENSION_VS_LOOP = [
    (
        "List comprehension",
        """\
def get_even_squares(numbers):
    return [x ** 2 for x in numbers if x % 2 == 0]
""",
    ),
    (
        "For loop",
        """\
def get_even_squares(numbers):
    result = []
    for x in numbers:
        if x % 2 == 0:
            result.append(x ** 2)
    return result
""",
    ),
]

DEMO_STRING_FORMATTING = [
    (
        "f-string",
        """\
def greet(name, age):
    return f"Hello, {name}! You are {age} years old."
""",
    ),
    (
        "str.format()",
        """\
def greet(name, age):
    return "Hello, {}! You are {} years old.".format(name, age)
""",
    ),
    (
        "% formatting",
        """\
def greet(name, age):
    return "Hello, %s! You are %d years old." % (name, age)
""",
    ),
    (
        "Concatenation",
        """\
def greet(name, age):
    return "Hello, " + name + "! You are " + str(age) + " years old."
""",
    ),
]

DEMO_CLASS_DEFINITION = [
    (
        "Regular class",
        """\
class User:
    def __init__(self, name, email, age):
        self.name = name
        self.email = email
        self.age = age
""",
    ),
    (
        "Dataclass",
        """\
from dataclasses import dataclass

@dataclass
class User:
    name: str
    email: str
    age: int
""",
    ),
    (
        "NamedTuple",
        """\
from typing import NamedTuple

class User(NamedTuple):
    name: str
    email: str
    age: int
""",
    ),
]

DEMO_CONTEXT_MANAGER = [
    (
        "try/finally",
        """\
def process_file(path):
    f = open(path)
    try:
        data = f.read()
        return data.strip()
    finally:
        f.close()
""",
    ),
    (
        "with statement",
        """\
def process_file(path):
    with open(path) as f:
        data = f.read()
        return data.strip()
""",
    ),
    (
        "pathlib",
        """\
from pathlib import Path

def process_file(path):
    return Path(path).read_text().strip()
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
        ("Comprehension vs Loop", DEMO_COMPREHENSION_VS_LOOP),
        ("String Formatting", DEMO_STRING_FORMATTING),
        ("Class Definition Style", DEMO_CLASS_DEFINITION),
        ("Context Manager", DEMO_CONTEXT_MANAGER),
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
        choices=[
            "variables", "comments", "types", "naming", "japanese",
            "comprehension", "formatting", "class", "context", "all",
        ],
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
    parser.add_argument(
        "-G", "--graph",
        action="store_true",
        help="Generate graphical visualizations (PNG) instead of terminal output",
    )
    parser.add_argument(
        "--graph-dir",
        type=str,
        default="figures",
        help="Directory to save graph images (default: figures)",
    )

    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            code = f.read()
        if args.graph:
            Path(args.graph_dir).mkdir(exist_ok=True)
            stem = Path(args.file).stem
            plot_token_heatmap(
                code,
                encoding_name=args.encoding,
                output_path=f"{args.graph_dir}/{stem}_heatmap.png",
                title=Path(args.file).name,
            )
            plot_token_breakdown_pie(
                code,
                encoding_name=args.encoding,
                output_path=f"{args.graph_dir}/{stem}_breakdown.png",
                title=Path(args.file).name,
            )
        else:
            encoding = tiktoken.get_encoding(args.encoding)
            visualize_tokens(code, encoding, label=args.file)
            console.print()
            analyze_token_breakdown(code, args.encoding)
    elif args.graph:
        run_graph_demos(args.demo, args.encoding, args.graph_dir)
    else:
        demo_map = {
            "variables": [("Variable Name Length", DEMO_VARIABLE_NAMES)],
            "comments": [("Comment Style", DEMO_COMMENTS)],
            "types": [("Type Hints", DEMO_TYPE_HINTS)],
            "naming": [("Naming Convention", DEMO_NAMING_CONVENTION)],
            "japanese": [("Japanese vs English", DEMO_JAPANESE_VS_ENGLISH)],
            "comprehension": [("Comprehension vs Loop", DEMO_COMPREHENSION_VS_LOOP)],
            "formatting": [("String Formatting", DEMO_STRING_FORMATTING)],
            "class": [("Class Definition Style", DEMO_CLASS_DEFINITION)],
            "context": [("Context Manager", DEMO_CONTEXT_MANAGER)],
        }

        if args.demo == "all":
            run_all_demos()
        else:
            for title, snippets in demo_map[args.demo]:
                console.print(f"\n{'=' * 60}")
                console.print(f"[bold cyan] Experiment: {title}[/bold cyan]")
                console.print("=" * 60)
                compare_snippets(snippets, args.encoding)
