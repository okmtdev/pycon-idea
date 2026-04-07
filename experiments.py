"""
トークン化実験スクリプト

さまざまなPythonコーディングスタイルのトークン数を定量的に比較し、
結果をCSVおよびグラフとして出力します。
"""

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import tiktoken

ENCODINGS = ["cl100k_base", "o200k_base"]


@dataclass
class ExperimentResult:
    """実験結果を格納するデータクラス。"""

    category: str
    label: str
    encoding: str
    code: str
    token_count: int
    char_count: int
    line_count: int
    tokens_per_line: float
    tokens_per_char: float


def count_tokens(code: str, encoding_name: str) -> int:
    """コードのトークン数を返す。"""
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(code))


def run_experiment(
    category: str,
    snippets: list[tuple[str, str]],
    encoding_name: str,
) -> list[ExperimentResult]:
    """1カテゴリの実験を実行する。"""
    results = []
    for label, code in snippets:
        token_count = count_tokens(code, encoding_name)
        char_count = len(code)
        line_count = len(code.strip().splitlines())
        results.append(
            ExperimentResult(
                category=category,
                label=label,
                encoding=encoding_name,
                code=code.strip(),
                token_count=token_count,
                char_count=char_count,
                line_count=line_count,
                tokens_per_line=round(token_count / max(line_count, 1), 2),
                tokens_per_char=round(token_count / max(char_count, 1), 4),
            )
        )
    return results


# ============================================================
# 実験データ
# ============================================================

EXPERIMENTS: dict[str, list[tuple[str, str]]] = {
    # --- 実験1: 変数名の長さ ---
    "variable_name_length": [
        (
            "1char",
            """\
def f(x, y):
    r = x + y
    return r
""",
        ),
        (
            "short",
            """\
def add(a, b):
    res = a + b
    return res
""",
        ),
        (
            "moderate",
            """\
def add(num_a, num_b):
    result = num_a + num_b
    return result
""",
        ),
        (
            "descriptive",
            """\
def add_two_numbers(first_number, second_number):
    calculated_result = first_number + second_number
    return calculated_result
""",
        ),
        (
            "verbose",
            """\
def add_two_integer_numbers_together(first_integer_number, second_integer_number):
    final_calculated_result_value = first_integer_number + second_integer_number
    return final_calculated_result_value
""",
        ),
    ],
    # --- 実験2: コメントスタイル ---
    "comment_style": [
        (
            "no_comment",
            """\
def fetch_users(db, status):
    query = db.query(User).filter(User.status == status)
    return query.all()
""",
        ),
        (
            "inline_english",
            """\
def fetch_users(db, status):
    query = db.query(User).filter(User.status == status)  # filter by status
    return query.all()
""",
        ),
        (
            "block_english",
            """\
def fetch_users(db, status):
    # Build query to fetch users filtered by their status
    query = db.query(User).filter(User.status == status)
    return query.all()
""",
        ),
        (
            "docstring_english",
            """\
def fetch_users(db, status):
    \"\"\"Fetch all users from the database filtered by status.\"\"\"
    query = db.query(User).filter(User.status == status)
    return query.all()
""",
        ),
        (
            "block_japanese",
            """\
def fetch_users(db, status):
    # ステータスでフィルタリングしてユーザーを取得する
    query = db.query(User).filter(User.status == status)
    return query.all()
""",
        ),
        (
            "docstring_japanese",
            """\
def fetch_users(db, status):
    \"\"\"ステータスに基づいてデータベースからユーザーを取得する。\"\"\"
    query = db.query(User).filter(User.status == status)
    return query.all()
""",
        ),
    ],
    # --- 実験3: 型ヒント ---
    "type_hints": [
        (
            "no_hints",
            """\
def calculate_average(numbers):
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)
""",
        ),
        (
            "basic_hints",
            """\
def calculate_average(numbers: list) -> float:
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)
""",
        ),
        (
            "detailed_hints",
            """\
def calculate_average(numbers: list[float]) -> float:
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)
""",
        ),
        (
            "full_annotations",
            """\
from typing import Sequence

def calculate_average(numbers: Sequence[float | int]) -> float:
    if not numbers:
        return 0.0
    total: float = sum(numbers)
    count: int = len(numbers)
    return total / count
""",
        ),
    ],
    # --- 実験4: snake_case vs camelCase ---
    "naming_convention": [
        (
            "snake_case",
            """\
def calculate_total_price(item_list):
    total_price = 0
    for current_item in item_list:
        item_price = current_item.unit_price * current_item.quantity
        total_price += item_price
    return total_price
""",
        ),
        (
            "camelCase",
            """\
def calculateTotalPrice(itemList):
    totalPrice = 0
    for currentItem in itemList:
        itemPrice = currentItem.unitPrice * currentItem.quantity
        totalPrice += itemPrice
    return totalPrice
""",
        ),
    ],
    # --- 実験5: 日本語 vs 英語識別子 ---
    "language_identifiers": [
        (
            "english",
            """\
def get_active_users(user_list):
    active_users = []
    for user in user_list:
        if user.is_active:
            active_users.append(user)
    return active_users
""",
        ),
        (
            "japanese",
            """\
def アクティブユーザー取得(ユーザーリスト):
    アクティブユーザー = []
    for ユーザー in ユーザーリスト:
        if ユーザー.有効:
            アクティブユーザー.append(ユーザー)
    return アクティブユーザー
""",
        ),
    ],
    # --- 実験6: リスト内包表記 vs forループ ---
    "comprehension_vs_loop": [
        (
            "list_comprehension",
            """\
def get_even_squares(numbers):
    return [x ** 2 for x in numbers if x % 2 == 0]
""",
        ),
        (
            "for_loop",
            """\
def get_even_squares(numbers):
    result = []
    for x in numbers:
        if x % 2 == 0:
            result.append(x ** 2)
    return result
""",
        ),
    ],
    # --- 実験7: f-string vs format vs % ---
    "string_formatting": [
        (
            "f_string",
            """\
def greet(name, age):
    return f"Hello, {name}! You are {age} years old."
""",
        ),
        (
            "str_format",
            """\
def greet(name, age):
    return "Hello, {}! You are {} years old.".format(name, age)
""",
        ),
        (
            "percent_format",
            """\
def greet(name, age):
    return "Hello, %s! You are %d years old." % (name, age)
""",
        ),
        (
            "concatenation",
            """\
def greet(name, age):
    return "Hello, " + name + "! You are " + str(age) + " years old."
""",
        ),
    ],
    # --- 実験8: インデントスタイル ---
    "indentation": [
        (
            "2_spaces",
            "def example():\n  if True:\n    for i in range(10):\n      if i > 5:\n        print(i)\n",
        ),
        (
            "4_spaces",
            "def example():\n    if True:\n        for i in range(10):\n            if i > 5:\n                print(i)\n",
        ),
        (
            "tab",
            "def example():\n\tif True:\n\t\tfor i in range(10):\n\t\t\tif i > 5:\n\t\t\t\tprint(i)\n",
        ),
    ],
    # --- 実験9: エラーハンドリングパターン ---
    "error_handling": [
        (
            "bare_except",
            """\
def read_config(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return {}
""",
        ),
        (
            "specific_except",
            """\
def read_config(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
""",
        ),
        (
            "except_with_logging",
            """\
def read_config(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("Config file not found: %s", path)
        return {}
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in %s: %s", path, e)
        return {}
""",
        ),
    ],
    # --- 実験10: クラス定義スタイル ---
    "class_definition": [
        (
            "regular_class",
            """\
class User:
    def __init__(self, name, email, age):
        self.name = name
        self.email = email
        self.age = age

    def __repr__(self):
        return f"User({self.name!r}, {self.email!r}, {self.age!r})"
""",
        ),
        (
            "dataclass",
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
            "named_tuple",
            """\
from typing import NamedTuple

class User(NamedTuple):
    name: str
    email: str
    age: int
""",
        ),
        (
            "pydantic_model",
            """\
from pydantic import BaseModel

class User(BaseModel):
    name: str
    email: str
    age: int
""",
        ),
    ],
    # --- 実験11: デコレータパターン ---
    "decorator_patterns": [
        (
            "no_decorator",
            """\
def get_users():
    if not hasattr(get_users, '_cache'):
        get_users._cache = db.query(User).all()
    return get_users._cache
""",
        ),
        (
            "functools_cache",
            """\
from functools import cache

@cache
def get_users():
    return db.query(User).all()
""",
        ),
        (
            "custom_decorator",
            """\
def retry(max_attempts=3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt == max_attempts - 1:
                        raise
        return wrapper
    return decorator
""",
        ),
    ],
    # --- 実験12: コンテキストマネージャ ---
    "context_manager": [
        (
            "try_finally",
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
            "with_statement",
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
    ],
}


def run_all_experiments() -> list[ExperimentResult]:
    """全実験を全エンコーディングで実行する。"""
    all_results = []
    for encoding_name in ENCODINGS:
        for category, snippets in EXPERIMENTS.items():
            results = run_experiment(category, snippets, encoding_name)
            all_results.extend(results)
    return all_results


def print_results(results: list[ExperimentResult]) -> None:
    """結果をテーブル形式で表示する。"""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()

        current_category = ""
        current_encoding = ""

        for r in results:
            if r.category != current_category or r.encoding != current_encoding:
                current_category = r.category
                current_encoding = r.encoding
                table = Table(
                    title=f"{r.category} ({r.encoding})",
                    show_lines=True,
                )
                table.add_column("Label", style="cyan", min_width=20)
                table.add_column("Tokens", justify="right", style="bold")
                table.add_column("Chars", justify="right")
                table.add_column("Lines", justify="right")
                table.add_column("Tok/Line", justify="right")
                table.add_column("Tok/Char", justify="right")

                category_results = [
                    x
                    for x in results
                    if x.category == current_category
                    and x.encoding == current_encoding
                ]
                min_tokens = min(x.token_count for x in category_results)
                for cr in category_results:
                    ratio = f"({cr.token_count / min_tokens:.2f}x)"
                    table.add_row(
                        cr.label,
                        f"{cr.token_count} {ratio}",
                        str(cr.char_count),
                        str(cr.line_count),
                        str(cr.tokens_per_line),
                        str(cr.tokens_per_char),
                    )
                console.print(table)
                console.print()
    except ImportError:
        # richがない場合はプレーンテキストで出力
        current_key = ""
        for r in results:
            key = f"{r.category}|{r.encoding}"
            if key != current_key:
                current_key = key
                print(f"\n=== {r.category} ({r.encoding}) ===")
                print(
                    f"{'Label':<25} {'Tokens':>7} {'Chars':>7} "
                    f"{'Lines':>6} {'Tok/Line':>9} {'Tok/Char':>9}"
                )
                print("-" * 70)
            print(
                f"{r.label:<25} {r.token_count:>7} {r.char_count:>7} "
                f"{r.line_count:>6} {r.tokens_per_line:>9} {r.tokens_per_char:>9}"
            )


def export_csv(results: list[ExperimentResult], output_path: str = "results.csv") -> None:
    """結果をCSVファイルに出力する。"""
    fieldnames = [
        "category",
        "label",
        "encoding",
        "token_count",
        "char_count",
        "line_count",
        "tokens_per_line",
        "tokens_per_char",
        "code",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "category": r.category,
                    "label": r.label,
                    "encoding": r.encoding,
                    "token_count": r.token_count,
                    "char_count": r.char_count,
                    "line_count": r.line_count,
                    "tokens_per_line": r.tokens_per_line,
                    "tokens_per_char": r.tokens_per_char,
                    "code": r.code,
                }
            )
    print(f"Results exported to {output_path}")


def plot_results(results: list[ExperimentResult], output_dir: str = "figures") -> None:
    """結果をグラフとして保存する。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is not installed. Install it with: pip install matplotlib",
            file=sys.stderr,
        )
        return

    Path(output_dir).mkdir(exist_ok=True)

    # エンコーディングごとにカテゴリ別の棒グラフを作成
    for encoding_name in ENCODINGS:
        enc_results = [r for r in results if r.encoding == encoding_name]
        categories = sorted(set(r.category for r in enc_results))

        for category in categories:
            cat_results = [r for r in enc_results if r.category == category]
            labels = [r.label for r in cat_results]
            token_counts = [r.token_count for r in cat_results]

            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(range(len(labels)), token_counts, color="#4A90D9")

            # 各バーにトークン数を表示
            for bar, count in zip(bars, token_counts):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel("Token Count")
            ax.set_title(f"{category} ({encoding_name})")
            plt.tight_layout()

            filename = f"{output_dir}/{category}_{encoding_name}.png"
            fig.savefig(filename, dpi=150)
            plt.close(fig)
            print(f"Saved: {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run tokenization experiments on Python code snippets"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="Export results to CSV (specify output path)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate bar chart plots (requires matplotlib)",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="figures",
        help="Directory to save plots (default: figures)",
    )

    args = parser.parse_args()

    print("Running tokenization experiments...\n")
    all_results = run_all_experiments()
    print_results(all_results)

    if args.csv:
        export_csv(all_results, args.csv)

    if args.plot:
        plot_results(all_results, args.plot_dir)
