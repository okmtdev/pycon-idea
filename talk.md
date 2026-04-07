# LLMが見ているPythonの世界 ― トークナイザから逆算するコードの書き方

## 概要

LLM（大規模言語モデル）がPythonコードを処理する際、人間が見ているテキストとは全く異なる「トークン列」として認識しています。本トークでは、tiktokenなどのトークナイザを使ってPythonコードがどのようにトークン化されるかを可視化し、「LLMにとって読みやすいコード」と「人間にとって読みやすいコード」の間にある興味深い差異を分析します。

変数名の長さ、コメントの書き方、型ヒントの有無といった要素がトークン数やLLMの理解精度にどう影響するかを実験を通じて明らかにし、LLM時代のコーディングスタイルについて考察します。

## 対象者

- LLMをコーディングアシスタントとして日常的に使っている開発者
- プロンプトエンジニアリングに興味のある方
- トークナイザの仕組みに興味のある方
- AI時代のコーディングベストプラクティスを探している方

## トークの構成

### 1. イントロダクション（5分）

LLMはコードを「読んで」いるのではなく、**トークン列として処理**しています。

```
# 人間が見ているもの
def calculate_total_price(items):
    return sum(item.price for item in items)

# LLMが見ているもの（トークン列）
['def', ' calculate', '_total', '_price', '(', 'items', '):', '\n', ...]
```

この「見え方の違い」を理解することで、LLMとより効果的にコミュニケーションできるようになります。

### 2. トークナイザの基礎知識（10分）

#### BPE（Byte Pair Encoding）とは

現代のLLMで広く使われるBPEアルゴリズムの概要：

1. 全てのテキストをバイト列（UTF-8）に分解
2. 最も頻出するバイトペアを新しいトークンとしてマージ
3. これを語彙サイズに達するまで繰り返す

#### BPEの具体例: `def add` がトークンになるまで

実際にBPEがどのようにマージを繰り返すかをステップバイステップで追ってみましょう。

```
ステップ0: 初期状態（バイト単位）
  ['d', 'e', 'f', ' ', 'a', 'd', 'd']

ステップ1: 'd' + 'd' が頻出ペアとしてマージ → 'dd'
  ['d', 'e', 'f', ' ', 'a', 'dd']

ステップ2: 'd' + 'e' がマージ → 'de'
  ['de', 'f', ' ', 'a', 'dd']

ステップ3: 'de' + 'f' がマージ → 'def'
  ['def', ' ', 'a', 'dd']

ステップ4: 'a' + 'dd' がマージ → 'add'
  ['def', ' ', 'add']

ステップ5: 'def' + ' ' がマージ → 'def '
  ['def ', 'add']
  → 最終的に2トークンで表現される
```

**ポイント**: 高頻出の部分文字列は積極的にマージされるため、`def`や`return`のようなPythonキーワードは1トークンに収まります。逆に、学習データに出現頻度が低い文字列（日本語識別子など）は細かく分割されます。

#### なぜBPEが重要なのか

BPEの仕組みを理解することで、以下の疑問に答えられるようになります：

- **なぜ `calculate_total` は `calcTotal` よりトークン数が多い？** → アンダースコアが分割点になるため
- **なぜ日本語コメントはコストが高い？** → UTF-8で3バイト/文字、マージ頻度も低い
- **なぜ `print` は1トークンで `my_custom_func` は複数トークン？** → 学習データでの出現頻度の差

#### 主要なトークナイザの比較

| トークナイザ | 使用モデル | 語彙サイズ | 特徴 |
|---|---|---|---|
| cl100k_base | GPT-4, GPT-3.5-turbo | ~100,000 | 安定した汎用トークナイザ |
| o200k_base | GPT-4o | ~200,000 | 語彙2倍で多言語対応が改善 |
| Claude tokenizer | Claude 3.x / 4.x | 非公開 | コード特化の最適化あり |

#### 語彙サイズの違いが意味するもの

```python
# 同じコードでもトークナイザによって結果が異なる

code = "def calculate_average(numbers: list[float]) -> float:"

# cl100k_base (GPT-4):    14トークン
# o200k_base  (GPT-4o):   13トークン ← 語彙が大きいため圧縮率が高い
```

語彙サイズが大きい = より長い部分文字列が1トークンにマージされている = 同じテキストをより少ないトークンで表現できます。ただし、語彙サイズが大きすぎるとモデルの埋め込み層が巨大になり、計算コストが増えるというトレードオフがあります。

#### 日本語テキストのトークン化コスト

```python
# 英語: 1単語 ≈ 1〜2トークン
"Filter positive values"          # 3トークン

# 日本語: 1文字 ≈ 1〜3トークン (UTF-8で3バイト/文字)
"正の値をフィルタリングする"         # 10〜15トークン

# → 同じ意味でも日本語は英語の3〜5倍のトークンを消費
```

**なぜ差が出るのか**: BPEの学習データは英語が大部分を占めるため、英語の部分文字列はマージが進んで効率的に圧縮されます。日本語はUTF-8で1文字3バイトである上に、マージの恩恵も少ないため、トークン効率が低くなります。o200k_baseでは語彙サイズの増加により多言語対応が改善されましたが、それでも英語との差は大きいです。

#### Pythonコード特有のトークン化パターン

```python
# インデントはどうトークン化される？
#   スペース4つ → 1トークン（頻出パターンのため）
#   スペース2つ → 1トークン
#   タブ文字    → 1トークン
#   → ネストが深くなると: 8スペース = 1トークン, 12スペース = 2トークン

# よくあるPythonキーワードは1トークン
# def, class, import, return, if, else, for, while, ...

# 標準ライブラリの関数名も多くが1トークン
# print, len, range, enumerate, isinstance, ...

# よく使われるパターンも1トークンになりやすい
# "self.", ".__", "def ", "return ", "(self" ...

# 一方、分割されやすいもの
# snake_caseの長い識別子 → アンダースコアで分割
# 日本語文字列 → 1文字が複数トークンに
# 珍しいライブラリ名 → 短くても複数トークンに
```

### 3. Pythonコードのトークン化を可視化する（10分）

#### デモ: `visualize_tokens.py`

本リポジトリに含まれる `visualize_tokens.py` を使って、実際にPythonコードがどのようにトークン化されるかをライブデモします。

```python
# 例: 同じ処理でもトークン数が変わる
code_a = "def calc(x): return x * 2"      # 少ないトークン
code_b = "def calculate_value(x): return x * 2"  # やや多い
code_c = """
def calculate_discounted_value(
    original_value: float,
    discount_rate: float = 0.1
) -> float:
    \"\"\"Calculate the discounted value.\"\"\"
    return original_value * (1 - discount_rate)
"""  # かなり多い
```

#### 可視化で見えてくること

- **インデント**: スペース4つは通常1トークンに圧縮される
- **snake_case vs camelCase**: アンダースコアで分割されるため、snake_caseの方がトークン数が多くなる傾向
- **文字列リテラル**: 一般的な英単語は1トークン、珍しい単語は複数トークンに分割
- **数値**: 整数は桁数によってトークン数が変わる

### 4. 実験: コーディングスタイルがトークン数に与える影響（20分）

本セクションでは `experiments.py` の定量的データを交えながら、8つの実験結果を紹介します。各実験はcl100k_base（GPT-4）とo200k_base（GPT-4o）の両方で測定しています。

#### 実験1: 変数名の長さとトークン効率

```python
# パターンA: 1文字変数名（13トークン）
def f(x, y):
    r = x + y
    return r

# パターンB: 短い変数名（16トークン）
def add(a, b):
    res = a + b
    return res

# パターンC: 適度な変数名（21トークン）
def add(num_a, num_b):
    result = num_a + num_b
    return result

# パターンD: 説明的な変数名（26トークン）
def add_two_numbers(first_number, second_number):
    calculated_result = first_number + second_number
    return calculated_result

# パターンE: 冗長な変数名（39トークン）
def add_two_integer_numbers_together(first_integer_number, second_integer_number):
    final_calculated_result_value = first_integer_number + second_integer_number
    return final_calculated_result_value
```

**結果の傾向**: トークン数はA(13) < B(16) < C(21) < D(26) < E(39)。パターンAからEで約3倍の差があります。

**考察**: ただしLLMの理解精度はB〜C ≈ D > A ≫ E。極端に短い名前はLLMが文脈を推測するコストが高く、冗長すぎる名前は情報密度が低下します。**1〜3単語の適度な変数名**がトークン効率と理解精度の最適バランスです。

#### 実験2: コメントの書き方

```python
# パターンA: コメントなし（21トークン）
def fetch_users(db, status):
    query = db.query(User).filter(User.status == status)
    return query.all()

# パターンB: 英語インラインコメント（28トークン, 1.33x）
def fetch_users(db, status):
    query = db.query(User).filter(User.status == status)  # filter by status
    return query.all()

# パターンC: 英語ブロックコメント（33トークン, 1.57x）
def fetch_users(db, status):
    # Build query to fetch users filtered by their status
    query = db.query(User).filter(User.status == status)
    return query.all()

# パターンD: 英語docstring（31トークン, 1.48x）
def fetch_users(db, status):
    """Fetch all users from the database filtered by status."""
    query = db.query(User).filter(User.status == status)
    return query.all()

# パターンE: 日本語ブロックコメント（39トークン, 1.86x）
def fetch_users(db, status):
    # ステータスでフィルタリングしてユーザーを取得する
    query = db.query(User).filter(User.status == status)
    return query.all()

# パターンF: 日本語docstring（40トークン, 1.90x）
def fetch_users(db, status):
    """ステータスに基づいてデータベースからユーザーを取得する。"""
    query = db.query(User).filter(User.status == status)
    return query.all()
```

**注目ポイント**:
- 日本語コメントは英語の**約1.4倍**のトークンを消費（同じ意味の場合）
- docstringはLLMが学習データで頻繁に見ているフォーマットで、**コメント以上の効果**がある
- 英語docstringはトークン効率と理解精度の**最適解**
- コメントが多すぎるとコンテキストウィンドウを圧迫する → 「なぜ」を書き、「何を」は書かない

#### 実験3: 型ヒントの有無

```python
# パターンA: 型ヒントなし（25トークン）
def calculate_average(numbers):
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)

# パターンB: 基本的な型ヒント（30トークン, 1.20x）
def calculate_average(numbers: list) -> float:
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)

# パターンC: 詳細な型ヒント（33トークン, 1.32x）
def calculate_average(numbers: list[float]) -> float:
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)

# パターンD: 完全なアノテーション（50トークン, 2.00x）
from typing import Sequence

def calculate_average(numbers: Sequence[float | int]) -> float:
    if not numbers:
        return 0.0
    total: float = sum(numbers)
    count: int = len(numbers)
    return total / count
```

**結果の傾向**: 型ヒントはトークン数を20〜100%増やすが、LLMがコードの意図を正確に理解する助けになります。

**深掘り**: 型ヒントのトークンコストは「固定費」であり、関数本体が長くなるほどその比率は下がります。関数シグネチャの型ヒントは**投資対効果が高い**一方、ローカル変数への型アノテーションは効果が限定的です。

#### 実験4: snake_case vs camelCase

```python
# パターンA: snake_case（37トークン）
def calculate_total_price(item_list):
    total_price = 0
    for current_item in item_list:
        item_price = current_item.unit_price * current_item.quantity
        total_price += item_price
    return total_price

# パターンB: camelCase（31トークン, 0.84x）
def calculateTotalPrice(itemList):
    totalPrice = 0
    for currentItem in itemList:
        itemPrice = currentItem.unitPrice * currentItem.quantity
        totalPrice += itemPrice
    return totalPrice
```

**注目ポイント**: camelCaseの方がトークン効率は高い（約16%少ない）。これはアンダースコアがBPEの分割点として機能するためです。`calculate_total_price` は `['calculate', '_total', '_price']` の3トークンに分割されますが、`calculateTotalPrice` は `['calculate', 'Total', 'Price']` と同じく3トークンになる場合もあれば、`['calcul', 'ateTotal', 'Price']` のようにより少ない分割になることもあります。

**ただし**: PythonではPEP 8でsnake_caseが標準です。LLMの学習データもPythonコードではsnake_caseが圧倒的多数のため、**PEP 8に従うことがLLMの理解精度を高めます**。トークン数の微差よりも規約の一貫性が重要です。

#### 実験5: 日本語 vs 英語の識別子

```python
# パターンA: 英語識別子（30トークン）
def get_active_users(user_list):
    active_users = []
    for user in user_list:
        if user.is_active:
            active_users.append(user)
    return active_users

# パターンB: 日本語識別子（62トークン, 2.07x）
def アクティブユーザー取得(ユーザーリスト):
    アクティブユーザー = []
    for ユーザー in ユーザーリスト:
        if ユーザー.有効:
            アクティブユーザー.append(ユーザー)
    return アクティブユーザー
```

**衝撃的な結果**: 同じロジックでも日本語識別子は**2倍以上**のトークンを消費します。日本語はUTF-8で1文字3バイトに加え、BPEのマージ効率も低いためです。さらに、LLMの学習データにおいて日本語識別子のPythonコードは極めて少ないため、理解精度も低下する可能性があります。

#### 実験6: リスト内包表記 vs forループ

```python
# パターンA: リスト内包表記（19トークン）
def get_even_squares(numbers):
    return [x ** 2 for x in numbers if x % 2 == 0]

# パターンB: forループ（31トークン, 1.63x）
def get_even_squares(numbers):
    result = []
    for x in numbers:
        if x % 2 == 0:
            result.append(x ** 2)
    return result
```

**結果**: リスト内包表記はforループと比べて**約40%少ない**トークンで同じ処理を表現できます。Pythonicなイディオムは学習データに豊富に含まれるため、LLMの理解精度も高くなります。ただし、複雑なネストされた内包表記は可読性が下がるため、適度な複雑さで使い分けましょう。

#### 実験7: 文字列フォーマット

```python
# パターンA: f-string（17トークン）
def greet(name, age):
    return f"Hello, {name}! You are {age} years old."

# パターンB: str.format()（24トークン, 1.41x）
def greet(name, age):
    return "Hello, {}! You are {} years old.".format(name, age)

# パターンC: %フォーマット（22トークン, 1.29x）
def greet(name, age):
    return "Hello, %s! You are %d years old." % (name, age)

# パターンD: 文字列連結（26トークン, 1.53x）
def greet(name, age):
    return "Hello, " + name + "! You are " + str(age) + " years old."
```

**結果**: f-stringが最もトークン効率が高く、連結が最も非効率。f-stringはPython 3.6以降の標準的な方法であり、LLMの学習データでも最も多く見られるため、**理解精度・トークン効率の両面で最適**です。

#### 実験8: インデントスタイル

```python
# パターンA: 2スペース（24トークン）
def example():
  if True:
    for i in range(10):
      if i > 5:
        print(i)

# パターンB: 4スペース（21トークン, 0.88x）
def example():
    if True:
        for i in range(10):
            if i > 5:
                print(i)

# パターンC: タブ（21トークン, 0.88x）
def example():
	if True:
		for i in range(10):
			if i > 5:
				print(i)
```

**意外な結果**: 4スペースとタブは同程度で、2スペースよりもトークン効率が**良い**。これはBPEの学習データで4スペースインデントが圧倒的に多く、`    `（4スペース）や `        `（8スペース）が1トークンにマージされているためです。PEP 8推奨の4スペースが、ここでもトークン効率の面で有利です。

#### 実験結果のまとめ

| 実験 | 最もトークン効率が高い | LLM理解度のベストバランス |
|---|---|---|
| 変数名の長さ | 1文字変数名 | 1〜3単語の適度な名前 |
| コメント | コメントなし | 英語docstring |
| 型ヒント | 型ヒントなし | 関数シグネチャの型ヒント |
| 命名規約 | camelCase | snake_case (PEP 8準拠) |
| 識別子の言語 | 英語 | 英語 |
| ループ vs 内包表記 | リスト内包表記 | リスト内包表記（適度な複雑さ） |
| 文字列フォーマット | f-string | f-string |
| インデント | 4スペース/タブ | 4スペース (PEP 8準拠) |

### 5. LLMにとって読みやすいコードとは（10分）

実験結果を総合すると、以下の傾向が見えてきます。

#### トークン効率 vs 理解しやすさのトレードオフ

```
トークン効率が高い                    理解しやすさが高い
(トークン数が少ない)                  (LLMの精度が高い)
←─────────────────────────────────────────────────→

短い変数名          適度な変数名          冗長な変数名
コメントなし        英語コメント          日本語の詳細コメント
型ヒントなし        基本的な型ヒント      完全な型アノテーション
内包表記            内包表記              forループ
f-string            f-string              文字列連結
```

ここで重要なのは、**左端と右端のどちらも最適ではない**ということです。中央付近に「スイートスポット」が存在します。

#### LLMフレンドリーなコードの特徴

1. **適度な変数名**: 1〜3単語のsnake_caseが最もバランスが良い
2. **英語のdocstring**: LLMの学習データに豊富で、トークン効率も良い
3. **型ヒントの活用**: トークンコストに見合うだけの理解度向上がある（特に関数シグネチャ）
4. **Pythonicなイディオム**: リスト内包表記、f-string、コンテキストマネージャなどは少ないトークンで表現でき、LLMも正確に理解する
5. **適切なコード分割**: 1関数あたり20-30行程度が、コンテキストウィンドウの有効活用と理解しやすさを両立
6. **PEP 8準拠**: 4スペースインデント、snake_case → LLMの学習データの多数派に合致

#### なぜ「LLMフレンドリー ≈ 人間フレンドリー」なのか

LLMの学習データはGitHub上の高品質なオープンソースコードです。つまり、**多くの人間が「良い」と判断して書いたコード**がLLMの「常識」を形成しています。

```
良いPythonコード（PEP 8準拠、適度な命名、型ヒント、docstring）
         ↓ 学習データの多数派
LLMが最もよく知っているパターン
         ↓
LLMが最も正確に理解・生成できるコード
```

この好循環は、**コードの品質を上げることが自動的にLLMとの協働を改善する**ことを意味します。

#### トークン化が理解精度に影響するメカニズム

トークン数が多い＝悪い、ではありません。重要なのは**トークンの「情報密度」**です。

```python
# ケース1: トークン数は少ないが、情報が不足
def f(d, s):
    return d.q(U).f(U.s == s).a()
# → LLMは d, s, U, q, f, a の意味を推測する必要がある
# → 文脈がなければ誤った補完をする可能性が高い

# ケース2: トークン数は多いが、情報が明確
def fetch_users_by_status(db: Session, status: str) -> list[User]:
    """Fetch all users matching the given status."""
    return db.query(User).filter(User.status == status).all()
# → 各トークンが意味のある情報を運んでいる
# → LLMは高い確信度で正確な補完ができる
```

**鍵となる指標**: `情報密度 = 意味のあるトークン数 / 全トークン数`

型ヒントやdocstringはトークン数を増やしますが、それ以上にLLMの「確信度」を高めます。これは人間がコードレビューする際に型情報やドキュメントがあると理解が速くなるのと同じ原理です。

#### 実践的なTips

```python
# ❌ LLMにとって非効率な例
def ゲットユーザーネーム(ユーザーリスト):  # 日本語識別子は大量のトークンを消費
    結果 = []
    for ユーザー in ユーザーリスト:
        結果.append(ユーザー.名前)
    return 結果
# → 62トークン、LLMの学習データにないパターン

# ✅ LLMフレンドリーな例
def get_user_names(users: list[User]) -> list[str]:
    """Return a list of user names."""
    return [user.name for user in users]
# → 27トークン、LLMが熟知したパターン
# → 同じ処理を57%少ないトークンで、より高い精度で理解される
```

#### Before/After: リファクタリング実例

```python
# ❌ Before: 非効率なコード（推定85トークン）
def データ変換処理(入力データ):
    # 入力データを検証してから変換する
    変換結果 = []
    for 項目 in 入力データ:
        if 項目 is not None:
            if type(項目) == str:
                変換結果.append(項目.strip())
            elif type(項目) == int:
                変換結果.append(str(項目))
    return 変換結果

# ✅ After: LLMフレンドリーなコード（推定38トークン）
def transform_data(items: list[str | int | None]) -> list[str]:
    """Transform non-None items to stripped strings."""
    return [
        str(item).strip()
        for item in items
        if item is not None
    ]
```

### 5.5. トークン数がコストとレイテンシに与える影響（5分）

トークン数は学術的な数字ではなく、**実際のお金と時間**に直結します。

#### APIコストへの直接的影響

```
2026年4月時点の主要LLM API価格（入力トークンあたり）:

GPT-4o:        $2.50 / 1Mトークン
GPT-4.1:       $2.00 / 1Mトークン
Claude Sonnet:  $3.00 / 1Mトークン
Claude Opus:   $15.00 / 1Mトークン

例: 500行のPythonファイル（約4,000トークン）をClaudeに送信
  → 1回あたりの入力コスト: 約$0.012（Sonnet） / $0.06（Opus）
  → 1日100回のAPI呼び出し: 約$1.2（Sonnet） / $6.0（Opus）
  → 月間: 約$36（Sonnet） / $180（Opus）
```

#### コーディングスタイルによるコスト差の試算

```
同じ500行のファイルで比較:

非効率なスタイル（日本語コメント多数、冗長な命名）:
  → 約6,000トークン → 月間 $54（Sonnet）

効率的なスタイル（英語docstring、適度な命名、内包表記）:
  → 約3,500トークン → 月間 $31.5（Sonnet）

差額: 月間 $22.5 / ファイル → チーム全体で数百ドルの差に
```

#### レイテンシへの影響

トークン数はレスポンス時間にも影響します。

```
入力トークン数と処理時間の関係（概算）:

 1,000トークン → TTFT約0.5秒
 5,000トークン → TTFT約1.5秒
20,000トークン → TTFT約4.0秒

TTFT = Time To First Token（最初のトークンが返るまでの時間）
```

大きなファイルをそのまま渡すと、**体感的な待ち時間**が増えます。特にIDE統合（GitHub Copilot、Cursor等）ではリアルタイム補完の速度に直結するため、コードのトークン効率は開発体験を左右します。

#### チーム開発での累積効果

```
開発者5人のチーム × 1日200回のLLM呼び出し × 月20日

非効率なコードベース: 200回 × 5,000トークン = 1Mトークン/人/日
                   → 100Mトークン/月 → $300/月（Sonnet）

効率的なコードベース: 200回 × 3,000トークン = 600Kトークン/人/日
                   → 60Mトークン/月 → $180/月（Sonnet）

年間節約額: $1,440 + レイテンシ改善による生産性向上
```

### 6. コンテキストウィンドウを意識したコーディング（5分）

LLMにコードを渡す際、コンテキストウィンドウは有限のリソースです。

#### トークン予算の考え方

```
コンテキストウィンドウ（例: 200Kトークン）
├── システムプロンプト:        ~2,000トークン
├── ツール定義（Function Calling）: ~3,000トークン
├── 会話履歴:                ~10,000トークン
├── 渡すコードファイル:       ~5,000トークン ← ここを最適化
├── LLMの思考・推論:         ~30,000トークン
└── 応答の生成:              ~5,000トークン
残り:                       ~145,000トークン（バッファ）
```

**注意**: コンテキストウィンドウの全量を「コードの投入」に使えるわけではありません。特に最近のモデルは「思考」に大量のトークンを消費するため、実質的にコードに使えるのは全体の20-30%程度です。

#### ファイルサイズとトークン数の目安

| Pythonコードの行数 | おおよそのトークン数 | 備考 |
|---|---|---|
| 50行 | ~300-500 | 小さな関数群、理想的なサイズ |
| 100行 | ~600-1,000 | 1モジュール、問題なし |
| 500行 | ~3,000-5,000 | 大きめ、分割を検討 |
| 1,000行 | ~6,000-10,000 | LLMに全文を渡すには大きい |
| 5,000行 | ~30,000-50,000 | 分割必須、関連部分のみ渡す |

#### コンテキストウィンドウの有効活用戦略

```python
# ❌ 1,000行のファイルをそのまま渡す
"このファイルのバグを修正してください: [1,000行全文]"

# ✅ 関連部分だけを渡す
"以下の関数にバグがあります。calculate_total が負の値を返す場合があります:
[関連する50行のみ]"

# ✅✅ さらに効果的: 構造情報 + 問題箇所
"ファイル構成:
- models.py: User, Order, Productクラス
- services.py: calculate_total(注目), validate_order
- utils.py: format_currency

問題箇所 (services.py:45-65):
[20行のコード]

エラー: calculate_totalが割引適用時に負の値を返す"
```

### 7. プロンプトエンジニアリングへの応用（5分）

トークナイザの知識は、LLMにコードを効果的に渡す「プロンプトエンジニアリング」に直接応用できます。

#### コードを含むプロンプトの最適化

```python
# ❌ 非効率なプロンプト: ファイル全体を貼り付け
"""
以下のファイルを見て、バグを修正してください。
[1,000行のファイル全文]
"""
# → 入力だけで6,000〜10,000トークン消費

# ✅ 効率的なプロンプト: 構造化された情報提供
"""
## Context
- FastAPI application with SQLAlchemy ORM
- Python 3.12, async/await pattern

## Problem
`calculate_discount` returns negative values when discount_rate > 1.0

## Relevant code (services/pricing.py:45-62)
def calculate_discount(price: float, discount_rate: float) -> float:
    return price * (1 - discount_rate)

## Expected behavior
Should clamp result to minimum 0.0
"""
# → 約150トークンで同等以上の情報を提供
```

#### トークン効率を意識したプロンプト構造

```
効率的なプロンプトの構成:

1. 構造化されたコンテキスト（箇条書き）    ~50トークン
2. 問題の明確な記述                        ~30トークン
3. 関連コードの最小限の抜粋                ~100トークン
4. 期待する出力形式の指定                   ~20トークン
─────────────────────────────────────────
合計:                                      ~200トークン

vs 構造化されていないプロンプト:
"このファイルを見てバグを直して" + 全文  → ~7,000トークン
```

#### LLMへのコード提供パターン集

```python
# パターン1: 関数シグネチャ + docstring だけを渡す（API概要を伝えたい時）
"""
def fetch_users(db: Session, status: str) -> list[User]: ...
def create_order(user: User, items: list[Item]) -> Order: ...
def calculate_total(order: Order) -> Decimal: ...
"""
# → 関数の全実装を渡さなくても、型ヒントとdocstringで意図が伝わる

# パターン2: 型定義 + 問題箇所（型エラーの相談時）
"""
@dataclass
class Order:
    user_id: int
    items: list[OrderItem]
    total: Decimal
    status: Literal["pending", "confirmed", "shipped"]

# この関数で型エラーが出ます:
def update_status(order: Order, new_status: str) -> None:
    order.status = new_status  # Type error here
"""
# → 型定義と問題箇所だけで、LLMは正確に修正を提案できる

# パターン3: テストコード + 実装（テストが通らない時）
"""
# テスト（期待する動作）:
def test_discount_boundary():
    assert calculate_discount(100, 1.5) == 0.0  # FAILS: returns -50.0

# 現在の実装:
def calculate_discount(price: float, rate: float) -> float:
    return price * (1 - rate)
"""
# → テストが「仕様」として機能し、修正の方向性が明確
```

#### 日本語プロンプトのトークンコスト

```
"この関数のバグを修正してください"     → 約20トークン
"Fix the bug in this function"       → 約8トークン

"型ヒントを追加して、docstringも書いてください"  → 約25トークン
"Add type hints and docstrings"                → 約7トークン
```

日本語プロンプトは英語の2〜3倍のトークンを消費します。**コードに関する指示は英語で書く**ことで、同じコンテキストウィンドウ内でより多くのコードを渡せます。ただし、細かいニュアンスの指示は無理に英語にせず日本語で書く方が良い結果を得られる場合もあります。

### 8. まとめと今後の展望（5分）

#### Key Takeaways

1. **トークナイザを知ることはLLMを知ること**: コードがどう分解されるかを理解すれば、LLMとの協働が改善する
2. **人間にとって良いコード ≈ LLMにとっても良いコード**: PEP 8準拠、適切な命名、型ヒント、docstringは人間にもLLMにも有効
3. **ただし日本語は要注意**: 日本語コメントや識別子はトークンコストが高い。識別子は英語、コメントは英語docstring優先、必要な場合のみ日本語を使う
4. **コンテキストウィンドウは有限資源**: ファイルのトークン数を意識した設計が重要。全文を渡すのではなく、必要な部分を構造化して渡す
5. **トークン数は実際のコストに直結**: API課金、レイテンシ、チーム全体の開発効率に影響する
6. **Pythonicなコードはトークン効率も高い**: 内包表記、f-string、コンテキストマネージャなどのイディオムは美しいだけでなく効率的

#### 実践チェックリスト

今日から始められるアクション：

```
□ 変数名は1〜3単語のsnake_caseにする
□ docstringは英語で書く（Google/NumPyスタイル）
□ 関数シグネチャには型ヒントを付ける
□ リスト内包表記・f-stringを積極的に使う
□ 1関数20〜30行を目安にする
□ LLMに渡す際は関連部分だけを構造化して渡す
□ 日本語コメントは「なぜ」の説明にのみ使う
□ PEP 8準拠（4スペースインデント）を守る
```

#### 今後の展望

- **トークナイザの進化**: o200k_baseで多言語対応が改善。今後も語彙サイズの拡大と多言語最適化が進む見込み
- **コード専用トークナイザの可能性**: プログラミング言語の構文を意識したトークン化（インデント、括弧のペアリングなど）
- **IDE統合の進化**: リアルタイムでトークン数を表示するプラグイン。例: VS Code拡張でステータスバーに「このファイル: 1,234 tokens」を表示
- **LLMフレンドリーなリンター/フォーマッターの開発**: ruffやblackに「トークン効率」の観点を追加するプラグイン
- **トークンバジェットの概念**: CI/CDにトークン数チェックを組み込み、「1ファイル5,000トークン以下」をルール化する
- **マルチモーダルの影響**: コードを画像として渡す（スクリーンショット）場合のトークンコストとの比較研究

#### この分野の研究の方向性

```
現在: トークン数の最適化（本トークの内容）
  ↓
近未来: LLMの「理解度」の定量的測定
  - 同じタスクに対するコード補完の正答率
  - トークン化方法と出力品質の相関分析
  ↓
将来: LLM-Aware Software Engineering
  - LLMの特性を前提としたソフトウェア設計手法
  - 人間とLLMの協調を最大化するコーディング規約
  - トークナイザ非依存の抽象的な「コード品質」指標
```

## 参考リンク

- [tiktoken (OpenAI)](https://github.com/openai/tiktoken)
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer)
- [Byte Pair Encoding (Wikipedia)](https://en.wikipedia.org/wiki/Byte_pair_encoding)
- [PEP 8 -- Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

## 登壇者メモ

- トーク時間: 約50分 + 質疑応答10分
- デモ環境: Python 3.11+, tiktoken, rich (ターミナル表示用)
- スライドに加えてライブコーディングデモを実施
- `visualize_tokens.py --demo all` でトークン化の可視化デモ
- `experiments.py` で定量データをリアルタイムに見せる
- デモのタイミング: セクション3（可視化）とセクション4（実験結果の確認）
