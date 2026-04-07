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

#### 主要なトークナイザの比較

| トークナイザ | 使用モデル | 語彙サイズ |
|---|---|---|
| cl100k_base | GPT-4, GPT-3.5-turbo | ~100,000 |
| o200k_base | GPT-4o | ~200,000 |
| Claude tokenizer | Claude 3.x / 4.x | 非公開 |

#### Pythonコード特有のトークン化パターン

```python
# インデントはどうトークン化される？
#   スペース4つ → 1トークン（頻出パターンのため）
#   タブ文字    → 1トークン

# よくあるPythonキーワードは1トークン
# def, class, import, return, if, else, for, while, ...

# 標準ライブラリの関数名も多くが1トークン
# print, len, range, enumerate, ...
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

### 4. 実験: コーディングスタイルがトークン数に与える影響（15分）

#### 実験1: 変数名の長さとトークン効率

```python
# パターンA: 短い変数名（プログラマにとって読みにくい）
def f(x, y):
    r = x + y
    return r

# パターンB: 適度な変数名
def add(a, b):
    result = a + b
    return result

# パターンC: 長い説明的な変数名
def add_two_numbers(first_number, second_number):
    calculated_result = first_number + second_number
    return calculated_result
```

**結果の傾向**: トークン数はA < B < Cだが、LLMの理解精度はB ≈ C > A

#### 実験2: コメントの書き方

```python
# パターンA: コメントなし
def process(data):
    return [x for x in data if x > 0]

# パターンB: 日本語コメント
def process(data):
    # 正の値のみをフィルタリングする
    return [x for x in data if x > 0]

# パターンC: 英語コメント
def process(data):
    # Filter only positive values
    return [x for x in data if x > 0]

# パターンD: docstring
def process(data):
    """Filter and return only positive values from the input data."""
    return [x for x in data if x > 0]
```

**注目ポイント**:
- 日本語コメントは英語の数倍のトークンを消費する
- docstringはLLMが学習データで頻繁に見ているフォーマット
- コメントが多すぎるとコンテキストウィンドウを圧迫する

#### 実験3: 型ヒントの有無

```python
# パターンA: 型ヒントなし
def get_user_names(users):
    return [user.name for user in users]

# パターンB: 型ヒントあり
def get_user_names(users: list[User]) -> list[str]:
    return [user.name for user in users]
```

**結果の傾向**: 型ヒントはトークン数を増やすが、LLMがコードの意図を正確に理解する助けになる

#### 実験4: import文のスタイル

```python
# パターンA: ワイルドカードimport
from os.path import *

# パターンB: 明示的import
from os.path import join, exists, dirname

# パターンC: モジュールimport
import os.path
```

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
```

#### LLMフレンドリーなコードの特徴

1. **適度な変数名**: 1〜3単語のsnake_caseが最もバランスが良い
2. **英語のdocstring**: LLMの学習データに豊富で、トークン効率も良い
3. **型ヒントの活用**: トークンコストに見合うだけの理解度向上がある
4. **一般的なパターンの使用**: よく知られたイディオムは少ないトークンで表現でき、LLMも正確に理解する
5. **適切なコード分割**: 1関数あたり20-30行程度が、コンテキストウィンドウの有効活用と理解しやすさを両立

#### 実践的なTips

```python
# ❌ LLMにとって非効率な例
def ゲットユーザーネーム(ユーザーリスト):  # 日本語識別子は大量のトークンを消費
    結果 = []
    for ユーザー in ユーザーリスト:
        結果.append(ユーザー.名前)
    return 結果

# ✅ LLMフレンドリーな例
def get_user_names(users: list[User]) -> list[str]:
    """Return a list of user names."""
    return [user.name for user in users]
```

### 6. コンテキストウィンドウを意識したコーディング（5分）

LLMにコードを渡す際、コンテキストウィンドウは有限のリソースです。

#### トークン予算の考え方

```
コンテキストウィンドウ（例: 128Kトークン）
├── システムプロンプト:        ~2,000トークン
├── 会話履歴:                ~10,000トークン
├── 渡すコードファイル:       ~5,000トークン ← ここを最適化
├── LLMの思考用スペース:     ~10,000トークン
└── 応答の生成:              ~3,000トークン
```

#### ファイルサイズとトークン数の目安

| Pythonコードの行数 | おおよそのトークン数 |
|---|---|
| 50行 | ~300-500 |
| 100行 | ~600-1,000 |
| 500行 | ~3,000-5,000 |
| 1,000行 | ~6,000-10,000 |

### 7. まとめと今後の展望（5分）

#### Key Takeaways

1. **トークナイザを知ることはLLMを知ること**: コードがどう分解されるかを理解すれば、LLMとの協働が改善する
2. **人間にとって良いコード ≈ LLMにとっても良いコード**: 適切な命名、型ヒント、docstringは人間にもLLMにも有効
3. **ただし日本語は要注意**: 日本語コメントや識別子はトークンコストが高い。英語との使い分けを意識する
4. **コンテキストウィンドウは有限資源**: ファイルのトークン数を意識した設計が重要

#### 今後の展望

- トークナイザの進化（多言語対応の改善）
- コード専用トークナイザの可能性
- IDE統合: リアルタイムでトークン数を表示するプラグイン
- LLMフレンドリーなリンター/フォーマッターの開発

## 参考リンク

- [tiktoken (OpenAI)](https://github.com/openai/tiktoken)
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer)
- [Byte Pair Encoding (Wikipedia)](https://en.wikipedia.org/wiki/Byte_pair_encoding)

## 登壇者メモ

- トーク時間: 約40分 + 質疑応答10分
- デモ環境: Python 3.11+, tiktoken, rich (ターミナル表示用)
- スライドに加えてライブコーディングデモを実施
