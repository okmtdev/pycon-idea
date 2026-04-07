FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# 依存関係のインストール（キャッシュ効率のためpyproject.tomlを先にコピー）
COPY pyproject.toml ./
RUN uv sync

# アプリケーションコードをコピー
COPY . .

CMD ["uv", "run", "python", "experiments.py"]
