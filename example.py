"""
==============================================================================
  example.py: harness.py を使った実行例
==============================================================================

harness.py で定義した最小エージェントハーネスを、実際に動かしてみる
runnable な例です。2 つの簡単なツールを LLM に渡し、それらを組み合わせて
答えを出させます。

──────────────────────────────────────────────────────────────────────────
  この例で見せたいこと
──────────────────────────────────────────────────────────────────────────

1. **ツールは普通の Python 関数** であること。特殊な装飾も継承も要らない
2. **LLM が「いつ・どのツールを呼ぶか」を自分で判断する** こと
   (Python 側でフローを書かない、prompt にも「最初に X を呼べ」と書かない)
3. **複数のツールを連鎖的に呼ぶ判断も LLM がする** こと
4. **これがエージェントハーネスの全て** であること

──────────────────────────────────────────────────────────────────────────
  実行方法
──────────────────────────────────────────────────────────────────────────

  1. 仮想環境を用意 (推奨):
       python -m venv .venv
       .venv\\Scripts\\activate    # Windows PowerShell
       source .venv/bin/activate    # Bash/Zsh
  2. 依存をインストール:
       pip install -r requirements.txt
  3. API key を設定:
       $env:ANTHROPIC_API_KEY="sk-ant-..."   # PowerShell
       export ANTHROPIC_API_KEY="sk-ant-..."  # Bash
  4. 実行:
       python example.py
"""

from __future__ import annotations

import datetime

from harness import run_agent


# ┌────────────────────────────────────────────────────────────────────┐
# │  ツール 1: 現在時刻を返す                                            │
# └────────────────────────────────────────────────────────────────────┘
#
# ツールは「LLM が知らない外界の情報」を取るために使います。
# LLM は学習時点までの知識しか持っていないので、「今何時か」は単独では
# 答えられません。このツールを介して初めて現在時刻を知ることができます。
#
# 引数なし、ISO 8601 形式の文字列を返すだけのシンプルな関数。
def get_current_time() -> str:
    """現在時刻 (JST) を ISO 8601 形式で返す。"""
    jst = datetime.timezone(datetime.timedelta(hours=9))
    return datetime.datetime.now(jst).isoformat(timespec="seconds")


# ┌────────────────────────────────────────────────────────────────────┐
# │  ツール 2: 簡易計算                                                  │
# └────────────────────────────────────────────────────────────────────┘
#
# LLM は計算が得意ではありません (特に多桁の掛け算や、桁数の多い差分)。
# Python に計算を委譲することで、LLM は「何を計算すればいいか」だけを
# 判断すればよくなります。
#
# eval() を使うと任意コード実行リスクがあるので、本物のシステムでは絶対
# NG。ここではデモのために、安全な記号セットだけ許可した制限版にして
# います。
def calculate(expression: str) -> str:
    """単純な算術式を評価して結果を返す。

    Args:
        expression: '120 - 47' のような算術式 (数値・四則演算・括弧のみ)

    Returns:
        計算結果の文字列。エラー時は 'ERROR: ...' を返す
    """
    # セキュリティ: 算術記号と数字以外を弾く (eval の任意コード実行を防ぐ)
    allowed = set("0123456789+-*/.() ")
    if not set(expression) <= allowed:
        return f"ERROR: invalid characters in expression: {expression!r}"
    try:
        # builtins と globals を空にした eval (安全のため)
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# ┌────────────────────────────────────────────────────────────────────┐
# │  ツール定義 (Anthropic API 形式)                                     │
# └────────────────────────────────────────────────────────────────────┘
#
# それぞれの dict は「LLM に見せるツールの説明書」です。
# - name        : LLM がツールを呼ぶ時に使う識別子。Python 関数名と一致
#                 させると後で対応が楽
# - description : LLM が「いつこのツールを使うべきか」を判断する根拠。
#                 ここを丁寧に書くほど、LLM が正しく使ってくれる
# - input_schema: ツールの引数を JSON Schema で記述。LLM はこれを見て
#                 引数を組み立てる

TOOLS = [
    {
        "name": "get_current_time",
        "description": (
            "Get the current date and time in JST (Japan Standard Time). "
            "Use this whenever you need to know what time it is now. "
            "Returns ISO 8601 formatted string like '2026-04-09T15:30:00+09:00'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Evaluate a simple arithmetic expression and return the numeric result. "
            "Use this for any calculation involving numbers — addition, subtraction, "
            "multiplication, division, parentheses. Do NOT try to compute multi-digit "
            "math in your head; always use this tool when numbers are involved. "
            "Example: calculate('(120 - 47) * 2') → '146'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "An arithmetic expression like '120 - 47' or '(3 + 4) * 5'",
                }
            },
            "required": ["expression"],
        },
    },
]

# ツール名 → 実際の Python 関数 のマッピング。
# LLM が "calculate" を呼んだら、ハーネスがここから calculate() を引いて実行する。
TOOL_FUNCS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
}


# ┌────────────────────────────────────────────────────────────────────┐
# │  実行                                                                │
# └────────────────────────────────────────────────────────────────────┘
def main() -> None:
    """エージェントに「今から日付が変わるまで何分か計算して」とお願いする。

    LLM はこのタスクを解くために自分でこう判断するはず:
      1. まず get_current_time を呼んで「今が何時か」を知る必要がある
      2. 次に「24:00 から現在時刻を引く」計算が必要なので calculate を呼ぶ
      3. 答えを日本語で返す

    こちら (Python 側) は「最初に get_current_time、次に calculate」という
    順序を **prompt にも書かない** し **コードでも指示しない** ことに注目
    してください。LLM がツール定義の説明を読んで、自分で判断します。
    """

    user_message = (
        "今から今日の終わり (24:00 JST) までちょうど何分あるか教えて。"
        "計算過程も簡単に説明して。"
    )

    print("=" * 70)
    print("  agent-harness-minimal: example run")
    print("=" * 70)
    print(f"\n[user] {user_message}\n")

    # ハーネスを起動。verbose=True で各 iteration の様子が見える。
    final_answer = run_agent(
        user_message=user_message,
        tools=TOOLS,
        tool_funcs=TOOL_FUNCS,
        system_prompt=(
            "You are a helpful AI agent. You have access to tools that let you "
            "look up information and perform calculations. Use them whenever needed."
        ),
        verbose=True,
    )

    print("=" * 70)
    print("  最終回答:")
    print("=" * 70)
    print(final_answer)


if __name__ == "__main__":
    main()
