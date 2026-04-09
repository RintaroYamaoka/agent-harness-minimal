"""
==============================================================================
  agent-harness-minimal: エージェントハーネスの最小実装
==============================================================================

このファイルは「エージェントハーネスとは何か」を **コードで定義** したものです。
余計な抽象化・フレームワーク・依存を全部削ぎ落として、本質だけを残しました。
全体で実コードは 50 行以下、残りはすべてコメントです。

──────────────────────────────────────────────────────────────────────────
  エージェントハーネスとは何か (1 段落で)
──────────────────────────────────────────────────────────────────────────

エージェントハーネスとは、LLM に「ツールを使って自律的に作業をやらせる」
ための **最小ラッパ** です。本質は次の 4 ステップのループだけ:

  1. LLM にメッセージを送る (ツール定義込みで)
  2. LLM が「ツールを呼びたい」と返してきたら、こちら (Python 側) でその
     ツールを実行する
  3. 実行結果を LLM に返す
  4. LLM が「もうツールは要らない (= end_turn)」と言うまで 1〜3 を繰り返す

ChatGPT の Code Interpreter も、Claude Code も、Cursor の Agent も、
Anthropic Managed Agents も、Devin も、AutoGPT も、
**すべて根本はこれと同じループ** です。違うのはツールの種類と prompt の長さ
と orchestration の上層レイヤだけ。harness 自体は本質的に小さい。

──────────────────────────────────────────────────────────────────────────
  なぜ「ハーネス」と呼ぶのか
──────────────────────────────────────────────────────────────────────────

LLM 単体は「次のトークンを予測する関数」で、外界に作用する手段を持って
いません。それに馬具 (= harness) を付けることで、初めて荷物を引っ張れる
馬になります。LLM に対する馬具 = ツール定義 + 実行ループ + 結果のフィード
バック。これが「エージェントハーネス」の語源的なニュアンスです。

──────────────────────────────────────────────────────────────────────────
  このファイルの構成
──────────────────────────────────────────────────────────────────────────

  - Tool 関連の型定義 (Schema と Callable のペア)
  - run_agent() 関数 = 4 ステップのループそのもの
  - ヘルパ: テキスト抽出と tool 実行

このファイルだけ読めば「エージェントハーネスとは何か」が完全に分かるよう
書いてあります。example.py で実際に動かせます。
"""

from __future__ import annotations

import json
from typing import Any, Callable

import anthropic

# ┌────────────────────────────────────────────────────────────────────┐
# │  ToolSpec: LLM に渡すツールの定義                                    │
# └────────────────────────────────────────────────────────────────────┘
#
# Anthropic API の `tools` パラメータには、各ツールについて 3 つの情報を
# 渡します:
#   - name        : LLM がツールを呼ぶ時に使う識別子
#   - description : LLM が「いつこのツールを使えばいいか」を判断する根拠
#   - input_schema: ツールの引数の JSON Schema (LLM が引数を組み立てる元)
#
# これを Python の dict として並べます。型は `dict` でも実用上は十分。

ToolSpec = dict[str, Any]

# ┌────────────────────────────────────────────────────────────────────┐
# │  ToolFunc: ツールの「実体」                                          │
# └────────────────────────────────────────────────────────────────────┘
#
# LLM がツール呼び出しを返してきた時に、こちら (Python) で実際に走らせる
# 関数です。引数は LLM が組み立てた dict (input_schema に従った形)、
# 戻り値は **str** に統一します (LLM に返す時は str になるので)。
# str 以外を返したい場合は json.dumps() 等で文字列化してから返します。

ToolFunc = Callable[..., str]


# ┌────────────────────────────────────────────────────────────────────┐
# │  run_agent: ループ本体 (これがエージェントハーネスの全て)              │
# └────────────────────────────────────────────────────────────────────┘
def run_agent(
    *,
    user_message: str,
    tools: list[ToolSpec],
    tool_funcs: dict[str, ToolFunc],
    system_prompt: str = "",
    model: str = "claude-opus-4-6",
    max_iterations: int = 20,
    verbose: bool = True,
) -> str:
    """
    エージェントループを 1 回 (= 1 タスク分) 走らせる。

    Args:
        user_message: ユーザーが LLM に投げる最初の指示 (= タスク内容)
        tools: ツール定義のリスト (Anthropic API の tools パラメータに渡す形)
        tool_funcs: ツール名 → 実際の Python 関数 のマッピング
                    (LLM がツールを呼んだ時、ここから関数を引いて実行する)
        system_prompt: LLM の役割設定 (省略可)
        model: 使う Claude モデル ID
        max_iterations: 無限ループ防止のための上限。これを超えたら強制終了
        verbose: True なら各ステップを print する (学習用)

    Returns:
        LLM が最後に出した自由テキスト (= タスクの最終答え)

    Raises:
        RuntimeError: max_iterations を超えた / 想定外の stop_reason が返った時

    ──────────────────────────────────────────────────────────────────
    動作の流れ (図):

        [Python: user_message を送る]
                  ↓
        [Claude API call (tools 込み)]
                  ↓
            stop_reason を見て分岐
                  │
        ┌─────────┴─────────┐
        │                   │
     "end_turn"          "tool_use"
        │                   │
   テキスト返して終了    Python 側で tool 実行
                            ↓
                        結果を tool_result として返す
                            ↓
                        ループ先頭に戻る
    ──────────────────────────────────────────────────────────────────
    """

    # --- ステップ 0: 準備 ---------------------------------------------------
    #
    # client = LLM API への接続。ANTHROPIC_API_KEY 環境変数を自動で読みます。
    # (環境変数が無い場合は、Anthropic() に api_key=... を渡せばよい)
    client = anthropic.Anthropic()

    # messages = LLM とのやり取りの履歴。Claude API は **stateless** なので、
    # 毎回の API 呼び出しで「これまでの全履歴」を渡す必要があります。
    # 初期状態は user の最初の message だけ。
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_message}
    ]

    # ループの繰り返し回数。max_iterations 防止用。
    iteration = 0

    # --- ステップ 1〜4 のループ ---------------------------------------------
    while True:
        iteration += 1
        if iteration > max_iterations:
            raise RuntimeError(
                f"max_iterations ({max_iterations}) を超えました。"
                f"LLM がツール呼び出しを終わらせていません。"
                f"prompt や max_iterations を見直してください。"
            )

        if verbose:
            print(f"\n─── iteration {iteration} ───")

        # ----------------------------------------------------------------
        # ステップ 1: LLM にメッセージを送る
        # ----------------------------------------------------------------
        #
        # ここが「LLM に問い合わせる」唯一の場所です。
        #
        # 渡しているもの:
        #   - model         : どの Claude モデルを使うか
        #   - max_tokens    : 1 回の応答で何トークンまで出力するか
        #                     (このループは複数回 API を呼ぶので 1 回分)
        #   - system        : 役割設定 (常に同じ)
        #   - tools         : LLM が呼べるツール一覧
        #   - messages      : これまでの会話履歴 (累積)
        #   - thinking      : adaptive thinking (Claude 4.6 推奨)
        #                     LLM が「どれくらい深く考えるか」を自分で決める
        #
        # API 呼び出しは同期で、応答が返ってくるまでブロックします。
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt or "You are a helpful AI agent.",
            tools=tools,
            thinking={"type": "adaptive"},
            messages=messages,
        )

        # ----------------------------------------------------------------
        # ステップ 2: assistant の応答を履歴に追加する
        # ----------------------------------------------------------------
        #
        # **これは絶対に忘れてはいけません**。次の API 呼び出しで「LLM が
        # 前にこう言った」という記録が無いと、tool 呼び出しが整合しません。
        # response.content は ContentBlock のリスト (TextBlock / ToolUseBlock /
        # ThinkingBlock などが混ざった polymorphic な配列)。
        # そのまま append して OK (Anthropic SDK が次回 serialize してくれる)。
        messages.append({"role": "assistant", "content": response.content})

        # ----------------------------------------------------------------
        # ステップ 3: stop_reason で分岐する
        # ----------------------------------------------------------------
        #
        # stop_reason は LLM が「なぜ生成を止めたか」を示す string:
        #   - "end_turn"    : 自然に終わった (= もう tool は要らない)
        #   - "tool_use"    : ツールを呼びたいから一旦止めた
        #   - "max_tokens"  : max_tokens の上限に達した (途中で切れた)
        #   - "stop_sequence": 指定した stop sequence に当たった
        #   - "refusal"     : 安全上の理由で拒否した
        #   - "pause_turn"  : server-side tool が iteration 上限で一時停止 (再送で継続)
        #
        # ハーネスとしては end_turn と tool_use の 2 つだけ扱えれば最小機能。

        if response.stop_reason == "end_turn":
            # 終わり。LLM が最後に書いたテキストを返す。
            final_text = _extract_text(response.content)
            if verbose:
                print(f"[end_turn] final answer:\n{final_text}\n")
            return final_text

        if response.stop_reason == "tool_use":
            # ツール呼び出しを実行する (ステップ 4)
            tool_results = _execute_tools(response.content, tool_funcs, verbose=verbose)

            # ----------------------------------------------------------------
            # ステップ 4: tool_result を user メッセージとして追加する
            # ----------------------------------------------------------------
            #
            # Claude API では、tool_result は **user role のメッセージとして**
            # 返します (assistant が tool を呼ぶ → user role で結果を返す、
            # という対称構造になっている)。
            #
            # 1 回の assistant turn で複数のツール呼び出しがあれば、その全部
            # の tool_result を 1 つの user メッセージにまとめて返します。
            messages.append({"role": "user", "content": tool_results})

            # ループ先頭に戻り、次の iteration へ
            continue

        # ここに来るのは想定外の stop_reason
        raise RuntimeError(
            f"想定外の stop_reason: {response.stop_reason!r}。"
            f"このハーネスは end_turn と tool_use しかサポートしていません。"
            f"max_tokens を増やすか、refusal の場合は prompt を見直してください。"
        )


# ┌────────────────────────────────────────────────────────────────────┐
# │  ヘルパ: テキスト抽出                                                │
# └────────────────────────────────────────────────────────────────────┘
def _extract_text(content_blocks: list) -> str:
    """ContentBlock のリストから text block だけを連結して 1 本の文字列に。

    response.content は TextBlock / ToolUseBlock / ThinkingBlock が混在する
    polymorphic な配列なので、type で filter してから .text を取り出す。
    """
    parts = []
    for block in content_blocks:
        if block.type == "text":
            parts.append(block.text)
    return "".join(parts)


# ┌────────────────────────────────────────────────────────────────────┐
# │  ヘルパ: ツール実行                                                  │
# └────────────────────────────────────────────────────────────────────┘
def _execute_tools(
    content_blocks: list,
    tool_funcs: dict[str, ToolFunc],
    *,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """assistant の応答から tool_use block を抜き出し、各ツールを実行する。

    1 回の assistant turn には複数の tool_use block が並ぶことがあります
    (LLM が「並列に複数のツールを使いたい」と判断した時)。そのすべてを
    順番に実行し、対応する tool_result を 1 つのリストにして返します。
    """
    results: list[dict[str, Any]] = []

    for block in content_blocks:
        if block.type != "tool_use":
            continue

        # block は ToolUseBlock。次の 3 属性を持つ:
        #   - block.id    : tool 呼び出しの一意 ID。tool_result でこの id を
        #                   指定して「どの呼び出しに対する結果か」を紐付ける
        #   - block.name  : LLM が呼びたいツール名
        #   - block.input : LLM が組み立てた引数 (dict)
        tool_name = block.name
        tool_input = block.input
        tool_use_id = block.id

        if verbose:
            print(f"  [tool_use] {tool_name}({json.dumps(tool_input, ensure_ascii=False)})")

        # tool_funcs から実体を引く
        func = tool_funcs.get(tool_name)
        if func is None:
            # LLM が定義に無いツールを呼んだ場合 (まずないが、prompt の問題で
            # 起こり得る)。エラーを返して次へ。
            result_str = f"ERROR: unknown tool {tool_name!r}"
            is_error = True
        else:
            try:
                # ツールを実行。tool_input は dict なので **kwargs として展開
                result_str = func(**tool_input)
                if not isinstance(result_str, str):
                    # 文字列以外を返したら json.dumps で str 化
                    result_str = json.dumps(result_str, ensure_ascii=False)
                is_error = False
            except Exception as e:
                # ツール実行が例外を投げた場合。LLM に「失敗した」と伝え、
                # LLM 側で別のアプローチを試させる。
                result_str = f"ERROR: {type(e).__name__}: {e}"
                is_error = True

        if verbose:
            preview = result_str[:200] + ("..." if len(result_str) > 200 else "")
            print(f"  [tool_result] {preview}")

        # tool_result block の dict 形を組み立てる
        results.append({
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": result_str,
            "is_error": is_error,
        })

    return results


# ┌────────────────────────────────────────────────────────────────────┐
# │  使い方の例                                                          │
# └────────────────────────────────────────────────────────────────────┘
#
# このファイルは import される前提です。実際に動かすには example.py を
# 見てください。最小例だけ載せておくと:
#
#     from harness import run_agent
#
#     TOOLS = [{
#         "name": "echo",
#         "description": "Echo back whatever string you pass.",
#         "input_schema": {
#             "type": "object",
#             "properties": {"text": {"type": "string"}},
#             "required": ["text"],
#         },
#     }]
#
#     TOOL_FUNCS = {
#         "echo": lambda text: f"echoed: {text}",
#     }
#
#     answer = run_agent(
#         user_message="echo ツールで 'hello' と返させて、その結果を教えて",
#         tools=TOOLS,
#         tool_funcs=TOOL_FUNCS,
#     )
#     print(answer)
#
# これだけでエージェントハーネスとして完結します。
#
# Claude Code / Managed Agents などの大型システムは、
# この最小ループの周りに「複数の agent の協調」「context 圧縮」「セッション
# 永続化」「permission 制御」「並列実行」などの上層レイヤを乗せたものです。
# 中心は常にこのループです。
