# Gemini-API-Test-Project

[![tests](https://github.com/aad61404/Gemini-API-Test-Project/actions/workflows/tests.yml/badge.svg)](https://github.com/aad61404/Gemini-API-Test-Project/actions/workflows/tests.yml)

兩個 AI agent 針對同一個問題互相辯論，最後收斂出一個整合結論。
一個 multi-agent 架構的入門練習專案。

## 架構

| | Agent A | Agent B |
|---|---|---|
| 來源 | Gemini API（雲端，免費層） | Ollama（本機） |
| base_url | `https://generativelanguage.googleapis.com/v1beta/openai/` | `http://localhost:11434/v1` |
| SDK | `openai` | `openai` |

兩邊都走 OpenAI 相容端點，所以用同一套 SDK、只換 `base_url` 就能切換。
Gemini 免費額度用完時，本機那一側完全不受影響。

## 辯論流程

1. 兩個 agent 輪流發言，每一輪都看得到完整的對話記錄。
2. 開場陳述一律記為 `[OPEN]` —— 第一個發言者還沒聽過任何對手論點，
   邏輯上不可能「收斂」，模型卻常常直接宣告收斂。這一則不問、不信。
3. 其餘每次發言必須用獨立一行或結尾標記 `[CONVERGED]`（沒有實質歧見）或 `[OPEN]`（仍有歧見）。
   標記可出現在開頭或結尾；若整段沒有標記，會再送一次極短的補問只要一個判定
   token（比重生整段便宜），仍判不出來就當作未收斂 —— 寧可多跑一輪，也不誤判結束。
4. 結束條件：跑滿 `--rounds`，或**同一輪內雙方同時**標記 `[CONVERGED]`
   且已跑滿 `--min-rounds`。
5. 加上 `--judge` 時，雙方自報收斂還不足以結束 —— 由第三個模型（預設
   `gemini-3.5-flash`，刻意不是兩位辯手）檢查分歧是否真的解決。否決的話，
   裁判的異議會**被放回對話裡**，讓下一輪必須回應它。
6. 最後由 Agent A 產生整合結論：共識、分歧、以及最終答案。

## 安裝

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 設定

1. 到 https://aistudio.google.com/apikey 申請免費 API key（`AIza` 開頭）。
2. 複製 `.env.example` 成 `.env`，填入 `GEMINI_API_KEY`。
3. 下載本機模型：

```bash
ollama pull qwen2.5:7b
```

## 執行

```bash
python local_debate.py "遠距工作對軟體團隊的生產力是利大於弊嗎？"
```

先確認兩端都活著（兩秒，不花辯論的錢）：

```bash
python local_debate.py --check
```

常用參數：

- `--rounds N` 最多幾輪（預設 4）
- `--min-rounds N` 至少跑幾輪才允許提前收斂（預設 2）
- `--save` 把完整記錄存成 `transcripts/debate-<時間>.json`
- `--check` 只檢查兩個 agent 是否可連線後離開
- `--no-check` 跳過開跑前的連線檢查
- `--judge` 由第三個模型查核收斂是否成立（多花一次呼叫，只在雙方自報收斂時觸發）
- `--local-only` 兩個 agent 都跑本機 Ollama：不需要 API key、不耗任何額度
- `--stream` 邊產生邊印出（本機模型一則要跑一分多鐘，否則畫面像當機）
- `--window N` 每次只重送開場兩則與最近 N 則發言，中間略過（預設 0 = 全部重送）
- `--stances` 讓兩個 agent 從對立立場開場（由 Agent A 依問題自動產生，多花一次呼叫）
- `--stance-a` / `--stance-b` 自己指定開場立場（兩個要一起給）

## 讀取結果

`transcripts/` 裡是 JSON（機器讀），要人看的話：

```bash
python show_debate.py              # 最近一場，渲染成 Markdown
python show_debate.py --stats      # 只看每輪判定，不看論證內容
python show_debate.py transcripts/debate-xxx.json -o debate.md
```

`--stats` 會列出每個 agent 的判定分布與逐輪序列，用來觀察「誰比較容易宣告收斂」。

## 測試

不需要 API key 也能跑（標記解析與收斂規則）：

```bash
python -m unittest test_debate -v
```

Lint（設定在 `pyproject.toml`，CI 會跑同一套）：

```bash
pip install -r requirements-dev.txt
ruff check .
```

## 量測筆記

這個專案的設計決定幾乎都是量出來的，不是憑直覺挑的 —— 自報收斂有多不可靠、
context 成本怎麼長、立場指派有沒有用、免費額度撞牆時會怎樣，都記在
[FINDINGS.md](FINDINGS.md)。

一個例子：試過一版「把收斂條件寫得更嚴格」的 prompt，A/B 量測顯示沒有改善，
所以撤回沒有 ship。

## 注意

`.env` 與 `transcripts/` 已列入 `.gitignore`，不會被 commit。
