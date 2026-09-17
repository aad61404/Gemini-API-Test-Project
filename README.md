# Gemini-API-Test-Project

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
2. 每次發言結尾必須標記 `[CONVERGED]`（沒有實質歧見）或 `[OPEN]`（仍有歧見）。
3. 結束條件：跑滿 `--rounds`，或**同一輪內雙方同時**標記 `[CONVERGED]`
   且已跑滿 `--min-rounds`。
4. 最後由 Agent A 產生整合結論：共識、分歧、以及最終答案。

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

常用參數：

- `--rounds N` 最多幾輪（預設 4）
- `--min-rounds N` 至少跑幾輪才允許提前收斂（預設 2）
- `--save` 把完整記錄存成 `transcripts/debate-<時間>.json`

## 測試

不需要 API key 也能跑（標記解析與收斂規則）：

```bash
python -m unittest test_debate -v
```

## 注意

`.env` 與 `transcripts/` 已列入 `.gitignore`，不會被 commit。
