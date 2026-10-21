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
2. 每次發言必須用獨立一行標記 `[CONVERGED]`（沒有實質歧見）或 `[OPEN]`（仍有歧見）。
   標記可出現在開頭或結尾；若整段沒有標記，會再送一次極短的補問只要一個判定
   token（比重生整段便宜），仍判不出來就當作未收斂 —— 寧可多跑一輪，也不誤判結束。
3. 結束條件：跑滿 `--rounds`，或**同一輪內雙方同時**標記 `[CONVERGED]`
   且已跑滿 `--min-rounds`。
4. 加上 `--judge` 時，雙方自報收斂還不足以結束 —— 由第三個模型（預設
   `gemini-3.5-flash`，刻意不是兩位辯手）檢查分歧是否真的解決。否決的話，
   裁判的異議會**被放回對話裡**，讓下一輪必須回應它。
5. 最後由 Agent A 產生整合結論：共識、分歧、以及最終答案。

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

## 測試

不需要 API key 也能跑（標記解析與收斂規則）：

```bash
python -m unittest test_debate -v
```

## 量測到的行為：自報收斂有多可信？

辯論迴圈信任每個 agent 自己回報「分歧是否已解決」。這份信任值多少，`bench_verdict.py`
直接量給你看 —— 固定一段「對方的異議明顯沒被回答」的對話，正確答案恆為 `[OPEN]`，
跑 N 次數錯誤率：

```bash
python bench_verdict.py --runs 8
```

`qwen2.5:7b` 的彙整結果（三批，共 20 / 12 次取樣）：

| 取得判定的方式 | 錯判為已收斂 |
|---|---|
| 發言中的行內標記 | 10 / 20 |
| 事後獨立補問 | 10 / 12 |

兩個結論：

1. **單一 7B 模型的自報收斂接近擲硬幣**，而且批次之間變異很大（同一組測試跑出過
   2/6、3/8、5/6）。小樣本不足以比較 prompt 寫法的優劣 —— 這也是為什麼本專案
   試過的一個「把收斂條件寫得更嚴格」的 prompt 改動最後被撤回：量測顯示它沒有幫助。
2. **獨立補問比行內標記更糟**，三批都是同方向。所以補問只作為「標記缺失」時的備援，
   不會變成預設路徑。

這正是 `--judge` 存在的理由：與其寄望辯手誠實自評，不如讓第三個模型查核。

## 注意

`.env` 與 `transcripts/` 已列入 `.gitignore`，不會被 commit。
