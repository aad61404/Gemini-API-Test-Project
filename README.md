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
- `--local-only` 兩個 agent 都跑本機 Ollama：不需要 API key、不耗任何額度
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

## 量測到的行為：自報收斂有多可信？

辯論迴圈信任每個 agent 自己回報「分歧是否已解決」。這份信任值多少，`bench_verdict.py`
直接量給你看 —— 固定一段「對方的異議明顯沒被回答」的對話，正確答案恆為 `[OPEN]`，
跑 N 次數錯誤率：

```bash
python bench_verdict.py --runs 8
```

量測結果：

| 判定來源 | `open` 案例答錯 | `resolved` 案例答錯 |
|---|---|---|
| `qwen2.5:7b` 行內標記 | 10 / 20 | 未測 |
| `qwen2.5:7b` 獨立補問 | 10 / 12 | 未測 |
| `gemini-3.5-flash` 裁判 | 0 / 9 | 0 / 10 |

三個結論：

1. **單一 7B 模型的自報收斂接近擲硬幣**，批次之間變異很大（同一組測試跑出過
   2/6、3/8、5/6）。小樣本不足以比較 prompt 寫法的優劣 —— 這也是為什麼本專案
   試過的一個「把收斂條件寫得更嚴格」的 prompt 改動最後被撤回：量測顯示它沒有幫助。
2. **獨立補問比行內標記更糟**，三批都是同方向。所以補問只作為「標記缺失」時的備援，
   不會變成預設路徑。
3. **裁判兩個案例都全對**，而且在 `resolved` 案例不會誤判為 OPEN —— 它是在判斷，
   不是一律唱反調。這就是 `--judge` 的價值。

兩個案例缺一不可：只測 `open` 的話，一個「永遠回答 OPEN」的模型也能拿滿分。

### context 成長與 `--window`

每一輪都把完整對話重送一次，所以 prompt 成本隨輪次累積。實測一場 5 輪本機辯論：

```
r1 約 550 tokens  →  r5 約 2400 tokens
```

`show_debate.py --stats` 會把這條曲線畫出來。

`--window N` 只保留開場兩則（各自的原始立場）與最近 N 則，中間用一行註記取代，
不額外花任何呼叫。效果**高度取決於辯論長度**：

| 輪數 | 重送的歷史內容（`--window 2` 相對完整重送） |
|---|---|
| 5 | −34% |
| 10 | −69% |
| 20 | −85% |
| 40 | −93% |

但**實測 5 輪的端對端節省只有 6%**（16114 vs 17226 prompt tokens）。
原因是這個長度下，每次呼叫的固定成本（system prompt、問題、立場）才是大宗，
歷史內容佔比還不高。所以短辯論不需要開 `--window`，它是為長辯論準備的。

上面的表是用真實 transcript 的發言長度推算「會被重送的歷史字元數」，
不是端對端 token 實測 —— 兩者差異正是固定成本造成的。

### 關於 `--stances`

放著不管的話，兩個 agent 往往一開場立場就很接近，然後整場互相同意 —— 這是兩個
agent 能做的最沒價值的事。指派對立立場讓辯論真的有東西要解決。

立場由 Agent A 產生，所以**非 `--local-only` 模式下品質明顯較好**。實測同一個問題：

```
qwen2.5:7b 產生： A: 程式碼註解應該寫為什麼。          ← 只是複述選項
gemini 產生：     A: 程式碼註解應該寫為什麼，因為程式碼本身已經清楚表達了在做什麼，
                     註解需要補充的是背後的業務邏輯與設計脈絡。
```

prompt 要求「一句話，說明具體主張與核心理由」，7B 模型做不到。

立場不是枷鎖：prompt 明確允許 agent 在論點被回答後讓步 ——
「放棄一個你守不住的立場是辯論在運作，不是你輸了」。

### 額度用完了怎麼辦

這正是「兩側用不同供應商」的用意。免費額度耗盡時：

```bash
python local_debate.py "你的問題" --local-only
```

兩個 agent 都改走本機 Ollama，完全不需要 API key。代價是雙方同模型，
辯論會比較平淡，但系統照常運作。`--judge` 也會一併改用本機模型。

### 免費額度

Gemini 免費層是**每個模型每天 20 次請求**。跑幾場辯論加上 benchmark 就會用完，
出現 429。此時 `--judge` 會降級為「接受辯手自己的判定」並在
`judge_rulings[].ruled = false` 記錄下來 —— 不會靜默裝作裁判同意。
本機那一側不受影響。

## 注意

`.env` 與 `transcripts/` 已列入 `.gitignore`，不會被 commit。
