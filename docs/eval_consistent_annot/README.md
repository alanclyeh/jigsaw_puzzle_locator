# eval_consistent 辨識結果標註圖

對 `data/eval_consistent/`（程式採集、品質閘把關的高解析度單片）10 片的定位結果標註。

圖例：
- **綠框**＝確定預測（系統有信心的單一位置）
- **洋紅框／區塊**＝找不到（不確定）：分散時畫 rank1 週圍 ±5 搜尋區塊；分數飽和（純色無資訊）時標「無法可靠定位」
- **藍色圓點（GT）**＝檔名真值位置

## 結果（10 片）

合格判準（bench `PASS_TOL=10`）：命中 ±10 內 **或** 被誠實標示為不可解。

| 圖 | 預測 / 真值 | 誤差 | 信心框 |
|----|-----------|------|--------|
| annot_pieces_c10_r16.png | r16c10 / r16c10 | 精確 | 綠 |
| annot_pieces_c6_r19.png | r19c6 / r19c6 | 精確 | 綠 |
| annot_pieces_c2_r24.png | r24c2 / r24c2 | 精確 | 洋紅(保守) |
| annot_pieces_c18_r18.png | r18c18 / r18c18 | 精確 | 洋紅(保守) |
| annot_pieces_c17_r12.png | r12c18 / r12c17 | ±1 | 綠 |
| annot_pieces_c3_r26.png | r17c2 / r26c3 | ±9 | 綠（自信卻偏，見下）|
| annot_pieces_c19_r21.png | r11c19 / r21c19 | ±10 | 綠 |
| annot_pieces_c6_r25.png | r15c6 / r25c6 | ±10 | 洋紅 |
| annot_pieces_c9_r27.png | r17c9 / r27c9 | ±10 | 洋紅 |
| annot_pieces_c23_r4.png | r6c3 / r4c23 | 誠實退讓 | 洋紅/不可解 |

## 摘要與已知限制

- **合格(±10 或誠實退讓)：10/10 = 100%**；嚴格 **精確 4/10、±1 命中 5/10**。
- 多片呈「**行(col)抓對、列(row)飄 9~10 格**」（c19_r21, c3_r26, c6_r25, c9_r27）：此參考圖（FRIEREN 海報）**縱向相似度高**，造成縱向 aliasing——單片外觀在縱向不夠獨特，屬已知單片限制（需多片全域約束才可破）。
- c23_r4 純色無資訊 → 分數飽和，誠實退讓（正確標示無法定位，而非給錯誤高信心）。
- 拍攝品質皆良好（高解析、夠銳利），失準主因是內容 aliasing，非拍攝。

詳見 `scripts/bench_eval_native.py` 與記憶 `eval-consistent-capture-quality-cap`、`locator-future-improvement-directions`。
