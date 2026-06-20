# eval_consistent 辨識結果標註圖

對 `data/eval_consistent/`（程式採集、品質閘把關的高解析度單片）5 片的定位結果標註。

圖例：
- **綠框**＝確定預測（系統有信心的單一位置）
- **洋紅框／區塊**＝找不到（不確定）：分散時畫 rank1 週圍 ±5 搜尋區塊；分數飽和（純色無資訊）時標「無法可靠定位」
- **藍色圓點（GT）**＝檔名真值位置

| 圖 | 結果 | 說明 |
|----|------|------|
| annot_pieces_c10_r16.png | 綠／精確 | 綠框疊住 GT 點 |
| annot_pieces_c6_r19.png | 綠／精確 | 綠框疊住 GT 點 |
| annot_pieces_c2_r24.png | 洋紅（保守）／位置正確 | GT 落在 ±5 搜尋框內 |
| annot_pieces_c19_r21.png | 綠／±10 | aliasing：col 正確、row 飄 10（綠框與 GT 同行、垂直差約 10）|
| annot_pieces_c23_r4.png | 洋紅／不可解 | 純色飽和→誠實退讓，正確標示無法定位 |

合格判準（bench `PASS_TOL=10`）：命中 ±10 內 **或** 被誠實標示為不可解 → 5/5。
詳見 `scripts/bench_eval_native.py` 與記憶 `eval-consistent-capture-quality-cap`。
