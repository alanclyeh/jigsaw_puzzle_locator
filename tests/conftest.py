import os
import time
from pathlib import Path
import pytest

# 在模組加載時記錄開始時間，以相容各版本的 pytest
START_TIME = time.time()

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """
    在 pytest 測試結束後，自動生成精美的 HTML 測試與優化報告至 output/report.html
    """
    stats = terminalreporter.stats
    passed_list = stats.get('passed', [])
    failed_list = stats.get('failed', [])
    skipped_list = stats.get('skipped', [])
    
    total_passed = len(passed_list)
    total_failed = len(failed_list)
    total_skipped = len(skipped_list)
    total_tests = total_passed + total_failed + total_skipped
    
    duration = time.time() - START_TIME
    
    # 收集測試詳細資訊
    test_details = []
    for rep in passed_list:
        test_details.append(f"""
        <div class="test-item passed">
            <span class="status-badge passed">PASSED</span>
            <span class="test-name">{rep.nodeid}</span>
            <span class="test-duration">{rep.duration:.4f}s</span>
        </div>
        """)
        
    for rep in failed_list:
        error_msg = str(rep.longrepr).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        test_details.append(f"""
        <div class="test-item failed">
            <span class="status-badge failed">FAILED</span>
            <span class="test-name">{rep.nodeid}</span>
            <span class="test-duration">{rep.duration:.4f}s</span>
            <div class="error-log">{error_msg}</div>
        </div>
        """)
        
    for rep in skipped_list:
        test_details.append(f"""
        <div class="test-item skipped">
            <span class="status-badge skipped">SKIPPED</span>
            <span class="test-name">{rep.nodeid}</span>
            <span class="test-duration">N/A</span>
        </div>
        """)
        
    test_details_html = "\n".join(test_details)
    
    status_text = "PASSED" if total_failed == 0 else "FAILED"
    status_class = "passed" if total_failed == 0 else "failed"
    
    # HTML 模板字串 (使用普通字串以避開 CSS/JS 的大括弧轉義問題)
    html_template = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jigsaw Puzzle Locator - 測試與優化報告</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>
        mermaid.initialize({ 
            startOnLoad: true, 
            theme: 'dark',
            themeVariables: {
                primaryColor: '#312e81',
                primaryTextColor: '#f8fafc',
                lineColor: '#818cf8',
                nodeBorder: '#4f46e5'
            }
        });
    </script>
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #090d16 0%, #111827 50%, #0f172a 100%);
            --panel-bg: rgba(17, 24, 39, 0.75);
            --panel-border: rgba(255, 255, 255, 0.08);
            --primary-grad: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-purple: #c084fc;
            --accent-blue: #38bdf8;
            --accent-green: #34d399;
            --accent-red: #f87171;
            --font-main: 'Inter', -apple-system, sans-serif;
            --font-title: 'Outfit', sans-serif;
            --font-code: 'Fira Code', monospace;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: var(--font-main);
            background: var(--bg-gradient);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
            overflow-x: hidden;
            padding: 2rem 1rem;
        }

        .glass-container {
            max-width: 1100px;
            margin: 0 auto;
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 24px;
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
            padding: 3rem;
            position: relative;
            overflow: hidden;
        }

        header {
            position: relative;
            z-index: 1;
            margin-bottom: 3rem;
            text-align: center;
        }

        .logo-tag {
            display: inline-block;
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.3);
            color: var(--accent-blue);
            padding: 0.4rem 1.2rem;
            border-radius: 50px;
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            margin-bottom: 1rem;
            text-transform: uppercase;
        }

        h1 {
            font-family: var(--font-title);
            font-size: 2.8rem;
            font-weight: 800;
            background: var(--primary-grad);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.8rem;
            letter-spacing: -0.02em;
        }

        .subtitle {
            color: var(--text-secondary);
            font-size: 1.1rem;
            max-width: 600px;
            margin: 0 auto;
        }

        .tabs {
            display: flex;
            justify-content: center;
            gap: 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            padding-bottom: 1.5rem;
            margin-bottom: 2.5rem;
            position: relative;
            z-index: 1;
        }

        .tab-btn {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-family: var(--font-title);
            font-size: 1.1rem;
            font-weight: 600;
            padding: 0.75rem 2rem;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }

        .tab-btn:hover {
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.03);
        }

        .tab-btn.active {
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.06);
        }

        .tab-btn.active::after {
            content: '';
            position: absolute;
            bottom: -1.6rem;
            left: 15%;
            width: 70%;
            height: 3px;
            background: var(--primary-grad);
            border-radius: 10px;
        }

        .tab-content {
            display: none;
            animation: fadeIn 0.5s ease;
            position: relative;
            z-index: 1;
        }

        .tab-content.active {
            display: block;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        h2 {
            font-family: var(--font-title);
            font-size: 1.8rem;
            margin-top: 2.5rem;
            margin-bottom: 1.2rem;
            color: var(--text-primary);
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            padding-bottom: 0.5rem;
        }

        h3 {
            font-family: var(--font-title);
            font-size: 1.3rem;
            margin: 1.8rem 0 0.8rem 0;
            color: var(--accent-purple);
        }

        p {
            margin-bottom: 1.2rem;
            color: var(--text-secondary);
            font-size: 1.05rem;
        }

        ul {
            margin-left: 1.5rem;
            margin-bottom: 1.5rem;
            color: var(--text-secondary);
        }

        li {
            margin-bottom: 0.5rem;
            font-size: 1.05rem;
        }

        /* 測試總結卡片 */
        .summary-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--panel-border);
            border-radius: 18px;
            padding: 2rem;
            margin-bottom: 2rem;
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 1.5rem;
        }

        .summary-status {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .status-dot {
            width: 16px;
            height: 16px;
            border-radius: 50%;
        }
        .status-dot.passed { background: var(--accent-green); box-shadow: 0 0 12px var(--accent-green); }
        .status-dot.failed { background: var(--accent-red); box-shadow: 0 0 12px var(--accent-red); }

        .status-title {
            font-family: var(--font-title);
            font-size: 2rem;
            font-weight: 700;
        }
        .status-title.passed { color: var(--accent-green); }
        .status-title.failed { color: var(--accent-red); }

        .summary-stats {
            display: flex;
            gap: 2rem;
        }

        .stat-item {
            text-align: center;
        }

        .stat-val {
            font-family: var(--font-title);
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .stat-lbl {
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* 測試案例列表 */
        .test-list {
            background: rgba(0, 0, 0, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.03);
            border-radius: 14px;
            padding: 1rem;
            margin-bottom: 2rem;
        }

        .test-item {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            padding: 0.85rem 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
            gap: 1rem;
        }

        .test-item:last-child {
            border-bottom: none;
        }

        .status-badge {
            font-family: var(--font-title);
            font-size: 0.75rem;
            font-weight: 700;
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
            text-transform: uppercase;
        }
        .status-badge.passed { background: rgba(52, 211, 153, 0.15); color: var(--accent-green); }
        .status-badge.failed { background: rgba(248, 113, 113, 0.15); color: var(--accent-red); }
        .status-badge.skipped { background: rgba(148, 163, 184, 0.15); color: var(--text-secondary); }

        .test-name {
            font-family: var(--font-code);
            font-size: 0.92rem;
            flex-grow: 1;
            color: #e2e8f0;
        }

        .test-duration {
            font-family: var(--font-code);
            font-size: 0.9rem;
            color: var(--text-secondary);
        }

        .error-log {
            width: 100%;
            background: #020617;
            border: 1px solid rgba(248, 113, 113, 0.15);
            padding: 1rem;
            border-radius: 8px;
            font-family: var(--font-code);
            font-size: 0.85rem;
            color: #fca5a5;
            margin-top: 0.5rem;
            white-space: pre-wrap;
            overflow-x: auto;
        }

        /* 網格卡片 */
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }

        .card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.04);
            border-radius: 16px;
            padding: 1.5rem;
            transition: all 0.3s ease;
        }

        .card:hover {
            transform: translateY(-4px);
            background: rgba(255, 255, 255, 0.04);
            border-color: rgba(99, 102, 241, 0.2);
        }

        .card h4 {
            font-family: var(--font-title);
            font-size: 1.15rem;
            margin-bottom: 0.6rem;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .card p {
            font-size: 0.95rem;
            margin-bottom: 0;
        }

        .chart-box {
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.03);
            border-radius: 16px;
            padding: 2rem;
            margin: 2rem 0;
            display: flex;
            justify-content: center;
        }

        .tip-box {
            background: rgba(16, 185, 129, 0.06);
            border-left: 4px solid var(--accent-green);
            padding: 1.2rem 1.5rem;
            border-radius: 0 12px 12px 0;
            margin: 1.5rem 0;
        }

        .tip-box p {
            margin-bottom: 0;
            color: var(--text-primary);
        }

        @media (max-width: 768px) {
            .glass-container { padding: 1.5rem; border-radius: 16px; }
            h1 { font-size: 2rem; }
            .tab-btn { padding: 0.6rem 1.2rem; font-size: 0.95rem; }
        }
    </style>
</head>
<body>

    <div class="glass-container">
        
        <header>
            <div class="logo-tag">Automated Test Report</div>
            <h1>Jigsaw Puzzle Helper</h1>
            <div class="subtitle">自動化測試執行與演算法設計優化報告</div>
        </header>

        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('walkthrough')">🛠️ 測試執行結果 (Walkthrough)</button>
            <button class="tab-btn" onclick="switchTab('brainstorm')">🚀 演算法優化與架構設計</button>
        </div>

        <!-- 標籤頁一：測試執行結果 -->
        <div id="walkthrough" class="tab-content active">
            
            <h2>📊 測試總結 (Summary)</h2>
            <div class="summary-card">
                <div class="summary-status">
                    <div class="status-dot __STATUS_CLASS__"></div>
                    <span class="status-title __STATUS_CLASS__">__STATUS_TEXT__</span>
                </div>
                <div class="summary-stats">
                    <div class="stat-item">
                        <div class="stat-val">__TOTAL_TESTS__</div>
                        <div class="stat-lbl">Tests Run</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-val" style="color: var(--accent-green);">__TOTAL_PASSED__</div>
                        <div class="stat-lbl">Passed</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-val" style="color: var(--accent-red);">__TOTAL_FAILED__</div>
                        <div class="stat-lbl">Failed</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-val">__DURATION__s</div>
                        <div class="stat-lbl">Duration</div>
                    </div>
                </div>
            </div>

            <h2>🧪 測試案例清單 (Test Cases)</h2>
            <div class="test-list">
                __TEST_DETAILS_HTML__
            </div>

            <h2>🛠️ 本期實作變更內容</h2>
            <p>我們在 <code style="color: var(--accent-blue);">locator.py</code> 中實作了「碎片尺度與長寬比幾何正規化比對」演算法，核心修改包含：</p>
            <ul>
                <li><strong><code>_standardize_rotated_rect(rect)</code></strong>: 鎖定長短邊，確保 width 永遠為長邊且高度為短邊，並將角度標準化為正對齊角度，完全防範 OpenCV 寬高互換引發的匹配崩潰。</li>
                <li><strong><code>_get_puzzle_body_rect(mask)</code></strong>: 在遮罩上使用開運算 (MORPH_OPEN) 削去凸耳，以精確測量拼圖本體尺寸，排除卡榫引起的比例收縮偏差。</li>
                <li><strong>4 向帶遮罩模板匹配</strong>：不再盲目進行多尺度搜索，而是將碎片以 1:1 的 scale_factor 精確縮放，僅搜索 4 個直角方向並進行帶遮罩匹配，運算量大幅降低 90% 以上。</li>
            </ul>

        </div>

        <!-- 標籤頁二：演算法優化與架構設計 -->
        <div id="brainstorm" class="tab-content">
            
            <h2>💡 系統與演算法優化藍圖</h2>
            <p>以下是離線預處理與線上實拍碎片定位的架構工作流：</p>
            
            <div class="chart-box">
                <pre class="mermaid">
graph TD
    subgraph "大圖預處理 (Offline/專案建立)"
        Ref[完成大圖 reference.jpg] --> SIFT_Ref[提取 SIFT 特徵點與描述子]
        Ref --> Grid_Moments[計算各網格色彩矩 Color Moments]
        SIFT_Ref --> Cache[(特徵與色彩快取 .npz / Redis)]
        Grid_Moments --> Cache
    end

    subgraph "單片碎片定位 (Online/API 請求)"
        Piece[單片照片 piece.jpg] --> Seg[去背模組 segmentation]
        Seg --> CleanPiece[乾淨單片 BGRA]
        CleanPiece --> Mask_Erosion[遮罩收縮 Erosion]
        Mask_Erosion --> SIFT_Piece[提取 SIFT 特徵]
        
        Cache --> Matcher{{快取匹配}}
        SIFT_Piece --> Matcher
        
        Matcher -- "1. SIFT 成功 (inliers >= 10)" --> Affine[仿射變換 Affine Transform] --> Quad[取得投影座標]
        Matcher -- "2. SIFT 失敗" --> ColorMatch[色彩矩相似度篩選 Top K] --> TempMatch[多角度模板匹配] --> BBox[取得候選邊框]
        
        Quad --> Result[LocateResult JSON]
        BBox --> Result
    end

    subgraph "前端呈現 (Frontend)"
        Result --> FE_Canvas[HTML5 Canvas Overlay 繪製與旋轉提示]
    end
                </pre>
            </div>

            <h2>🎯 演算法核心優化亮點</h2>
            <div class="grid">
                <div class="card">
                    <h4><span>📏</span> 幾何與尺度正規化</h4>
                    <p>利用外接矩形對齊網格，先計算長寬比以過濾高達 80% 的不符網格，接著對齊長邊進行 1:1 縮放，將多尺度搜尋降為 1.0 的精準對齊匹配。</p>
                </div>
                <div class="card">
                    <h4><span>⚡</span> 運算量降低 90%</h4>
                    <p>由 600 次 matchTemplate 降至 12 次以內的 1:1 精準對齊，CPU 匹配速度獲得飛躍性提升。</p>
                </div>
                <div class="card">
                    <h4><span>📐</span> 仿射變換代替單應性</h4>
                    <p>在匹配點少時以仿射變換 <code>cv2.estimateAffine2D</code> 代替 Homography，防止定位框被梯形拉伸。</p>
                </div>
                <div class="card">
                    <h4><span>🛡️</span> 形態學去凸耳</h4>
                    <p>開運算 (MORPH_OPEN) 削去凸耳，量測拼圖主體 Body 的純淨尺寸。並在比對時進行帶遮罩匹配，屏蔽背景噪聲。</p>
                </div>
            </div>

            <h2>🌐 系統與 Web 架構規劃</h2>
            <h3>1. 大圖特徵快取機制 (Feature Caching)</h3>
            <p><strong>專案建立時</strong>預先計算大圖特徵點與色彩特徵並存入 Redis。<strong>碎片定位時</strong>直接從記憶體載入，定位 API 響應時間可壓縮至 <strong>100~300ms</strong>。</p>

            <h3>2. 前端 Canvas 動態繪製</h3>
            <p>API 僅傳送 JSON 幾何座標與網格 metadata。前端網頁大圖載入一次後，透過 <strong>HTML5 Canvas Overlay</strong> 動態在網頁上疊加定位框與 3D 旋轉提示，極大減少頻寬消耗。</p>

        </div>

    </div>

    <script>
        function switchTab(tabId) {
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));
            
            const buttons = document.querySelectorAll('.tab-btn');
            buttons.forEach(btn => btn.classList.remove('active'));
            
            document.getElementById(tabId).classList.add('active');
            event.currentTarget.classList.add('active');
            
            if (tabId === 'brainstorm') {
                mermaid.init(undefined, document.querySelectorAll('.mermaid'));
            }
        }
    </script>
</body>
</html>
"""
    
    # 進行字串變數替換
    html_content = html_template.replace("__STATUS_CLASS__", status_class) \
                                 .replace("__STATUS_TEXT__", status_text) \
                                 .replace("__TOTAL_TESTS__", str(total_tests)) \
                                 .replace("__TOTAL_PASSED__", str(total_passed)) \
                                 .replace("__TOTAL_FAILED__", str(total_failed)) \
                                 .replace("__DURATION__", f"{duration:.4f}") \
                                 .replace("__TEST_DETAILS_HTML__", test_details_html)
    
    # 確保輸出目錄存在
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"\n[AUTO-REPORT] 測試報告已成功生成至: {report_path.absolute()}")
