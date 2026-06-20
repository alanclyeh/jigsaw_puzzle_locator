#!/usr/bin/env python3
"""啟動拼圖定位助手 Web App（正式版）。

預設綁 0.0.0.0:8000 並啟用自簽 TLS，方便手機同網段以 https 連線
（行動裝置上非 localhost 必須為 secure context 才能開鏡頭）。

用法：
    python scripts/run_webapp_server.py            # https + 自簽憑證（手機用）
    python scripts/run_webapp_server.py --no-tls   # http（桌機 localhost 測試）
    python scripts/run_webapp_server.py --port 9000

資料存於 data/webapp/（SQLite jp.db + 各專案影像）。
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 重用採集伺服器的自簽憑證邏輯（同一份 .certs）
from scripts.run_capture_server import ensure_cert, local_ip  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="拼圖定位助手 Web App 啟動器")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--no-tls", action="store_true", help="走 http（桌機 localhost 測試）")
    parser.add_argument("--reload", action="store_true", help="開發熱重載")
    args = parser.parse_args()

    import uvicorn

    ip = local_ip()
    if args.no_tls:
        print(f"\n→ 桌機開啟：http://localhost:{args.port}\n")
        uvicorn.run("source.webapp.app:app", host=args.host, port=args.port, reload=args.reload)
    else:
        cert_path, key_path = ensure_cert()
        print(f"\n→ 手機開啟：https://{ip}:{args.port}")
        print("  (首次連線需在手機上手動信任此自簽憑證)\n")
        uvicorn.run(
            "source.webapp.app:app",
            host=args.host,
            port=args.port,
            ssl_certfile=str(cert_path),
            ssl_keyfile=str(key_path),
            reload=args.reload,
        )


if __name__ == "__main__":
    main()
