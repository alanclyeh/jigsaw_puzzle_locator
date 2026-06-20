#!/usr/bin/env python3
"""啟動 Puzzle Locator 本機應用（行動端 UI + 定位 API）。

預設綁 0.0.0.0:8001 並啟用自簽 TLS，方便手機同網段以 https 連線
（行動裝置非 localhost 必須為 secure context 才能開相機）。
TLS 憑證重用 scripts/run_capture_server.py 既有邏輯。

用法：
    python scripts/run_locator_server.py            # https + 自簽憑證（手機用）
    python scripts/run_locator_server.py --no-tls   # http（桌機 localhost 測試）
    python scripts/run_locator_server.py --port 9001
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 重用 capture server 的本機 IP 偵測與自簽憑證產生
from scripts.run_capture_server import ensure_cert, local_ip  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Puzzle Locator 本機應用啟動器")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--no-tls", action="store_true", help="走 http（桌機 localhost 測試）")
    args = parser.parse_args()

    import uvicorn

    ip = local_ip()
    if args.no_tls:
        print(f"\n→ 桌機開啟：http://localhost:{args.port}\n")
        uvicorn.run("source.locator_web.app:app", host=args.host, port=args.port)
    else:
        cert_path, key_path = ensure_cert()
        print(f"\n→ 手機開啟：https://{ip}:{args.port}")
        print("  (首次連線需在手機上手動信任此自簽憑證)\n")
        uvicorn.run(
            "source.locator_web.app:app",
            host=args.host,
            port=args.port,
            ssl_certfile=str(cert_path),
            ssl_keyfile=str(key_path),
        )


if __name__ == "__main__":
    main()
