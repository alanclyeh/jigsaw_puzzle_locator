#!/usr/bin/env python3
"""啟動拼圖單片採集 Web App。

預設綁 0.0.0.0:8000 並啟用自簽 TLS，方便手機同網段以 https 連線
（行動裝置上非 localhost 必須為 secure context 才能開鏡頭）。

用法：
    python scripts/run_capture_server.py            # https + 自簽憑證（手機用）
    python scripts/run_capture_server.py --no-tls   # http（桌機 localhost 測試）
    python scripts/run_capture_server.py --port 9000
"""
import argparse
import socket
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CERT_DIR = PROJECT_ROOT / ".certs"


def local_ip() -> str:
    """取得對外網卡 IP（用於提示手機連線網址）。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def ensure_cert() -> tuple[Path, Path]:
    """確保自簽憑證存在（重用既有）。回傳 (cert, key)。

    憑證檔名綁定當下 IP；若機器 IP（DHCP）變動會產生新檔並重新簽發，
    避免重用到 SAN 不符的舊憑證導致手機端 TLS 失敗。
    """
    ip = local_ip()
    suffix = ip.replace(".", "_")
    cert_path = CERT_DIR / f"dev_cert_{suffix}.pem"
    key_path = CERT_DIR / f"dev_key_{suffix}.pem"
    if cert_path.exists() and key_path.exists():
        return cert_path, key_path

    CERT_DIR.mkdir(exist_ok=True)
    try:
        import datetime

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        sys.exit(
            "缺少 cryptography 套件，無法產生自簽憑證。\n"
            "請執行 `pip install cryptography`，或改用 --no-tls，"
            "或手動以 openssl 產生 .certs/dev_cert.pem 與 .certs/dev_key.pem。"
        )

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "jp-locator-dev")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=825))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("localhost"), x509.IPAddress(__import__("ipaddress").ip_address(ip))]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    print(f"已產生自簽憑證於 {CERT_DIR}")
    return cert_path, key_path


def main():
    parser = argparse.ArgumentParser(description="拼圖單片採集 Web App 啟動器")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--no-tls", action="store_true", help="走 http（桌機 localhost 測試）")
    args = parser.parse_args()

    import uvicorn

    ip = local_ip()
    if args.no_tls:
        print(f"\n→ 桌機開啟：http://localhost:{args.port}\n")
        uvicorn.run("source.capture.app:app", host=args.host, port=args.port)
    else:
        cert_path, key_path = ensure_cert()
        print(f"\n→ 手機開啟：https://{ip}:{args.port}")
        print("  (首次連線需在手機上手動信任此自簽憑證)\n")
        uvicorn.run(
            "source.capture.app:app",
            host=args.host,
            port=args.port,
            ssl_certfile=str(cert_path),
            ssl_keyfile=str(key_path),
        )


if __name__ == "__main__":
    main()
