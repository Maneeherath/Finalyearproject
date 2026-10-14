# ============================================================
#  DESKTOP & MOBILE LAUNCHER
#  Desktop: starts the app server and opens it in its own native window
#           (pywebview / Microsoft Edge WebView2) - no browser tab needed.
#  Mobile:  --lan serves the app to phones on the same Wi-Fi over HTTPS
#           (phone browsers only allow the camera on secure pages), using a
#           self-signed certificate generated on first run.
#
#  Usage:   python desktop_app.py            # desktop window
#           python desktop_app.py --lan      # desktop window + phones on the same network
#           python desktop_app.py --browser  # open in the default browser instead
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================

import argparse
import datetime
import ipaddress
import os
import socket
import ssl
import subprocess
import sys
import time
import urllib.request
import webbrowser

BASE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(BASE, 'app_v5.py')
CERT_DIR = os.path.join(BASE, '.streamlit', 'certs')
TITLE = "Gaming Behavior & Cognitive Predictor"


def free_port(preferred=8501):
    for port in [preferred] + list(range(8502, 8600)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    raise SystemExit("No free port found between 8501 and 8599.")


def lan_ip():
    # Picks the address of the interface used for outbound traffic (no packet is actually sent)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(('10.255.255.255', 1))
            return s.getsockname()[0]
        except OSError:
            return '127.0.0.1'


def ensure_certificate(ip):
    """Self-signed certificate for localhost + this machine's LAN IP, regenerated if the IP changes."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    os.makedirs(CERT_DIR, exist_ok=True)
    cert_path, key_path = os.path.join(CERT_DIR, 'cert.pem'), os.path.join(CERT_DIR, 'key.pem')
    ip_path = os.path.join(CERT_DIR, 'ip.txt')
    if os.path.exists(cert_path) and os.path.exists(ip_path) and open(ip_path).read() == ip:
        return cert_path, key_path

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, TITLE)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName([
                x509.DNSName('localhost'),
                x509.IPAddress(ipaddress.ip_address('127.0.0.1')),
                x509.IPAddress(ipaddress.ip_address(ip)),
            ]), critical=False)
            .sign(key, hashes.SHA256()))
    with open(key_path, 'wb') as f:
        f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                  serialization.NoEncryption()))
    with open(cert_path, 'wb') as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(ip_path, 'w') as f:
        f.write(ip)
    return cert_path, key_path


def wait_until_ready(url, timeout=180):
    ctx = ssl._create_unverified_context()     # our own self-signed certificate
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + '/_stcore/health', timeout=2, context=ctx) as r:
                if r.status == 200:
                    return True
        except OSError:
            time.sleep(0.5)
    return False


def main():
    ap = argparse.ArgumentParser(description=f"Launch the {TITLE} as a desktop app.")
    ap.add_argument('--lan', action='store_true', help="also serve phones on the same Wi-Fi (HTTPS)")
    ap.add_argument('--browser', action='store_true', help="open in the default browser instead of a window")
    ap.add_argument('--port', type=int, default=8501)
    args = ap.parse_args()

    port = free_port(args.port)
    cmd = [sys.executable, '-m', 'streamlit', 'run', APP,
           '--server.headless', 'true', '--server.port', str(port),
           '--browser.gatherUsageStats', 'false']
    env = dict(os.environ)
    scheme = 'http'
    if args.lan:
        ip = lan_ip()
        cert, key = ensure_certificate(ip)
        cmd += ['--server.address', '0.0.0.0', '--server.sslCertFile', cert, '--server.sslKeyFile', key]
        scheme = 'https'
        env['APP_LAN_URL'] = f"https://{ip}:{port}"
    else:
        cmd += ['--server.address', '127.0.0.1']

    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    server = subprocess.Popen(cmd, cwd=BASE, env=env, creationflags=flags)
    local_url = f"{scheme}://localhost:{port}"
    try:
        print(f"Starting {TITLE} (first start loads the models and can take ~30 s)...")
        if not wait_until_ready(local_url):
            raise SystemExit("The app server did not start. Run `streamlit run app_v5.py` to see the error.")
        if args.lan:
            print(f"\nPhones on the same Wi-Fi: open {env['APP_LAN_URL']}")
            print("  The certificate is self-signed: tap 'Advanced' -> 'Proceed' once on each phone.\n")

        if args.browser:
            webbrowser.open(local_url)
            print(f"Opened {local_url} — press Ctrl+C here to stop the app.")
            server.wait()
        else:
            try:
                import webview
            except ImportError:
                webbrowser.open(local_url)
                print("pywebview is not installed, so the app opened in your browser. Press Ctrl+C to stop.")
                server.wait()
                return
            webview.settings['ALLOW_DOWNLOADS'] = True     # needed for the PDF report download button
            if args.lan:
                # This window only ever loads our own localhost server, whose certificate is self-signed
                os.environ['WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS'] = (
                    os.environ.get('WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS', '') + ' --ignore-certificate-errors').strip()
            webview.create_window(TITLE, local_url, width=1400, height=900, min_size=(420, 600))
            webview.start()       # returns when the window is closed
    except KeyboardInterrupt:
        pass
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == '__main__':
    main()
