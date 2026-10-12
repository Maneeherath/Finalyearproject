# ============================================================
#  Creates Windows desktop shortcuts that open the app in its own window
#  (via desktop_app.py), with no terminal window.
#  Usage:  python install_desktop_shortcut.py [--dest FOLDER]
#  Student: P R M K Herath | D/BCS/23/0009
# ============================================================

import argparse
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))


def make_shortcut(path, target, arguments, workdir, description):
    ps = ("$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:LNK_PATH); "
          "$s.TargetPath = $env:LNK_TARGET; $s.Arguments = $env:LNK_ARGS; "
          "$s.WorkingDirectory = $env:LNK_DIR; $s.Description = $env:LNK_DESC; $s.Save()")
    env = dict(os.environ, LNK_PATH=path, LNK_TARGET=target, LNK_ARGS=arguments, LNK_DIR=workdir, LNK_DESC=description)
    subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', ps], env=env, check=True)


def main():
    if os.name != 'nt':
        raise SystemExit("Desktop shortcuts are created for Windows only. Run `python desktop_app.py` instead.")
    ap = argparse.ArgumentParser(description="Create desktop shortcuts for the app.")
    ap.add_argument('--dest', default=os.path.join(os.environ['USERPROFILE'], 'Desktop'))
    args = ap.parse_args()

    # pythonw.exe runs without a console window
    pythonw = os.path.join(os.path.dirname(sys.executable), 'pythonw.exe')
    target = pythonw if os.path.exists(pythonw) else sys.executable
    launcher = os.path.join(BASE, 'desktop_app.py')
    for name, extra, desc in [
        ("Gaming Cognitive Predictor", "", "Open the app in its own window"),
        ("Gaming Cognitive Predictor (phone access)", " --lan", "Open the app and allow phones on the same Wi-Fi"),
    ]:
        path = os.path.join(args.dest, name + '.lnk')
        make_shortcut(path, target, f'"{launcher}"{extra}', BASE, desc)
        print(f"Created {path}")


if __name__ == '__main__':
    main()
