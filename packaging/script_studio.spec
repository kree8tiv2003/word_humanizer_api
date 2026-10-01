# PyInstaller spec for the Script Studio desktop app (Windows and macOS).
# Build from the repository root:  pyinstaller packaging/script_studio.spec --noconfirm
import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, '..'))
datas = [(os.path.join(ROOT, 'script_studio', 'static'), 'script_studio/static'),
         (os.path.join(ROOT, 'script_studio', 'data'), 'script_studio/data')]
binaries = []
hiddenimports = collect_submodules('uvicorn') + collect_submodules('script_studio') + ['multipart', 'python_multipart']

# Packages that ship data files or load modules lazily. Everything else (numpy, scipy, sklearn, numba,
# onnxruntime, ctranslate2, av...) is handled by PyInstaller's own hooks, which bundle far less.
for pkg in ('librosa', 'lazy_loader', 'trafilatura', 'justext', 'courlan', 'htmldate', 'tld', 'faster_whisper',
            'docx', 'reportlab', 'certifi', 'soundfile', 'soxr', 'audioread'):
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception as e:
        print('skip', pkg, e)

a = Analysis([os.path.join(SPECPATH, 'launch.py')], pathex=[ROOT], binaries=binaries, datas=datas,
             hiddenimports=hiddenimports, excludes=['torch', 'tensorflow', 'matplotlib', 'PyQt5', 'PySide6', 'IPython', 'pytest', 'gi', 'onnxruntime.transformers', 'onnxruntime.quantization', 'onnxruntime.tools', 'sklearn.datasets', 'scipy.datasets'],
             noarchive=False)
pyz = PYZ(a.pure)
icon = os.path.join(SPECPATH, 'icon.ico' if sys.platform == 'win32' else 'icon.icns')
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='Script Studio', console=False,
          icon=icon if os.path.exists(icon) else None)
coll = COLLECT(exe, a.binaries, a.datas, name='Script Studio')
if sys.platform == 'darwin':
    app = BUNDLE(coll, name='Script Studio.app', icon=icon if os.path.exists(icon) else None,
                 bundle_identifier='com.scriptstudio.app',
                 info_plist={'CFBundleShortVersionString': '1.0.0', 'NSHighResolutionCapable': True})
