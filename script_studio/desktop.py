"""Desktop launcher: runs Script Studio on this computer and opens it in the browser.

Used as the entry point of the packaged Windows / macOS app. Also works from source:
    python -m script_studio.desktop
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser

PREFERRED_PORT = 8765


def _prepare_environment() -> str:
    from script_studio import userconfig
    home = userconfig.data_dir()
    # A windowed app has no console: send output to a log file.
    log = open(os.path.join(home, 'log.txt'), 'a', encoding='utf-8', buffering=1)
    if sys.stdout is None or userconfig.is_frozen():
        sys.stdout = log
    if sys.stderr is None or userconfig.is_frozen():
        sys.stderr = log
    os.environ.setdefault('NUMBA_CACHE_DIR', os.path.join(home, 'cache'))
    os.environ.setdefault('HF_HOME', os.path.join(home, 'models'))
    os.environ.setdefault('WHISPER_MODEL', 'base')
    os.environ.setdefault('SCRIPT_JOBS_DIR', os.path.join(home, 'scripts'))
    return home


def _ours(port: int) -> bool:
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/health', timeout=1.5) as r:
            return b'script-studio' in r.read()
    except Exception:
        return False


def _free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False


def _pick_port() -> int:
    if _free(PREFERRED_PORT):
        return PREFERRED_PORT
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _control_window(url: str, stop) -> None:
    """A small window so people can see the app is running, reopen it, and quit."""
    try:
        import tkinter as tk
    except Exception:
        while True:          # no GUI toolkit: keep serving until the process is ended
            time.sleep(3600)
    root = tk.Tk()
    root.title('Script Studio')
    root.geometry('360x170')
    root.resizable(False, False)
    root.configure(bg='#181b22')
    tk.Label(root, text='▶  Script Studio is running', fg='#d6a440', bg='#181b22', font=('Helvetica', 15, 'bold')).pack(pady=(18, 4))
    tk.Label(root, text='It opens in your web browser. Keep this window\nopen while you work; close it to quit.',
             fg='#c9ccd4', bg='#181b22', font=('Helvetica', 10)).pack()
    row = tk.Frame(root, bg='#181b22')
    row.pack(pady=14)

    def quit_app():
        stop()
        root.destroy()
        os._exit(0)
    tk.Button(row, text='Open Script Studio', width=16, command=lambda: webbrowser.open(url)).pack(side='left', padx=6)
    tk.Button(row, text='Quit', width=8, command=quit_app).pack(side='left', padx=6)
    root.protocol('WM_DELETE_WINDOW', quit_app)
    root.mainloop()


def selftest(out_path: str) -> int:
    """Exercise every bundled component (used by the build pipeline on the packaged app)."""
    import io
    import tempfile
    import traceback
    results, ok = [], True

    def check(name, fn):
        nonlocal ok
        try:
            fn()
            results.append(f'PASS {name}')
        except Exception:
            ok = False
            results.append(f'FAIL {name}\n{traceback.format_exc()}')

    def server():
        import uvicorn
        from script_studio.app import app
        port = _pick_port()
        srv = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_config=None))
        threading.Thread(target=srv.run, daemon=True).start()
        for _ in range(100):
            if _ours(port):
                break
            time.sleep(0.1)
        assert _ours(port), 'server did not start'
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/', timeout=5) as r:
            assert b'Script Studio' in r.read()
        srv.should_exit = True

    def documents():
        import docx
        from reportlab.platypus import Paragraph, SimpleDocTemplate
        from reportlab.lib.styles import getSampleStyleSheet
        from script_studio.ingest import load_document, from_html
        d = docx.Document(); d.add_paragraph('Hello from Word.'); b = io.BytesIO(); d.save(b)
        assert 'Word' in load_document('a.docx', b.getvalue()).text
        b = io.BytesIO(); SimpleDocTemplate(b).build([Paragraph('Hello from PDF.', getSampleStyleSheet()['Normal'])])
        assert 'PDF' in load_document('a.pdf', b.getvalue()).text
        assert 'story' in from_html('<html><body><article><p>' + 'A long story. ' * 40 + '</p></article></body></html>')[1]

    def audio():
        import numpy as np
        import soundfile as sf
        from script_studio.audio import analyze
        sr = 22050
        t = np.arange(sr * 30) / sr
        y = 0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 2 * t) > 0)
        p = os.path.join(tempfile.mkdtemp(), 't.wav'); sf.write(p, y.astype('float32'), sr)
        a = analyze(p)
        assert 29 < a.duration < 31 and a.sections

    def transcription():
        import ctranslate2  # noqa: F401
        import faster_whisper  # noqa: F401
        from faster_whisper.audio import decode_audio  # noqa: F401
        if os.getenv('SELFTEST_WHISPER') == '1':   # real run: downloads the model and transcribes on the CPU
            import numpy as np
            import soundfile as sf
            from script_studio import transcribe
            sr = 16000
            p = os.path.join(tempfile.mkdtemp(), 's.wav')
            sf.write(p, (0.01 * np.random.randn(sr * 3)).astype('float32'), sr)
            transcribe._faster_whisper(p)
            assert transcribe._FW_DEVICE == 'cpu'

    def export():
        from script_studio import exporters  # noqa: F401
        from reportlab.pdfgen import canvas
        canvas.Canvas(io.BytesIO()).save()

    def claude_sdk():
        import anthropic  # noqa: F401
        from script_studio.models import SegmentsOut
        from pydantic import TypeAdapter
        from anthropic.lib._parse._transform import transform_schema
        transform_schema(TypeAdapter(SegmentsOut).json_schema())

    for name, fn in (('server', server), ('documents', documents), ('audio', audio), ('transcription', transcription),
                     ('export', export), ('claude_sdk', claude_sdk)):
        check(name, fn)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results) + ('\nALL PASSED\n' if ok else '\nSOME FAILED\n'))
    return 0 if ok else 1


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == '--selftest':
        _prepare_environment()
        os._exit(selftest(sys.argv[2]))
    home = _prepare_environment()
    if _ours(PREFERRED_PORT):                 # already running: just show it
        webbrowser.open(f'http://127.0.0.1:{PREFERRED_PORT}/')
        return
    port = _pick_port()
    url = f'http://127.0.0.1:{port}/'

    import uvicorn
    from script_studio.app import app
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_config=None, access_log=False))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(150):
        if _ours(port):
            break
        time.sleep(0.1)
    print(f'Script Studio running at {url} (data folder: {home})', flush=True)
    webbrowser.open(url)

    def stop():
        server.should_exit = True
    _control_window(url, stop)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
