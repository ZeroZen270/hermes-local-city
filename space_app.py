"""Hermes Local city - Hugging Face Space bootstrap (doorway edition).

This is the Space's app.py, kept in the repo for the record (the
Space itself holds the live copy; update it through the Space's
Files page, then restart).

Doorway design (Aaron's rules, 2026-11-25):
  * The Space runs NO world of its own. The city lives only on
    Aaron's tablet, inside the Hermes app shell, thinking with the
    app's own Nous/LongCat sign-in - no hosted brain, no quotas.
  * The tablet dials OUT to this Space (doorway.py, stdlib-only)
    and the Space relays /world/* and /api/* through that one
    WebSocket to the tablet's bridge.
  * Tablet down = world paused: /world serves a paused page and
    nothing ticks. One world, one writer, no divergence.

So this file only: downloads the city source fresh from the public
GitHub repo (the doorway server module lives there), builds a small
Gradio status shell (ZeroGPU still expects a launch() and at least
one @spaces.GPU function), and registers the doorway routes ahead
of Gradio's own.
"""
import io
import os
import pathlib
import shutil
import sys
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent
CODE = pathlib.Path(os.environ.get("HERMES_CITY_CODE", ROOT / "city-code"))
ZIP_URL = ("https://codeload.github.com/ZeroZen270/"
           "hermes-local-city/zip/refs/heads/main")

if not os.environ.get("HERMES_CITY_SKIP_DOWNLOAD"):
    print("fetching city code...", flush=True)
    data = urllib.request.urlopen(ZIP_URL, timeout=300).read()
    tmp = ROOT / "_zip"
    if tmp.exists():
        shutil.rmtree(tmp)
    zipfile.ZipFile(io.BytesIO(data)).extractall(tmp)
    if CODE.exists():
        shutil.rmtree(CODE)
    next(tmp.iterdir()).rename(CODE)

os.chdir(CODE)
sys.path.insert(0, str(CODE))

import gradio as gr
import spaces

from hermes_local.doorway_server import Doorway

# The doorway authenticates the tablet with the same token the
# tablet already uses for world sync (a Space secret Aaron set).
DOORWAY = Doorway(os.environ.get("HERMES_WORLD_TOKEN", ""))


@spaces.GPU(duration=10)
def doorway_check():
    """ZeroGPU slot check: reports the live doorway state."""
    return DOORWAY.status_text()


_CSS = (".gradio-container{max-width:100%!important;padding:0!important}"
        "footer{display:none!important}")
with gr.Blocks(title="Hermes Local City") as demo:
    gr.HTML('<iframe src="/world/" title="Hermes Local City" '
            'style="width:100%;height:92vh;border:0" '
            'allow="fullscreen"></iframe>')
    with gr.Row():
        _btn = gr.Button("Doorway check (ZeroGPU slot)", size="sm")
        _out = gr.Textbox(label="Doorway status", interactive=False)
    _btn.click(doorway_check, outputs=_out)

demo.launch(server_name="0.0.0.0", server_port=7860,
            prevent_thread_lock=True, css=_CSS, ssr_mode=False)

# Doorway routes go FIRST on Gradio's own FastAPI app so the city's
# paths win; Gradio's own routes (UI, /gradio_api, its /assets)
# are untouched.
for _route in reversed(DOORWAY.routes()):
    demo.app.router.routes.insert(0, _route)
print("Doorway routes registered on the Gradio app.", flush=True)

demo.block_thread()
