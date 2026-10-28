#!/usr/bin/env python3
"""Gradio demo for the baseline.

The side panel showing which branch fired and why is the point: routing errors
are otherwise invisible, and most surprising results turn out to be routing
errors rather than ranking errors.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import threading  # noqa: E402

import yaml  # noqa: E402

from pixarch.clip_model import ClipEncoder  # noqa: E402
from pixarch.query.engine import SearchEngine  # noqa: E402
from pixarch.store import sqlite_store as store  # noqa: E402

CFG = yaml.safe_load((ROOT / "configs/baseline.yaml").read_text())
RAW_DIR = ROOT / CFG["paths"]["raw_dir"]

# Gradio dispatches handlers on worker threads while this connection is opened
# on the main thread, so the default thread check would reject every query.
# The demo is read-only, and _QUERY_LOCK serialises access.
_conn = store.connect(ROOT / CFG["paths"]["index_db"], create=False, check_same_thread=False)
_QUERY_LOCK = threading.Lock()
_encoder = ClipEncoder(
    model_name=CFG["clip"]["model"], pretrained=CFG["clip"]["pretrained"],
    device=CFG["clip"]["device"], batch_size=CFG["clip"]["batch_size"],
)
_engine = SearchEngine(_conn, _encoder, CFG)


def run_query(query: str, top_k: int, force: str):
    if not query.strip():
        return [], "Enter a query."

    branch = None if force == "auto (router)" else force

    with _QUERY_LOCK:
        result = _engine.search(query, top_k=int(top_k), force_branch=branch)

        gallery, detail = [], []
        for rank, hit in enumerate(result.hits, start=1):
            row = _conn.execute(
                "SELECT path, gold_bucket, gold_subtype FROM items WHERE item_id=?", (hit.item_id,)
            ).fetchone()
            if not row:
                continue
            gallery.append((str(RAW_DIR / row["path"]), f"{rank}. {hit.item_id} ({hit.score:.3f})"))
            line = f"**{rank}. `{hit.item_id}`** score {hit.score:.3f} - gold `{row['gold_bucket']}/{row['gold_subtype']}`"
            if hit.snippet:
                line += f"\n> {hit.snippet[:220]}"
            detail.append(line)

        ocr_note = ""
        if result.route.branch == "document" and result.hits:
            top = _conn.execute(
                "SELECT doc_type, doc_type_rule, substr(full_text,1,300) t FROM doc_fields WHERE item_id=?",
                (result.hits[0].item_id,),
            ).fetchone()
            if top:
                ocr_note = (f"\n\n**Top hit OCR** - doc_type `{top['doc_type']}` "
                            f"(rule `{top['doc_type_rule'] or '-'}`)\n\n> {top['t']}")

    info = (
        f"### Branch: `{result.route.branch}`\n"
        f"- confidence **{result.route.confidence:.3f}**\n"
        f"- reason `{result.route.reason}`\n"
        f"- latency **{result.elapsed_ms:.1f} ms**\n"
        f"- hits **{len(result.hits)}**\n\n---\n\n" + "\n\n".join(detail) + ocr_note
    )
    return gallery, info


def build_ui():
    import gradio as gr

    n_items = _conn.execute("SELECT COUNT(*) c FROM items").fetchone()["c"]
    with gr.Blocks(title="Pixarch baseline") as demo:
        gr.Markdown(
            f"# Pixarch - baseline search\n"
            f"{n_items} items | CLIP `{_encoder.id}` on `{_encoder.device}` | "
            f"single-shot router, no fusion, no agentic loop"
        )
        with gr.Row():
            box = gr.Textbox(label="Query", scale=4,
                             placeholder="give me my aadhaar card / find a photo of a beach")
            k = gr.Slider(1, 20, value=CFG["search"]["top_k"], step=1, label="top-k")
            force = gr.Dropdown(["auto (router)", "document", "scene"],
                                value="auto (router)", label="branch")
        btn = gr.Button("Search", variant="primary")
        with gr.Row():
            gallery = gr.Gallery(label="Results", columns=5, height=380, scale=3)
            info = gr.Markdown(scale=2)
        gr.Examples(
            [["give me my aadhaar card"], ["find a photo of a beach"],
             ["tax invoice with gst number"], ["gym workout"]], inputs=box,
        )
        btn.click(run_query, [box, k, force], [gallery, info])
        box.submit(run_query, [box, k, force], [gallery, info])
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", share=False)
