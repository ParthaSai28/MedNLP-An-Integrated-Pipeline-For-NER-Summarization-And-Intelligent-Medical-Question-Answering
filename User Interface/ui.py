import os
import requests
import gradio as gr

# Config
BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8090")
ENDPOINTS = {
    "summarize": "/summarize",
    "ner": "/ner",
    "rag": "/ask"
}
TIMEOUT = 3600

# Helpers
def _headers():
    h = {"Content-Type": "application/json"}
    return h

def ner_pairs(text: str, entities: list[dict]):
    """
    Convert character spans to (chunk, label) pairs for HighlightedText.
    - Keeps unlabeled text as (chunk, None)
    - Assumes entities do not heavily overlap. Overlaps are clipped/skipped.
    """
    print("Started")
    if not text:
        return []

    # sanitize + sort
    spans = []
    N = len(text)
    for e in entities or []:
        s = max(0, int(e.get("start", 0)))
        t = min(N, int(e.get("end", 0)))
        if s < t:
            spans.append((s, t, str(e.get("label", "ENT"))))
    spans.sort(key=lambda x: (x[0], x[1]))

    pairs = []
    cursor = 0
    for s, t, label in spans:
        # skip entirely before cursor (overlap)
        if t <= cursor:
            continue
        # add gap before entity
        if s > cursor:
            pairs.append((text[cursor:s], None))
        # clip start to cursor if partial overlap
        s = max(s, cursor)
        pairs.append((text[s:t], label))
        cursor = t
    # tail
    if cursor < N:
        pairs.append((text[cursor:], None))

    print(pairs)
    return pairs


def call_api(path: str, payload: dict):
    try:
        resp = requests.post(
            f"{BASE_URL}{path}",
            json=payload,
            headers=_headers(),
            timeout=TIMEOUT
        )
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        return None, str(e)


def pills_html(labels):
    if not labels:
        return "<i>No sources returned.</i>"
    return (
        "<div class='pill-container'>"
        + "".join([f"<span class='pill'>{lbl}</span>" for lbl in labels])
        + "</div>"
    )

# UI callbacks
def ui_summarize(text):
    if not text.strip():
        return "", {}, gr.update(value="Please provide text.", visible=True)

    payload = {
        "text": text
    }
    data, err = call_api(ENDPOINTS["summarize"], payload)
    if err:
        return "", gr.update(value=f"{err}", visible=True)

    summary = data.get("summary", "")
    return summary, gr.update(value="", visible=False)


def ui_ner(text):
    if not text.strip():
        return ("", []), gr.update(value="Please provide text.", visible=True)

    payload = {"text": text}
    data, err = call_api(ENDPOINTS["ner"], payload)
    if err:
        return ("", []), gr.update(value=f"{err}", visible=True)

    raw_text = data.get("text", text)
    entities = data.get("entities", [])

    spans = [(e["start"], e["end"], e["label"]) for e in entities]
    return (raw_text, spans), gr.update(value="", visible=False)

def ui_ner_multi(text):
    print("ui ner multi")
    if not text.strip():
        empty = []
        hide = gr.update(visible=False)
        return empty, hide, empty, hide, empty, hide, empty, hide, gr.update(value="Please provide text.", visible=True)

    payload = {"text": text}
    data, err = call_api(ENDPOINTS["ner"], payload)
    print(data, err)
    if err:
        empty = []
        hide = gr.update(visible=False)
        return empty, hide, empty, hide, empty, hide, empty, hide, gr.update(value=f"{err}", visible=True)

    raw_text = data.get("text", text)
    results = data.get("results", [])

    outs, vis = [], []
    for i in range(4):
        if i < len(results):
            r = results[i]
            pairs = ner_pairs(raw_text, r.get("entities", []))  # <-- convert to pairs
            outs.append(pairs)  # HighlightedText expects list[(chunk, label)]
            vis.append(gr.update(label=r.get("name", f"Model {i+1}"), visible=True))
        else:
            outs.append([])  # empty pairs
            vis.append(gr.update(visible=False))

    return outs[0], vis[0], outs[1], vis[1], outs[2], vis[2], outs[3], vis[3], gr.update(value="", visible=False)

def ui_rag(question):
    if not question.strip():
        msg = "Please ask a question."
        return "", gr.update(value="<i>No contexts</i>"), "", {}, gr.update(value=msg, visible=True)

    payload = {"query": question}
    data, err = call_api(ENDPOINTS["rag"], payload)
    if err:
        return "", gr.update(value="<i>No contexts</i>"), "", {}, gr.update(value=f"{err}", visible=True)

    resp = data.get("response") or {}
    scores = data.get("scores") or {}

    # 1) Answer
    answer = resp.get("answer") or data.get("answer", "")

    # 2) Contexts -> Markdown (optional but nice)
    contexts = resp.get("contexts") or []
    if not contexts and "sources" in data:
        # Map older "sources" shape into contexts-style md if present
        contexts = [s.get("text", "") if isinstance(s, dict) else str(s) for s in data.get("sources", [])]

    ctx_md = ""
    if contexts:
        for i, c in enumerate(contexts, 1):
            ctx_md += f"*Context {i}*\n\n\n{c}\n\n\n"
    else:
        ctx_md = "<i>No contexts</i>"

    # 3) Metadata -> pills
    metadata = resp.get("metadata") or []
    # Build labels like: "Surgery · Total Knee Replacement - 1"
    labels = []
    for m in metadata:
        if isinstance(m, dict):
            spec = (m.get("medical_specialty") or "").strip()
            name = (m.get("sample_name") or "").strip()
            label = " · ".join([p for p in [spec, name] if p])
            if label:
                labels.append(label)
    pills = pills_html(sorted(list(dict.fromkeys(labels))))  # de-dupe, keep order

    # 4) Scores (JSON component)
    return answer, gr.update(value=ctx_md), pills, (scores or {}), gr.update(value="", visible=False)

# Build UI
with gr.Blocks(
    css="""
.pill-container {
  display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px;
}
.pill {
  padding: 4px 10px; border-radius: 9999px; background-color: #eaeaea; color: #000000; font-size: 0.85rem;
}
    """,
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown("# MedNLP: An Integrated Pipeline for Named Entity Recognition, Summarization, and Intelligent Medical Question Answering")

    with gr.Tabs():
        # -------- Summarization --------
        with gr.Tab("Summarization"):
            gr.Markdown("Enter Medical text to get a Summarized version.")
            in_text = gr.Textbox(label="Input Text", lines=12)
            btn = gr.Button("Summarize", variant="primary")
            out_summary = gr.Textbox(label="Summary", lines=8)
            err_sum = gr.Markdown(visible=False)

            btn.click(
                ui_summarize,
                inputs=[in_text],
                outputs=[out_summary, err_sum]
            )

        with gr.Tab("Named Entity Recognition (NER)"):
            gr.Markdown("Enter Medical text to extract Medical Entities.")
            ner_in = gr.Textbox(label="Input Text", lines=10)
            ner_btn = gr.Button("Extract", variant="primary")

            with gr.Row():
                ner_out1 = gr.HighlightedText(label="Model 1", combine_adjacent=True, visible=False)
                ner_out2 = gr.HighlightedText(label="Model 2", combine_adjacent=True, visible=False)
                ner_out3 = gr.HighlightedText(label="Model 3", combine_adjacent=True, visible=False)
            with gr.Row():
                ner_out4 = gr.HighlightedText(label="Model 4", combine_adjacent=True, visible=False)

            err_ner = gr.Markdown(visible=False)

            ner_btn.click(
                ui_ner_multi,
                inputs=[ner_in],
                outputs=[ner_out1, ner_out1, ner_out2, ner_out2, ner_out3, ner_out3, ner_out4, ner_out4, err_ner]
            )

        with gr.Tab("Medical Question Answering Bot"):
            gr.Markdown("Ask a Medical Question to get an Answer.")
            question = gr.Textbox(label="Question", placeholder="Eg: What is the procedure for knee replacement?")
            ask_btn = gr.Button("Ask", variant="primary")

            with gr.Row():
                answer = gr.Textbox(label="Answer", lines=6)
                scores_json = gr.JSON(label="Scores")

            sources_md = gr.Markdown(label="Retrieved Contexts")
            pills = gr.HTML()
            err_rag = gr.Markdown(visible=False)

            ask_btn.click(
                ui_rag,
                inputs=[question],
                outputs=[answer, sources_md, pills, scores_json, err_rag]
            )

if __name__ == "__main__":
    demo.launch()