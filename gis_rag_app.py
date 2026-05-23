"""
🗺️ GIS Document Assistant — RAG App
ITI GIS Track | Section 7 Lab Project

Zero C-compilation dependencies.
Uses: streamlit, pypdf, google-generativeai, numpy only.
"""

import os, json, math, tempfile
from datetime import datetime
from collections import Counter

import streamlit as st

st.set_page_config(
    page_title="🗺️ GIS Document Assistant",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f5f7f5; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e0e0e0; }
.stat-card {
    background:#fff; border:1px solid #e0ede9; border-radius:10px;
    padding:12px 8px; text-align:center; margin-bottom:4px;
}
.stat-num { font-size:24px; font-weight:700; color:#0f6e56; line-height:1; }
.stat-lbl { font-size:11px; color:#6c757d; margin-top:2px; }
.bubble-user {
    background:#0f6e56; color:#fff; padding:11px 15px;
    border-radius:18px 18px 4px 18px;
    margin:6px 0 6px auto; max-width:72%; width:fit-content;
    font-size:14px; line-height:1.6;
}
.bubble-bot {
    background:#fff; color:#212529; padding:11px 15px;
    border:1px solid #dee2e6; border-radius:18px 18px 18px 4px;
    margin:6px auto 6px 0; max-width:82%;
    font-size:14px; line-height:1.6;
}
.bubble-bot-ar { direction:rtl; text-align:right; }
.msg-time { font-size:10px; color:#adb5bd; margin-top:3px; }
.src-chip {
    display:inline-block; background:#e6f4f0; color:#0f6e56;
    border:1px solid #9fe1cb; border-radius:20px;
    padding:2px 10px; font-size:11px; margin:2px 4px 2px 0;
}
.top-row {
    display:flex; justify-content:space-between;
    font-size:12px; padding:3px 0; border-bottom:1px solid #f1f3f5;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
def _init():
    for k, v in {
        "chunks":         [],      # list of {text, page, source}
        "embeddings":     [],      # parallel list of embedding vectors
        "messages":       [],
        "docs_meta":      [],
        "question_count": 0,
        "page_hits":      Counter(),
        "lang":           "en",
        "api_key":        "",
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

# ── Translations ──────────────────────────────────────────────────────────────
LANG = {
    "en": {
        "title":      "🗺️ GIS Document Assistant",
        "subtitle":   "RAG-powered chat with your GIS documents",
        "upload_lbl": "📄 Upload PDF(s)",
        "processing": "Processing",
        "ask_ph":     "Ask anything about your GIS document…",
        "send":       "Send ↗",
        "no_key":     "⚠️ Enter your Gemini API key in the sidebar first.",
        "no_doc":     "⚠️ Upload at least one PDF first.",
        "thinking":   "🤔 Thinking…",
        "src_hdr":    "📚 Sources",
        "export":     "⬇️ Export conversation (JSON)",
        "stats_q":    "Questions",
        "stats_c":    "Chunks",
        "stats_d":    "Docs",
        "top_pages":  "🔥 Top referenced pages",
        "hint":       "Upload a PDF then start asking questions!",
        "system": (
            "You are a GIS expert assistant. "
            "Answer using ONLY the provided document context. "
            "Cite page numbers when possible. "
            "If the answer is not in the context say so clearly. "
            "Use markdown for code blocks."
        ),
    },
    "ar": {
        "title":      "🗺️ مساعد مستندات GIS",
        "subtitle":   "محادثة ذكية مع مستندات GIS الخاصة بك",
        "upload_lbl": "📄 ارفع ملفات PDF",
        "processing": "جارٍ المعالجة",
        "ask_ph":     "اسأل أي شيء عن مستند GIS…",
        "send":       "إرسال ↗",
        "no_key":     "⚠️ أدخل مفتاح Gemini API في الشريط الجانبي أولاً.",
        "no_doc":     "⚠️ ارفع ملف PDF واحداً على الأقل أولاً.",
        "thinking":   "🤔 جارٍ التفكير…",
        "src_hdr":    "📚 المصادر",
        "export":     "⬇️ تصدير المحادثة (JSON)",
        "stats_q":    "أسئلة",
        "stats_c":    "مقاطع",
        "stats_d":    "ملفات",
        "top_pages":  "🔥 أكثر الصفحات استخداماً",
        "hint":       "ارفع ملف PDF وابدأ بطرح الأسئلة!",
        "system": (
            "أنت مساعد خبير في GIS. "
            "أجب باستخدام سياق المستند المقدم فقط. "
            "اذكر أرقام الصفحات عند الإمكان. "
            "اكتب إجاباتك باللغة العربية."
        ),
    },
}

def t(k): return LANG[st.session_state.lang].get(k, k)

# ── Pure-Python RAG helpers ───────────────────────────────────────────────────

def pdf_to_chunks(file_bytes: bytes, filename: str) -> list[dict]:
    """Extract text from PDF and split into overlapping chunks."""
    import pypdf, io
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    chunks = []
    chunk_size, overlap = 600, 80

    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue
        # slide a window over the page text
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if len(chunk) > 40:
                chunks.append({
                    "text":   chunk,
                    "page":   page_num,
                    "source": filename,
                })
            start += chunk_size - overlap

    return chunks


def get_embedding(text: str, api_key: str) -> list[float]:
    """Call Gemini embedding API."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document",
    )
    return result["embedding"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def retrieve(query: str, k: int = 4) -> list[tuple[dict, float]]:
    """Embed query, find top-k chunks by cosine similarity."""
    if not st.session_state.chunks:
        return []
    q_emb = get_embedding(query, st.session_state.api_key)
    scored = [
        (chunk, cosine_similarity(q_emb, emb))
        for chunk, emb in zip(st.session_state.chunks, st.session_state.embeddings)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def ask_gemini(prompt: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=st.session_state.api_key)
    model  = genai.GenerativeModel("gemini-2.0-flash")
    result = model.generate_content(prompt)
    return result.text


def build_prompt(question: str, results: list[tuple[dict, float]]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {r['source']}, page {r['page']}, score {score:.2f}]\n{r['text']}"
        for r, score in results
    )
    return (
        f"{t('system')}\n\n"
        f"DOCUMENT CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )


def export_json() -> str:
    data = {
        "export_time":  datetime.now().isoformat(),
        "language":     st.session_state.lang,
        "documents":    st.session_state.docs_meta,
        "stats": {
            "questions":    st.session_state.question_count,
            "total_chunks": len(st.session_state.chunks),
            "top_pages":    [
                {"page": k, "hits": v}
                for k, v in st.session_state.page_hits.most_common(10)
            ],
        },
        "conversation": [
            {k: v for k, v in m.items() if k != "sources_raw"}
            for m in st.session_state.messages
        ],
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗺️ GIS Assistant")

    # Language
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🇬🇧 English",
                     type="primary" if st.session_state.lang == "en" else "secondary",
                     use_container_width=True):
            st.session_state.lang = "en"; st.rerun()
    with col2:
        if st.button("🇪🇬 العربية",
                     type="primary" if st.session_state.lang == "ar" else "secondary",
                     use_container_width=True):
            st.session_state.lang = "ar"; st.rerun()

    st.divider()

    # API key
    st.markdown("**🔑 Gemini API Key**")
    key_input = st.text_input("key", value=st.session_state.api_key,
                               type="password", placeholder="AIza…",
                               label_visibility="collapsed")
    if key_input != st.session_state.api_key:
        st.session_state.api_key = key_input
    if st.session_state.api_key:
        st.success("✅ Key ready")
    else:
        st.caption("Get key → [aistudio.google.com](https://aistudio.google.com/apikey)")

    st.divider()

    # Upload
    st.markdown(f"**{t('upload_lbl')}**")
    uploaded = st.file_uploader("pdfs", type=["pdf"],
                                 accept_multiple_files=True,
                                 label_visibility="collapsed")

    if uploaded and st.session_state.api_key:
        new_names = sorted(f.name for f in uploaded)
        old_names = sorted(d["name"] for d in st.session_state.docs_meta)
        if new_names != old_names:
            all_chunks, all_embeddings = [], []
            progress = st.progress(0, text=f"{t('processing')}…")
            total = sum(1 for _ in uploaded)  # count
            for i, uf in enumerate(uploaded):
                progress.progress((i) / total, text=f"📄 {uf.name}…")
                file_bytes = uf.read()
                chunks = pdf_to_chunks(file_bytes, uf.name)
                # embed each chunk
                for j, chunk in enumerate(chunks):
                    emb = get_embedding(chunk["text"], st.session_state.api_key)
                    all_chunks.append(chunk)
                    all_embeddings.append(emb)
                progress.progress((i + 1) / total, text=f"✅ {uf.name}")

            st.session_state.chunks     = all_chunks
            st.session_state.embeddings = all_embeddings
            st.session_state.docs_meta  = [{"name": f.name} for f in uploaded]
            progress.empty()
            st.success(f"✅ {len(all_chunks)} chunks indexed!")

    elif uploaded and not st.session_state.api_key:
        st.warning("Enter API key first, then re-upload.")

    # Doc list
    if st.session_state.docs_meta:
        st.markdown("**📂 Indexed docs**")
        for d in st.session_state.docs_meta:
            st.markdown(
                f'<div style="background:#e6f4f0;border-radius:6px;padding:5px 9px;'
                f'font-size:12px;margin-bottom:4px;">📄 {d["name"]}</div>',
                unsafe_allow_html=True)

    st.divider()

    # Stats
    st.markdown("**📊 Stats**")
    c1, c2, c3 = st.columns(3)
    for col, num, lbl in [
        (c1, st.session_state.question_count,     t("stats_q")),
        (c2, len(st.session_state.chunks),         t("stats_c")),
        (c3, len(st.session_state.docs_meta),      t("stats_d")),
    ]:
        with col:
            st.markdown(
                f'<div class="stat-card"><div class="stat-num">{num}</div>'
                f'<div class="stat-lbl">{lbl}</div></div>',
                unsafe_allow_html=True)

    if st.session_state.page_hits:
        st.markdown(f"**{t('top_pages')}**")
        for page_key, hits in st.session_state.page_hits.most_common(5):
            st.markdown(
                f'<div class="top-row"><span>📄 {page_key}</span>'
                f'<span style="color:#0f6e56;font-weight:600">{hits}×</span></div>',
                unsafe_allow_html=True)

    st.divider()

    if st.session_state.messages:
        st.download_button(
            label=t("export"),
            data=export_json(),
            file_name=f"gis-chat-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages       = []
            st.session_state.question_count = 0
            st.session_state.page_hits      = Counter()
            st.rerun()


# ── MAIN ──────────────────────────────────────────────────────────────────────
st.markdown(f"## {t('title')}")
st.caption(t("subtitle"))

# Quick buttons
QUICK = {
    "en": ["What is this document about?", "What renderer types are available?",
           "How do Promises work?", "Explain Map and View",
           "List all widget types", "What is the Accessor pattern?"],
    "ar": ["عمّ يتحدث هذا المستند؟", "ما أنواع الـ Renderer المتاحة؟",
           "كيف تعمل Promises في الـ API؟", "اشرح Map و View",
           "اذكر أنواع الـ Widgets"],
}
triggered = None
qs = QUICK[st.session_state.lang]
cols = st.columns(len(qs))
for col, q in zip(cols, qs):
    with col:
        if st.button(q, use_container_width=True, key=f"q_{q[:18]}"):
            triggered = q

st.divider()

# Chat history
if not st.session_state.messages:
    st.info(f"💡 {t('hint')}")
else:
    for msg in st.session_state.messages:
        role    = msg["role"]
        content = msg["content"]
        ts      = msg.get("ts", "")
        sources = msg.get("sources", [])

        if role == "user":
            st.markdown(
                f'<div style="display:flex;justify-content:flex-end">'
                f'<div class="bubble-user">{content}'
                f'<div class="msg-time">{ts}</div></div></div>',
                unsafe_allow_html=True)
        else:
            ar = "bubble-bot-ar" if st.session_state.lang == "ar" else ""
            safe = content.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")
            chips = "".join(
                f'<span class="src-chip">📄 {r["source"]} · p.{r["page"]} '
                f'<span style="opacity:.6">({score:.2f})</span></span>'
                for r, score in sources
            )
            src_block = (
                f'<div style="margin-top:8px;font-size:11px;font-weight:600;'
                f'color:#6c757d;text-transform:uppercase;letter-spacing:.05em">'
                f'{t("src_hdr")}</div>{chips}'
            ) if sources else ""
            st.markdown(
                f'<div class="bubble-bot {ar}">{safe}'
                f'<div class="msg-time">{ts}</div>'
                f'{src_block}</div>',
                unsafe_allow_html=True)

# Input
st.markdown("---")
ic, bc = st.columns([9, 1])
with ic:
    user_input = st.text_area("q", placeholder=t("ask_ph"), height=80,
                               label_visibility="collapsed", key="input_area")
with bc:
    st.write("")
    send = st.button(t("send"), type="primary", use_container_width=True)

# Process
question = triggered or (user_input.strip() if send else None)

if question:
    if not st.session_state.api_key:
        st.error(t("no_key")); st.stop()
    if not st.session_state.chunks:
        st.error(t("no_doc")); st.stop()

    ts_now = datetime.now().strftime("%H:%M")
    st.session_state.messages.append(
        {"role": "user", "content": question, "ts": ts_now, "sources": []})
    st.session_state.question_count += 1

    with st.spinner(t("thinking")):
        results = retrieve(question, k=4)

        for chunk, _ in results:
            key = f"{chunk['source']}:p{chunk['page']}"
            st.session_state.page_hits[key] += 1

        prompt = build_prompt(question, results)
        try:
            answer = ask_gemini(prompt)
        except Exception as e:
            answer = f"❌ Gemini error: {e}"

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "ts":      datetime.now().strftime("%H:%M"),
        "sources": results,
    })
    st.rerun()
