"""
🗺️ GIS Document Assistant — RAG App
ITI GIS Track | Section 7 Lab Project

Features:
  ✅ Upload multiple PDFs
  ✅ Chunk + embed + store in ChromaDB
  ✅ Chat with sources & page refs
  ✅ Arabic / English toggle
  ✅ Export conversation as JSON
  ✅ Live statistics (questions, chunks, top pages)
"""

import os
import json
import time
import tempfile
from datetime import datetime
from collections import Counter

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="🗺️ GIS Document Assistant",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports with friendly error messages ─────────────────────────────────
def check_deps():
    missing = []
    for pkg, imp in [
        ("langchain-google-genai", "langchain_google_genai"),
        ("langchain-chroma",       "langchain_chroma"),
        ("langchain-community",    "langchain_community"),
        ("langchain-text-splitters","langchain_text_splitters"),
        ("chromadb",               "chromadb"),
        ("pypdf",                  "pypdf"),
    ]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    return missing


missing = check_deps()
if missing:
    st.error("⚠️ Missing packages. Run this in your terminal, then restart:")
    st.code("pip install " + " ".join(missing))
    st.stop()

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] { background: #f8f9fa; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e9ecef; }

/* ── Sidebar header ── */
.sidebar-title {
    font-size: 18px; font-weight: 600; color: #0f6e56;
    display: flex; align-items: center; gap: 8px;
    padding: 4px 0 12px 0; border-bottom: 1px solid #e9ecef; margin-bottom: 12px;
}

/* ── Stat cards ── */
.stat-row { display: flex; gap: 10px; margin-bottom: 16px; }
.stat-card {
    flex: 1; background: #fff; border: 1px solid #e9ecef;
    border-radius: 10px; padding: 12px 10px; text-align: center;
}
.stat-num  { font-size: 22px; font-weight: 700; color: #0f6e56; line-height: 1; }
.stat-lbl  { font-size: 11px; color: #6c757d; margin-top: 3px; }

/* ── Chat bubbles ── */
.chat-wrapper { max-width: 860px; margin: 0 auto; }
.bubble-user {
    background: #0f6e56; color: #fff; padding: 12px 16px;
    border-radius: 18px 18px 4px 18px; margin: 6px 0 6px auto;
    max-width: 70%; width: fit-content; font-size: 14px; line-height: 1.6;
}
.bubble-bot {
    background: #ffffff; color: #212529; padding: 12px 16px;
    border: 1px solid #dee2e6; border-radius: 18px 18px 18px 4px;
    margin: 6px auto 6px 0; max-width: 80%; font-size: 14px; line-height: 1.6;
}
.bubble-bot-ar {
    direction: rtl; text-align: right;
}
.msg-time { font-size: 10px; color: #adb5bd; margin-top: 4px; }

/* ── Source chips ── */
.src-header {
    font-size: 11px; font-weight: 600; color: #6c757d;
    margin-top: 10px; margin-bottom: 4px; text-transform: uppercase;
    letter-spacing: .05em;
}
.src-chip {
    display: inline-block; background: #e6f4f0; color: #0f6e56;
    border: 1px solid #9fe1cb; border-radius: 20px;
    padding: 2px 10px; font-size: 11px; margin: 2px 4px 2px 0;
}

/* ── Doc pills in sidebar ── */
.doc-pill {
    background: #e6f4f0; color: #085041;
    border-radius: 8px; padding: 6px 10px;
    font-size: 12px; margin-bottom: 6px;
    display: flex; justify-content: space-between; align-items: center;
}
.doc-pill span { font-weight: 600; }

/* ── Input area ── */
.input-hint { font-size: 11px; color: #adb5bd; margin-top: 4px; }

/* ── Top-page table ── */
.top-page-row { display: flex; justify-content: space-between;
    font-size: 12px; padding: 3px 0; border-bottom: 1px solid #f1f3f5; }
</style>
""", unsafe_allow_html=True)


# ── Session state defaults ────────────────────────────────────────────────────
def _init():
    defaults = {
        "vectorstore":    None,
        "messages":       [],          # [{role, content, sources, ts}]
        "docs_meta":      [],          # [{name, chunks, pages}]
        "total_chunks":   0,
        "question_count": 0,
        "page_hits":      Counter(),   # "filename:pageN" → count
        "lang":           "en",
        "api_key":        "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ── Helpers ───────────────────────────────────────────────────────────────────
LANG = {
    "en": {
        "title":       "🗺️ GIS Document Assistant",
        "subtitle":    "RAG-powered chat with your GIS documents",
        "upload_lbl":  "📄 Upload PDF(s)",
        "processing":  "Processing",
        "ask_ph":      "Ask anything about your GIS document…",
        "send":        "Send ↗",
        "no_key":      "⚠️ Please enter your Gemini API key in the sidebar.",
        "no_doc":      "⚠️ Please upload at least one PDF first.",
        "thinking":    "🤔 Thinking…",
        "sources_hdr": "📚 Sources used",
        "export":      "⬇️ Export conversation (JSON)",
        "stats_q":     "Questions",
        "stats_c":     "Chunks",
        "stats_d":     "Documents",
        "top_pages":   "🔥 Most-referenced pages",
        "no_chat":     "No conversation yet.",
        "empty_hint":  "Upload a PDF and start asking questions!",
        "system": (
            "You are a GIS expert assistant. "
            "Answer questions using ONLY the provided document context. "
            "Be specific and cite page numbers when possible. "
            "If the answer is not in the context, say so clearly. "
            "Format code with markdown code fences."
        ),
    },
    "ar": {
        "title":       "🗺️ مساعد مستندات GIS",
        "subtitle":    "محادثة ذكية مع مستندات GIS الخاصة بك",
        "upload_lbl":  "📄 ارفع ملفات PDF",
        "processing":  "جارٍ المعالجة",
        "ask_ph":      "اسأل أي شيء عن مستند GIS…",
        "send":        "إرسال ↗",
        "no_key":      "⚠️ الرجاء إدخال مفتاح Gemini API في الشريط الجانبي.",
        "no_doc":      "⚠️ الرجاء رفع ملف PDF واحد على الأقل أولاً.",
        "thinking":    "🤔 جارٍ التفكير…",
        "sources_hdr": "📚 المصادر المستخدمة",
        "export":      "⬇️ تصدير المحادثة (JSON)",
        "stats_q":     "أسئلة",
        "stats_c":     "مقاطع",
        "stats_d":     "مستندات",
        "top_pages":   "🔥 الصفحات الأكثر استخداماً",
        "no_chat":     "لا توجد محادثة بعد.",
        "empty_hint":  "ارفع ملف PDF وابدأ بطرح الأسئلة!",
        "system": (
            "أنت مساعد خبير في GIS. "
            "أجب على الأسئلة باستخدام سياق المستند المقدم فقط. "
            "كن محدداً واذكر أرقام الصفحات عندما يكون ذلك ممكناً. "
            "إذا لم تكن الإجابة في السياق، قل ذلك بوضوح. "
            "اكتب إجاباتك باللغة العربية."
        ),
    },
}


def t(key: str) -> str:
    """Translate key using current language."""
    return LANG[st.session_state.lang].get(key, key)


def build_vectorstore(pdf_paths: list[str], api_key: str) -> tuple[object, int]:
    """Load PDFs → chunk → embed → store in Chroma. Returns (store, n_chunks)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        pages  = loader.load()
        chunks = splitter.split_documents(pages)
        all_docs.extend(chunks)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )
    store = Chroma.from_documents(all_docs, embeddings)
    return store, len(all_docs)


def retrieve(query: str, k: int = 4):
    """Similarity search; returns list of (Document, score)."""
    if st.session_state.vectorstore is None:
        return []
    return st.session_state.vectorstore.similarity_search_with_score(query, k=k)


def ask_gemini(prompt: str, api_key: str) -> str:
    """Call Gemini and return text answer."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key=api_key,
    )
    response = llm.invoke(prompt)
    return response.content


def build_prompt(question: str, docs_with_scores) -> str:
    context_parts = []
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        page   = doc.metadata.get("page", "?")
        source = os.path.basename(doc.metadata.get("source", "doc"))
        context_parts.append(
            f"[Source {i}: {source}, page {page}, similarity {score:.3f}]\n"
            f"{doc.page_content}"
        )
    context = "\n\n---\n\n".join(context_parts)
    return (
        f"{t('system')}\n\n"
        f"DOCUMENT CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )


def source_chips_html(docs_with_scores) -> str:
    chips = []
    for doc, score in docs_with_scores:
        page   = doc.metadata.get("page", "?")
        source = os.path.basename(doc.metadata.get("source", "doc"))
        chips.append(
            f'<span class="src-chip">📄 {source} · p.{page} '
            f'<span style="opacity:.6">({score:.2f})</span></span>'
        )
    return "".join(chips)


def export_json() -> str:
    top_pages = [
        {"page": k, "hits": v}
        for k, v in st.session_state.page_hits.most_common(10)
    ]
    data = {
        "export_time": datetime.now().isoformat(),
        "language": st.session_state.lang,
        "documents": st.session_state.docs_meta,
        "stats": {
            "questions":    st.session_state.question_count,
            "total_chunks": st.session_state.total_chunks,
            "top_pages":    top_pages,
        },
        "conversation": st.session_state.messages,
    }
    return json.dumps(data, ensure_ascii=False, indent=2)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🗺️ GIS Assistant</div>',
                unsafe_allow_html=True)

    # Language toggle
    col_en, col_ar = st.columns(2)
    with col_en:
        if st.button("🇬🇧 English",
                     type="primary" if st.session_state.lang == "en" else "secondary",
                     use_container_width=True):
            st.session_state.lang = "en"
            st.rerun()
    with col_ar:
        if st.button("🇪🇬 العربية",
                     type="primary" if st.session_state.lang == "ar" else "secondary",
                     use_container_width=True):
            st.session_state.lang = "ar"
            st.rerun()

    st.divider()

    # API key
    st.markdown("**🔑 Gemini API Key**")
    api_input = st.text_input(
        "API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="AIza…",
        label_visibility="collapsed",
    )
    if api_input != st.session_state.api_key:
        st.session_state.api_key = api_input

    if st.session_state.api_key:
        st.success("✅ Key saved", icon="🔑")
    else:
        st.caption("Get key at [aistudio.google.com](https://aistudio.google.com/apikey)")

    st.divider()

    # PDF upload
    st.markdown(f"**{t('upload_lbl')}**")
    uploaded = st.file_uploader(
        "PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded and st.session_state.api_key:
        # Check if this is a new set of files
        new_names = sorted([f.name for f in uploaded])
        old_names = sorted([d["name"] for d in st.session_state.docs_meta])
        if new_names != old_names:
            with st.spinner(f"{t('processing')}…"):
                tmp_paths = []
                for uf in uploaded:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    tmp.write(uf.read())
                    tmp.flush()
                    tmp_paths.append((uf.name, tmp.name))

                try:
                    store, n_chunks = build_vectorstore(
                        [p for _, p in tmp_paths],
                        st.session_state.api_key,
                    )
                    st.session_state.vectorstore  = store
                    st.session_state.total_chunks  = n_chunks
                    st.session_state.docs_meta     = [
                        {"name": name, "tmp_path": path}
                        for name, path in tmp_paths
                    ]
                    st.success(f"✅ {n_chunks} chunks indexed!")
                except Exception as e:
                    st.error(f"Error building index: {e}")
                finally:
                    for _, p in tmp_paths:
                        try: os.unlink(p)
                        except: pass

    elif uploaded and not st.session_state.api_key:
        st.warning("Enter your API key first, then re-upload.")

    # Indexed documents list
    if st.session_state.docs_meta:
        st.markdown("**📂 Indexed documents**")
        for doc in st.session_state.docs_meta:
            st.markdown(
                f'<div class="doc-pill"><span>📄 {doc["name"]}</span></div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # Statistics
    st.markdown("**📊 Stats**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="stat-card"><div class="stat-num">{st.session_state.question_count}</div>'
            f'<div class="stat-lbl">{t("stats_q")}</div></div>',
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            f'<div class="stat-card"><div class="stat-num">{st.session_state.total_chunks}</div>'
            f'<div class="stat-lbl">{t("stats_c")}</div></div>',
            unsafe_allow_html=True)
    with c3:
        st.markdown(
            f'<div class="stat-card"><div class="stat-num">{len(st.session_state.docs_meta)}</div>'
            f'<div class="stat-lbl">{t("stats_d")}</div></div>',
            unsafe_allow_html=True)

    # Top pages
    if st.session_state.page_hits:
        st.markdown(f"**{t('top_pages')}**")
        for page_key, hits in st.session_state.page_hits.most_common(5):
            st.markdown(
                f'<div class="top-page-row"><span>📄 {page_key}</span>'
                f'<span style="color:#0f6e56;font-weight:600">{hits}×</span></div>',
                unsafe_allow_html=True)

    st.divider()

    # Export
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


# ── MAIN AREA ─────────────────────────────────────────────────────────────────
st.markdown(f"## {t('title')}")
st.caption(t("subtitle"))

# ── Quick-question chips ──────────────────────────────────────────────────────
QUICK_EN = [
    "What is this document about?",
    "What renderer types are available?",
    "How do Promises work in the API?",
    "Explain Map and View",
    "List all widget types",
    "What is the Accessor pattern?",
]
QUICK_AR = [
    "عمّ يتحدث هذا المستند؟",
    "ما أنواع الـ Renderer المتاحة؟",
    "كيف تعمل Promises في الـ API؟",
    "اشرح Map و View",
    "اذكر أنواع الـ Widgets",
]

quick_qs = QUICK_AR if st.session_state.lang == "ar" else QUICK_EN
cols = st.columns(len(quick_qs))
triggered_quick = None
for col, q in zip(cols, quick_qs):
    with col:
        if st.button(q, use_container_width=True, key=f"quick_{q[:20]}"):
            triggered_quick = q

st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.info(f"💡 {t('empty_hint')}")
    else:
        for msg in st.session_state.messages:
            role    = msg["role"]
            content = msg["content"]
            ts      = msg.get("ts", "")
            sources = msg.get("sources", [])

            if role == "user":
                align = "right" if st.session_state.lang == "en" else "right"
                st.markdown(
                    f'<div style="display:flex;justify-content:flex-end">'
                    f'<div class="bubble-user">{content}'
                    f'<div class="msg-time">{ts}</div></div></div>',
                    unsafe_allow_html=True,
                )
            else:
                ar_cls = "bubble-bot-ar" if st.session_state.lang == "ar" else ""
                # Render answer (supports markdown via st.markdown inside expander)
                with st.container():
                    st.markdown(
                        f'<div class="bubble-bot {ar_cls}">'
                        f'{content.replace(chr(10), "<br>")}'
                        f'<div class="msg-time">{ts}</div></div>',
                        unsafe_allow_html=True,
                    )
                    if sources:
                        chips = source_chips_html(sources)
                        st.markdown(
                            f'<div class="src-header">{t("sources_hdr")}</div>{chips}',
                            unsafe_allow_html=True,
                        )

# ── Input row ─────────────────────────────────────────────────────────────────
st.markdown("---")
input_col, btn_col = st.columns([9, 1])

with input_col:
    direction = "rtl" if st.session_state.lang == "ar" else "ltr"
    user_input = st.text_area(
        "Question",
        placeholder=t("ask_ph"),
        height=80,
        label_visibility="collapsed",
        key="user_input_area",
    )

with btn_col:
    st.write("")  # vertical spacer
    send_clicked = st.button(t("send"), type="primary", use_container_width=True)

st.markdown(
    '<p class="input-hint">Shift+Enter for new line · Enter to send</p>',
    unsafe_allow_html=True,
)

# ── Process question ──────────────────────────────────────────────────────────
question = triggered_quick or (user_input.strip() if send_clicked else None)

if question:
    # Validation
    if not st.session_state.api_key:
        st.error(t("no_key"))
        st.stop()
    if st.session_state.vectorstore is None:
        st.error(t("no_doc"))
        st.stop()

    # Save user message
    ts_now = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "role":    "user",
        "content": question,
        "ts":      ts_now,
        "sources": [],
    })
    st.session_state.question_count += 1

    # Retrieve + generate
    with st.spinner(t("thinking")):
        docs_with_scores = retrieve(question, k=4)

        # Track page hits
        for doc, _ in docs_with_scores:
            page   = doc.metadata.get("page", "?")
            source = os.path.basename(doc.metadata.get("source", "doc"))
            st.session_state.page_hits[f"{source}:p{page}"] += 1

        prompt = build_prompt(question, docs_with_scores)

        try:
            answer = ask_gemini(prompt, st.session_state.api_key)
        except Exception as e:
            answer = f"❌ Error calling Gemini: {e}"

    # Save assistant message
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "ts":      datetime.now().strftime("%H:%M"),
        "sources": docs_with_scores,   # (Document, score) tuples
    })

    st.rerun()
