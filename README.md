# 🗺️ GIS Document Assistant

RAG-powered Streamlit app — ITI GIS Track, Section 7 Lab Project.

## ✅ Features

| Feature | Status |
|---|---|
| Upload multiple PDFs | ✅ |
| Chunk + embed + ChromaDB | ✅ |
| Chat with Gemini | ✅ |
| Sources with page numbers | ✅ |
| Arabic / English toggle | ✅ |
| Export conversation JSON | ✅ |
| Live stats (questions, chunks, top pages) | ✅ |

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run gis_rag_app.py
```

The app opens at **http://localhost:8501**
or at: **https://gis-rag-assistant.streamlit.app/**

## 🔑 API Key

Get your free Gemini key from:
👉 https://aistudio.google.com/apikey

Paste it in the sidebar when the app opens.

## 📄 Suggested PDFs

- ArcGIS JS API docs (included in course)
- QGIS User Guide (free from qgis.org)
- OGC Standards (GeoJSON, KML)
- Any GIS PDF manual or tutorial
