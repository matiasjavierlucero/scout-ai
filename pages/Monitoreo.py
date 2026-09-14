"""
Página de documentación — embebe monitoreo-llm-produccion.html dentro de la app.
"""

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Monitoreo LLM — Scout AI",
    page_icon="📊",
    layout="wide",
)

# Ocultar el padding de Streamlit para que el HTML ocupe toda la pantalla
st.markdown("""
<style>
  .block-container { padding-top: 0 !important; padding-bottom: 0 !important; }
  header[data-testid="stHeader"] { display: none; }
</style>
""", unsafe_allow_html=True)

html_path = Path(__file__).parent.parent / "monitoreo-llm-produccion.html"

if not html_path.exists():
    st.error(f"No se encontró el archivo: {html_path}")
    st.stop()

html_content = html_path.read_text(encoding="utf-8")

st.components.v1.html(html_content, height=9000, scrolling=True)
