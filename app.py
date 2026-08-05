"""
Scout AI — Interfaz Streamlit

Dashboard de scouting táctico sobre datos de La Liga (StatsBomb).
El usuario escribe queries en lenguaje natural y recibe informes estructurados.
"""

import streamlit as st

st.set_page_config(
    page_title="Scout AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos ──────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Fondo general */
  .stApp { background-color: #0d0f14; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: #13161e;
    border-right: 1px solid #1f2330;
  }

  /* Input de query — estilamos el contenedor externo, no el textarea interno */
  [data-testid="stChatInput"] {
    background-color: #1a1d27 !important;
    border: 1px solid #2d3347 !important;
    border-radius: 8px !important;
  }
  [data-testid="stChatInput"] textarea {
    background-color: transparent !important;
    color: #e2e8f0 !important;
    border: none !important;
    box-shadow: none !important;
  }

  /* Burbujas de chat usuario */
  [data-testid="stChatMessageContent"] {
    background-color: #1a1d27;
    border-radius: 8px;
    border: 1px solid #2d3347;
  }

  /* Métricas */
  [data-testid="stMetric"] {
    background-color: #1a1d27;
    border: 1px solid #2d3347;
    border-radius: 8px;
    padding: 12px 16px;
  }

  /* Expanders */
  .streamlit-expanderHeader {
    background-color: #1a1d27 !important;
    border: 1px solid #2d3347 !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
  }

  /* Texto general */
  .stMarkdown, p, li { color: #cbd5e1; }
  h1, h2, h3 { color: #f1f5f9; }

  /* Badge de agente */
  .agent-badge {
    display: inline-block;
    background-color: #1e293b;
    border: 1px solid #334155;
    color: #64748b;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 4px;
    margin-right: 6px;
  }
  .agent-badge.active {
    border-color: #3b82f6;
    color: #3b82f6;
  }

  /* Tabla comparativa — monospace */
  .comp-table {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.82rem;
    background-color: #111318;
    border: 1px solid #1f2330;
    border-radius: 6px;
    padding: 16px;
    color: #94a3b8;
    white-space: pre;
    overflow-x: auto;
  }

  /* Historial sidebar */
  .history-item {
    padding: 8px 12px;
    border-radius: 6px;
    border: 1px solid #1f2330;
    margin-bottom: 6px;
    cursor: pointer;
    color: #64748b;
    font-size: 0.82rem;
    background-color: #111318;
  }

  /* Divider */
  hr { border-color: #1f2330; }
</style>
""", unsafe_allow_html=True)

# ── Estado de sesión ─────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

if "last_players" not in st.session_state:
    st.session_state.last_players = []

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚽ Scout AI")
    st.markdown("<p style='color:#475569;font-size:0.8rem'>La Liga · StatsBomb Open Data</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("<p style='color:#475569;font-size:0.75rem;letter-spacing:0.08em;text-transform:uppercase'>Consultas recientes</p>", unsafe_allow_html=True)

    if st.session_state.history:
        for item in reversed(st.session_state.history[-10:]):
            st.markdown(f"<div class='history-item'>↳ {item}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:#2d3347;font-size:0.8rem'>Sin consultas aún</p>", unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <p style='color:#334155;font-size:0.75rem'>
    <b style='color:#475569'>Ejemplos:</b><br><br>
    dame las stats de Busquets<br><br>
    quiero un mediocentro defensivo que presione<br><br>
    compará a Messi con Xavier Hernandez
    </p>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("# Scout AI")
st.markdown("<p style='color:#475569;margin-top:-12px'>Scouting táctico · La Liga · 1885 jugadores</p>", unsafe_allow_html=True)
st.divider()

def _render_badges(agentes: list[str]) -> str:
    all_agents = ["rag", "stats", "comp"]
    badges = ""
    for a in all_agents:
        css = "active" if a in agentes else ""
        label = {"rag": "Búsqueda", "stats": "Stats", "comp": "Comparativa"}[a]
        badges += f"<span class='agent-badge {css}'>{label}</span>"
    return badges


def _render_informe(informe) -> None:
    """Renderiza un InformeScouting en la interfaz."""
    st.markdown(_render_badges(informe.agentes_ejecutados), unsafe_allow_html=True)
    st.markdown("")

    if informe.jugadores_sugeridos:
        with st.expander("🔍 Jugadores sugeridos por perfil", expanded=True):
            st.markdown(f"```\n{informe.jugadores_sugeridos}\n```")

    if informe.estadisticas:
        with st.expander("📊 Estadísticas", expanded=True):
            st.markdown(f"```\n{informe.estadisticas}\n```")

    if informe.comparativa:
        with st.expander("⚖️ Comparativa", expanded=True):
            st.markdown(f"<div class='comp-table'>{informe.comparativa}</div>", unsafe_allow_html=True)

    if informe.conclusion:
        st.markdown("---")
        st.markdown(f"<p style='color:#94a3b8;font-style:italic'>{informe.conclusion}</p>", unsafe_allow_html=True)


# ── Historial de mensajes ─────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "⚽"):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            _render_informe(msg["informe"])

# ── Input de query ────────────────────────────────────────────────────────────

if query := st.chat_input("Describí un perfil, pedí stats o compará jugadores..."):

    # Mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.history.append(query)

    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)

    # Ejecutar el agente
    with st.chat_message("assistant", avatar="⚽"):
        with st.spinner("Analizando..."):
            from scout.graph import run
            informe = run(query, context_players=st.session_state.last_players)

        if informe.jugadores_detectados:
            st.session_state.last_players = informe.jugadores_detectados

        _render_informe(informe)

    st.session_state.messages.append({"role": "assistant", "informe": informe})
    st.rerun()
