"""
Scout AI — Interfaz Streamlit

Dashboard de scouting táctico sobre datos de La Liga (StatsBomb).
El usuario escribe queries en lenguaje natural y recibe informes estructurados
con visualizaciones interactivas: radar charts, posición en el campo, y cards.
"""

import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="Scout AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .stApp { background-color: #0d0f14; }

  [data-testid="stSidebar"] {
    background-color: #13161e;
    border-right: 1px solid #1f2330;
  }

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

  [data-testid="stChatMessageContent"] {
    background-color: #1a1d27;
    border-radius: 8px;
    border: 1px solid #2d3347;
  }

  [data-testid="stMetric"] {
    background-color: #13161e;
    border: 1px solid #1f2330;
    border-radius: 8px;
    padding: 10px 14px;
  }
  [data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.75rem !important; }
  [data-testid="stMetricValue"] { color: #e2e8f0 !important; font-size: 1.2rem !important; }

  .streamlit-expanderHeader {
    background-color: #1a1d27 !important;
    border: 1px solid #2d3347 !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
  }

  .stMarkdown, p, li { color: #cbd5e1; }
  h1, h2, h3 { color: #f1f5f9; }

  /* Badges de agente activo/inactivo */
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
  .agent-badge.active { border-color: #3b82f6; color: #3b82f6; }

  /* Card de jugador (RAG) */
  .player-card {
    background: #13161e;
    border: 1px solid #1f2330;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
  }
  .player-card:hover { border-color: #2d3347; }
  .player-card-rank {
    color: #334155;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
  }
  .player-card-name {
    color: #e2e8f0;
    font-size: 1rem;
    font-weight: 600;
    margin: 2px 0 6px 0;
  }
  .player-card-meta {
    color: #64748b;
    font-size: 0.78rem;
    margin-top: 6px;
  }

  /* Barra de similitud */
  .sim-bar-bg {
    background: #1e293b;
    border-radius: 3px;
    height: 4px;
    width: 100%;
    margin-top: 8px;
  }
  .sim-bar-fill {
    background: #3b82f6;
    border-radius: 3px;
    height: 4px;
  }

  /* Header de jugador en Stats */
  .player-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
  }
  .player-header-name {
    color: #f1f5f9;
    font-size: 1.25rem;
    font-weight: 700;
  }

  /* Sección de stats interna */
  .stats-section-title {
    color: #475569;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 10px;
    margin-top: 16px;
  }

  /* Historial sidebar */
  .history-item {
    padding: 8px 12px;
    border-radius: 6px;
    border: 1px solid #1f2330;
    margin-bottom: 6px;
    color: #64748b;
    font-size: 0.82rem;
    background-color: #111318;
  }

  hr { border-color: #1f2330; }
</style>
""", unsafe_allow_html=True)

# ── Estado de sesión ──────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "last_players" not in st.session_state:
    st.session_state.last_players = []
if "last_rag_results" not in st.session_state:
    st.session_state.last_rag_results = []
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())
if "view" not in st.session_state:
    st.session_state.view = "chat"

# ── Sidebar ───────────────────────────────────────────────────────────────────

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

    st.divider()
    if st.button("📊 Monitoreo & Docs", use_container_width=True):
        st.session_state.view = "monitoreo"
        st.rerun()

# ── Vista Monitoreo ───────────────────────────────────────────────────────────

if st.session_state.view == "monitoreo":
    if st.button("← Volver al Scout"):
        st.session_state.view = "chat"
        st.rerun()
    html_path = Path("monitoreo-llm-produccion.html")
    if html_path.exists():
        st.components.v1.html(html_path.read_text(encoding="utf-8"), height=9000, scrolling=True)
    else:
        st.error(f"Archivo no encontrado: {html_path.resolve()}")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("# Scout AI")
st.markdown("<p style='color:#475569;margin-top:-12px'>Scouting táctico · La Liga · 1885 jugadores</p>", unsafe_allow_html=True)
st.divider()

# ── Helpers de renderizado ────────────────────────────────────────────────────

def _agent_badges_html(agentes: list[str]) -> str:
    all_agents = ["rag", "stats", "comp"]
    labels = {"rag": "Búsqueda", "stats": "Stats", "comp": "Comparativa"}
    return "".join(
        f"<span class='agent-badge {'active' if a in agentes else ''}'>{labels[a]}</span>"
        for a in all_agents
    )


def _position_badge_html(position: str) -> str:
    """Badge de color según la zona del campo."""
    p = position.lower()
    if "goalkeeper" in p:
        bg = "#475569"
    elif "back" in p:
        bg = "#0e7490"
    elif "midfield" in p:
        bg = "#6d28d9"
    else:  # forward, wing, striker
        bg = "#c2410c"
    return (
        f"<span style='background:{bg};color:white;font-size:0.7rem;"
        f"padding:2px 9px;border-radius:4px;letter-spacing:0.03em;"
        f"white-space:nowrap;font-weight:500'>{position}</span>"
    )


def _parse_rag_scores(text: str) -> list[float]:
    """Extrae los scores de similitud del texto de salida del agente RAG."""
    return [float(m.group(1)) for m in re.finditer(r"similitud:\s+([\d.]+)", text)]


def _minutes_label(minutes: int) -> str:
    warn = " ⚠️" if minutes < 450 else ""
    return f"{minutes:,} min{warn}"


# ── Sección RAG ───────────────────────────────────────────────────────────────

def _render_rag_section(informe) -> None:
    """
    Muestra los resultados del agente RAG como player cards visuales.

    Cada card muestra: rank, nombre, badge de posición, barra de similitud
    y minutos jugados. Si los datos no están disponibles, cae al texto plano.
    """
    from scout.viz import get_player_data

    if not informe.rag_players:
        if informe.jugadores_sugeridos:
            with st.expander("🔍 Jugadores sugeridos por perfil", expanded=True):
                st.markdown(f"```\n{informe.jugadores_sugeridos}\n```")
        return

    scores = _parse_rag_scores(informe.jugadores_sugeridos)
    max_score = max(scores) if scores else 1.0

    st.markdown("<div class='stats-section-title'>🔍 Jugadores sugeridos por perfil</div>", unsafe_allow_html=True)

    cols = st.columns(2)
    for i, player_name in enumerate(informe.rag_players):
        data = get_player_data(player_name)
        score = scores[i] if i < len(scores) else None
        bar_pct = int((score / max_score) * 100) if score and max_score else 0

        with cols[i % 2]:
            pos_badge = _position_badge_html(data["position"]) if data else ""
            sim_html = (
                f"<div class='sim-bar-bg'><div class='sim-bar-fill' style='width:{bar_pct}%'></div></div>"
                if score else ""
            )
            score_txt = f"<span style='color:#475569;font-size:0.75rem'>similitud {score:.3f}</span>" if score else ""
            mins_txt = _minutes_label(data["minutes_played"]) if data else ""
            name = data["name"] if data else player_name

            st.markdown(f"""
            <div class='player-card'>
              <div class='player-card-rank'>#{i+1}</div>
              <div class='player-card-name'>{name}</div>
              {pos_badge}
              {sim_html}
              <div class='player-card-meta'>{score_txt} &nbsp;·&nbsp; {mins_txt}</div>
            </div>
            """, unsafe_allow_html=True)


# ── Sección Stats ─────────────────────────────────────────────────────────────

def _render_stats_section(informe) -> None:
    """
    Muestra las stats de un jugador con:
    - Header con nombre, badge de posición y minutos
    - 4 métricas clave como st.metric
    - Radar chart (percentiles vs el dataset)
    - Posición en el campo
    """
    from scout.viz import build_pitch, build_radar, get_player_data

    if not informe.jugadores_detectados:
        if informe.estadisticas:
            with st.expander("📊 Estadísticas", expanded=True):
                st.markdown(f"```\n{informe.estadisticas}\n```")
        return

    player_name = informe.jugadores_detectados[0]
    data = get_player_data(player_name)

    if data is None:
        with st.expander("📊 Estadísticas", expanded=True):
            st.markdown(f"```\n{informe.estadisticas}\n```")
        return

    st.markdown("<div class='stats-section-title'>📊 Estadísticas</div>", unsafe_allow_html=True)

    # Header del jugador
    warn_html = (
        "<span style='color:#f59e0b;font-size:0.75rem;margin-left:8px'>⚠️ muestra pequeña</span>"
        if data["minutes_played"] < 450 else ""
    )
    st.markdown(
        f"<div class='player-header'>"
        f"<span class='player-header-name'>{data['name']}</span>"
        f"{_position_badge_html(data['position'])}"
        f"{warn_html}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Métricas clave
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("⏱ Minutos", f"{data['minutes_played']:,}")
    m2.metric("⚽ xG/90",   f"{data['xg_p90']:.2f}")
    m3.metric("🔑 Pases/90", f"{data['passes_completed_p90']:.1f}")
    m4.metric("💪 Presiones/90", f"{data['pressures_p90']:.1f}")

    # Radar + campo
    col_radar, col_pitch = st.columns([3, 2])
    with col_radar:
        st.markdown("<p style='color:#475569;font-size:0.72rem;letter-spacing:0.07em;text-transform:uppercase;margin-bottom:4px'>Percentiles vs dataset</p>", unsafe_allow_html=True)
        st.plotly_chart(build_radar(data), use_container_width=True, config={"displayModeBar": False})

    with col_pitch:
        st.markdown(
            f"<p style='color:#475569;font-size:0.72rem;letter-spacing:0.07em;text-transform:uppercase;margin-bottom:4px'>"
            f"Posición · {data['position']}</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(build_pitch(data["position"]), use_container_width=True, config={"displayModeBar": False})


# ── Sección Comparativa ───────────────────────────────────────────────────────

def _render_comp_section(informe) -> None:
    """
    Muestra la comparativa entre dos jugadores con:
    - Radar chart dual superpuesto (azul vs naranja)
    - Tabla de métricas lado a lado con marcador del mejor (◀)
    """
    from scout.viz import build_radar, get_player_data

    if len(informe.jugadores_detectados) < 2:
        if informe.comparativa:
            with st.expander("⚖️ Comparativa", expanded=True):
                st.markdown(
                    f"<pre style='font-family:monospace;font-size:0.82rem;color:#94a3b8;"
                    f"background:#111318;border:1px solid #1f2330;border-radius:6px;"
                    f"padding:16px;overflow-x:auto'>{informe.comparativa}</pre>",
                    unsafe_allow_html=True,
                )
        return

    name_a = informe.jugadores_detectados[0]
    name_b = informe.jugadores_detectados[1]
    data_a = get_player_data(name_a)
    data_b = get_player_data(name_b)

    if data_a is None or data_b is None:
        if informe.comparativa:
            with st.expander("⚖️ Comparativa", expanded=True):
                st.markdown(
                    f"<pre style='font-family:monospace;font-size:0.82rem;color:#94a3b8;"
                    f"background:#111318;border:1px solid #1f2330;border-radius:6px;"
                    f"padding:16px;overflow-x:auto'>{informe.comparativa}</pre>",
                    unsafe_allow_html=True,
                )
        return

    st.markdown("<div class='stats-section-title'>⚖️ Comparativa</div>", unsafe_allow_html=True)

    # Headers de ambos jugadores
    h1, h2 = st.columns(2)
    with h1:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px'>"
            f"<span style='width:12px;height:12px;border-radius:50%;background:#3b82f6;display:inline-block'></span>"
            f"<span style='color:#e2e8f0;font-weight:600'>{data_a['name']}</span>"
            f"</div>"
            f"<div style='margin-top:4px'>{_position_badge_html(data_a['position'])}</div>",
            unsafe_allow_html=True,
        )
    with h2:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px'>"
            f"<span style='width:12px;height:12px;border-radius:50%;background:#f97316;display:inline-block'></span>"
            f"<span style='color:#e2e8f0;font-weight:600'>{data_b['name']}</span>"
            f"</div>"
            f"<div style='margin-top:4px'>{_position_badge_html(data_b['position'])}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Radar superpuesto
    st.plotly_chart(
        build_radar(data_a, data_b),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # Tabla comparativa estructurada
    COMPARE_ROWS: list[tuple[str, str]] = [
        ("⏱ Minutos jugados",    "minutes_played"),
        ("⚽ xG/90",             "xg_p90"),
        ("🎯 Tiros/90",          "shots_p90"),
        ("🔑 Pases completados/90", "passes_completed_p90"),
        ("💪 Presiones/90",      "pressures_p90"),
        ("🛡 Recuperaciones/90", "ball_recoveries_p90"),
        ("✨ Regates ganados/90", "dribbles_won_p90"),
    ]

    st.markdown("<br>", unsafe_allow_html=True)
    c_label, c_a, c_b = st.columns([3, 2, 2])

    c_label.markdown("<p style='color:#475569;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.07em'>Métrica</p>", unsafe_allow_html=True)
    c_a.markdown(f"<p style='color:#3b82f6;font-size:0.72rem;font-weight:600'>{data_a['name'].split()[0]}</p>", unsafe_allow_html=True)
    c_b.markdown(f"<p style='color:#f97316;font-size:0.72rem;font-weight:600'>{data_b['name'].split()[0]}</p>", unsafe_allow_html=True)

    for label, col in COMPARE_ROWS:
        val_a = data_a[col]
        val_b = data_b[col]
        fmt = "{:,}" if col == "minutes_played" else "{:.2f}"

        win_a = " ◀" if val_a > val_b else ""
        win_b = " ◀" if val_b > val_a else ""
        color_a = "#e2e8f0" if val_a >= val_b else "#475569"
        color_b = "#e2e8f0" if val_b >= val_a else "#475569"

        cl, ca, cb = st.columns([3, 2, 2])
        cl.markdown(f"<p style='color:#64748b;font-size:0.82rem;margin:2px 0'>{label}</p>", unsafe_allow_html=True)
        ca.markdown(f"<p style='color:{color_a};font-size:0.9rem;margin:2px 0;font-weight:500'>{fmt.format(val_a)}{win_a}</p>", unsafe_allow_html=True)
        cb.markdown(f"<p style='color:{color_b};font-size:0.9rem;margin:2px 0;font-weight:500'>{fmt.format(val_b)}{win_b}</p>", unsafe_allow_html=True)


# ── Informe completo ──────────────────────────────────────────────────────────

def _render_informe(informe, stream_fn=None) -> None:
    """
    Renderiza un InformeScouting.

    Args:
        informe: Objeto InformeScouting a mostrar.
        stream_fn: Generator para conclusión token-a-token (solo en queries nuevas).
                   Si es None, renderiza informe.conclusion directamente (historial).
    """
    st.markdown(_agent_badges_html(informe.agentes_ejecutados), unsafe_allow_html=True)
    st.markdown("")

    if "rag" in informe.agentes_ejecutados:
        _render_rag_section(informe)

    if "stats" in informe.agentes_ejecutados:
        if "rag" in informe.agentes_ejecutados:
            st.markdown("<br>", unsafe_allow_html=True)
        _render_stats_section(informe)

    if "comp" in informe.agentes_ejecutados:
        _render_comp_section(informe)

    # Conclusión
    st.markdown("---")
    if stream_fn is not None:
        conclusion_text = st.write_stream(stream_fn())
        informe.conclusion = conclusion_text
    elif informe.conclusion:
        st.markdown(
            f"<p style='color:#94a3b8;font-style:italic'>{informe.conclusion}</p>",
            unsafe_allow_html=True,
        )


# ── Historial de mensajes ─────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "⚽"):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            _render_informe(msg["informe"])

# ── Input de query ────────────────────────────────────────────────────────────

if query := st.chat_input("Describí un perfil, pedí stats o compará jugadores..."):

    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.history.append(query)

    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="⚽"):
        from scout.graph import run, stream_conclusion
        from scout.tracing import get_langfuse, get_callback_handler

        langfuse = get_langfuse()
        handler = get_callback_handler()
        callbacks = [handler] if handler else []

        with st.spinner("Analizando..."):
            if langfuse:
                # session_id agrupa todos los turnos de la conversación en Langfuse Sessions.
                # start_as_current_observation activa el contexto OTEL para que el
                # CallbackHandler genere spans hijos (graph + agentes + tools).
                with langfuse.start_as_current_observation(
                    name="scout-query",
                    as_type="agent",
                    input={"query": query},
                ):
                    informe = run(
                        query,
                        context_players=st.session_state.last_players,
                        rag_context=st.session_state.last_rag_results,
                        callbacks=callbacks,
                    )
                    langfuse.update_current_span(
                        output={
                            "agentes_ejecutados": informe.agentes_ejecutados,
                            "jugadores_detectados": informe.jugadores_detectados,
                        },
                    )
            else:
                informe = run(
                    query,
                    context_players=st.session_state.last_players,
                    rag_context=st.session_state.last_rag_results,
                )

        if langfuse:
            langfuse.flush()

        # Actualizar contexto de jugadores para el próximo turno
        if informe.jugadores_detectados:
            for p in reversed(informe.jugadores_detectados):
                if p in st.session_state.last_players:
                    st.session_state.last_players.remove(p)
                st.session_state.last_players.insert(0, p)
            st.session_state.last_players = st.session_state.last_players[:2]

        if informe.rag_players:
            st.session_state.last_rag_results = informe.rag_players

        _render_informe(
            informe,
            stream_fn=lambda: stream_conclusion(query, informe, callbacks=callbacks),
        )

    st.session_state.messages.append({"role": "assistant", "informe": informe})
    st.rerun()
