"""
Visualizaciones interactivas para Scout AI.

Genera figuras Plotly para:
  - Radar chart de stats por 90 min (individual y comparativa)
  - Campo de fútbol con zona de posición resaltada
  - Datos estructurados de jugadores para los cards del dashboard
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Paleta (coherente con el dark theme de app.py) ────────────────────────────

_C = {
    "bg":          "#0d0f14",
    "card":        "#13161e",
    "border":      "#1f2330",
    "text":        "#e2e8f0",
    "muted":       "#64748b",
    "blue":        "#3b82f6",    # jugador A
    "orange":      "#f97316",    # jugador B
    "pitch":       "#1a3d19",
    "pitch_line":  "rgba(255,255,255,0.80)",
    "zone_fill":   "rgba(59,130,246,0.20)",
    "zone_border": "rgba(59,130,246,0.60)",
}

# ── Métricas del radar (columna CSV → etiqueta del eje) ───────────────────────

RADAR_METRICS: list[tuple[str, str]] = [
    ("passes_completed_p90", "Pases"),
    ("xg_p90",               "xG"),
    ("shots_p90",            "Tiros"),
    ("ball_recoveries_p90",  "Recuperaciones"),
    ("pressures_p90",        "Presiones"),
    ("dribbles_won_p90",     "Regates"),
]

# ── Coordenadas de posición en el campo (x: 0-100, y: 0-68) ──────────────────
# x crece de defensa (izquierda) a ataque (derecha)

POSITION_COORDS: dict[str, tuple[float, float]] = {
    "Goalkeeper":                (8,  34),
    "Right Back":                (28, 58),  "Left Back":                (28, 10),
    "Right Center Back":         (22, 48),  "Left Center Back":         (22, 20),
    "Center Back":               (22, 34),
    "Right Wing Back":           (40, 62),  "Left Wing Back":           (40,  6),
    "Right Midfield":            (52, 60),  "Left Midfield":            (52,  8),
    "Right Center Midfield":     (50, 47),  "Left Center Midfield":     (50, 21),
    "Center Midfield":           (50, 34),
    "Right Defensive Midfield":  (40, 52),  "Left Defensive Midfield":  (40, 16),
    "Center Defensive Midfield": (40, 34),
    "Right Attacking Midfield":  (65, 52),  "Left Attacking Midfield":  (65, 16),
    "Center Attacking Midfield": (65, 34),
    "Right Wing":                (78, 62),  "Left Wing":                (78,  6),
    "Right Center Forward":      (82, 48),  "Left Center Forward":      (82, 20),
    "Center Forward":            (85, 34),
    "Secondary Striker":         (75, 34),
}

# ── Carga lazy del DataFrame ──────────────────────────────────────────────────

_df: pd.DataFrame | None = None
_mins: dict[str, float] = {}
_maxs: dict[str, float] = {}


def _get_df() -> pd.DataFrame:
    global _df, _mins, _maxs
    if _df is None:
        _df = pd.read_csv(DATA_DIR / "player_stats.csv")
        for col, _ in RADAR_METRICS:
            _mins[col] = float(_df[col].min())
            _maxs[col] = float(_df[col].max())
    return _df


def _pct(value: float, col: str) -> float:
    """Normaliza un valor a 0-100 relativo al rango del dataset."""
    lo, hi = _mins.get(col, 0), _maxs.get(col, 1)
    if hi == lo:
        return 0.0
    return round(max(0.0, min(100.0, (value - lo) / (hi - lo) * 100)), 1)


# ── API pública ───────────────────────────────────────────────────────────────

def get_player_data(name: str) -> dict | None:
    """
    Devuelve un dict con los datos de un jugador del CSV.

    Usa la misma lógica de matching que tools.py (token-match primero,
    fuzzy de caracteres como fallback) para garantizar que "Busquets"
    encuentre "Sergio Busquets Burgos" igual que el agente Stats.

    Returns:
        Dict con name, position, minutes_played y todas las métricas p90,
        o None si el jugador no se encuentra.
    """
    from scout.tools import _find_candidates

    df = _get_df()

    # Match exacto primero (evita llamar _find_candidates innecesariamente)
    row = df[df["name"] == name]

    if row.empty:
        candidates = _find_candidates(name, k=1)
        if not candidates:
            return None
        row = df[df["name"] == candidates[0]["name"]]

    if row.empty:
        return None

    r = row.iloc[0]
    return {
        "name":                  r["name"],
        "position":              r["position"],
        "minutes_played":        int(r["minutes_played"]),
        "passes_completed_p90":  float(r["passes_completed_p90"]),
        "passes_total_p90":      float(r["passes_total_p90"]),
        "shots_p90":             float(r["shots_p90"]),
        "xg_p90":                float(r["xg_p90"]),
        "ball_recoveries_p90":   float(r["ball_recoveries_p90"]),
        "pressures_p90":         float(r["pressures_p90"]),
        "dribbles_won_p90":      float(r["dribbles_won_p90"]),
    }


def build_radar(
    data_a: dict,
    data_b: dict | None = None,
) -> go.Figure:
    """
    Radar chart (spider web) con las 6 métricas por 90 normalizadas a 0-100.

    Un punto en el percentil 80 significa que el jugador supera al 80 % del
    dataset en esa métrica. Si se provee data_b, superpone un segundo trazado
    en naranja para comparación visual.

    Args:
        data_a: Dict de get_player_data() para el jugador principal (azul).
        data_b: Dict de get_player_data() para el segundo jugador (naranja).

    Returns:
        go.Figure lista para pasar a st.plotly_chart(use_container_width=True).
    """
    cats = [label for _, label in RADAR_METRICS]
    cats_closed = cats + [cats[0]]

    def _trace(data: dict, color: str, fill_color: str) -> go.Scatterpolar:
        vals = [_pct(data[col], col) for col, _ in RADAR_METRICS]
        raw  = [data[col] for col, _ in RADAR_METRICS]
        return go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats_closed,
            fill="toself",
            fillcolor=fill_color,
            line=dict(color=color, width=2.5),
            name=data["name"].split()[0],
            customdata=[[f"{v:.2f}" for v in raw + [raw[0]]]],
            hovertemplate=(
                "<b>%{theta}</b><br>"
                "Percentil: %{r:.0f}/100<br>"
                "Real: %{customdata[0]}"
                "<extra></extra>"
            ),
        )

    fig = go.Figure()
    fig.add_trace(_trace(data_a, _C["blue"], "rgba(59,130,246,0.18)"))
    if data_b:
        fig.add_trace(_trace(data_b, _C["orange"], "rgba(249,115,22,0.15)"))

    fig.update_layout(
        polar=dict(
            bgcolor=_C["card"],
            radialaxis=dict(
                range=[0, 100],
                showticklabels=False,
                gridcolor=_C["border"],
                linecolor=_C["border"],
            ),
            angularaxis=dict(
                gridcolor=_C["border"],
                linecolor=_C["border"],
                tickfont=dict(color=_C["text"], size=11),
            ),
        ),
        paper_bgcolor=_C["bg"],
        showlegend=bool(data_b),
        legend=dict(font=dict(color=_C["text"]), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=40, t=20, b=20),
        height=310,
    )
    return fig


def build_pitch(position: str) -> go.Figure:
    """
    Campo de fútbol simplificado con la zona de la posición resaltada.

    El campo se orienta horizontalmente (defensa izquierda → ataque derecha).
    Las coordenadas son 0-100 en x y 0-68 en y (estándar StatsBomb).

    Args:
        position: Posición del jugador (e.g., "Right Wing", "Center Forward").

    Returns:
        go.Figure con el campo y un marcador circular en la posición del jugador.
    """
    px, py = POSITION_COORDS.get(position, (50, 34))

    shapes = [
        # Campo completo
        dict(type="rect", x0=0, y0=0, x1=100, y1=68,
             fillcolor=_C["pitch"], line=dict(color=_C["pitch_line"], width=1.5)),
        # Línea de medio campo
        dict(type="line", x0=50, y0=0, x1=50, y1=68,
             line=dict(color=_C["pitch_line"], width=1.5)),
        # Círculo central (radio ≈ 9.15m sobre 105m de largo = 8.7%)
        dict(type="circle", x0=41, y0=25.5, x1=59, y1=42.5,
             line=dict(color=_C["pitch_line"], width=1.5), fillcolor="rgba(0,0,0,0)"),
        # Punto central
        dict(type="circle", x0=49.4, y0=33.4, x1=50.6, y1=34.6,
             fillcolor=_C["pitch_line"], line=dict(color=_C["pitch_line"], width=1)),
        # Área penal izquierda (16.5m ≈ 15.7% del largo)
        dict(type="rect", x0=0, y0=13.84, x1=16.5, y1=54.16,
             line=dict(color=_C["pitch_line"], width=1.5), fillcolor="rgba(0,0,0,0)"),
        # Área chica izquierda
        dict(type="rect", x0=0, y0=24.84, x1=5.5, y1=43.16,
             line=dict(color=_C["pitch_line"], width=1.5), fillcolor="rgba(0,0,0,0)"),
        # Portería izquierda
        dict(type="rect", x0=-2, y0=29.68, x1=0, y1=38.32,
             fillcolor="rgba(255,255,255,0.25)", line=dict(color=_C["pitch_line"], width=1.5)),
        # Área penal derecha
        dict(type="rect", x0=83.5, y0=13.84, x1=100, y1=54.16,
             line=dict(color=_C["pitch_line"], width=1.5), fillcolor="rgba(0,0,0,0)"),
        # Área chica derecha
        dict(type="rect", x0=94.5, y0=24.84, x1=100, y1=43.16,
             line=dict(color=_C["pitch_line"], width=1.5), fillcolor="rgba(0,0,0,0)"),
        # Portería derecha
        dict(type="rect", x0=100, y0=29.68, x1=102, y1=38.32,
             fillcolor="rgba(255,255,255,0.25)", line=dict(color=_C["pitch_line"], width=1.5)),
        # Zona de influencia del jugador
        dict(type="circle",
             x0=px - 9, y0=py - 7, x1=px + 9, y1=py + 7,
             fillcolor=_C["zone_fill"],
             line=dict(color=_C["zone_border"], width=1, dash="dot")),
    ]

    fig = go.Figure()
    fig.update_layout(shapes=shapes)

    # Marcador del jugador
    fig.add_trace(go.Scatter(
        x=[px], y=[py],
        mode="markers+text",
        marker=dict(size=16, color=_C["blue"], line=dict(color="white", width=2.5)),
        text=["●"], textposition="middle center",
        hovertext=[position],
        hovertemplate="%{hovertext}<extra></extra>",
        showlegend=False,
    ))

    fig.update_layout(
        paper_bgcolor=_C["bg"],
        xaxis=dict(range=[-3, 103], visible=False),
        yaxis=dict(range=[-3, 71], visible=False, scaleanchor="x", scaleratio=0.68),
        margin=dict(l=0, r=0, t=8, b=0),
        height=210,
    )
    return fig
