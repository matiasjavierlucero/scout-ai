"""
Golden set de evaluación — 10 queries con ground truth manual.

Cubre los tres tipos de routing del orquestador:
  - RAG (búsqueda semántica por perfil descriptivo)
  - Stats (stats de un jugador por nombre)
  - Comp (comparación de dos jugadores)

Ground truth no es el texto exacto de la respuesta — es un conjunto
de aserciones verificables determinísticamente (rule-based sanity)
que una respuesta correcta SIEMPRE debe satisfacer.
"""

from typing import TypedDict


class GoldenCase(TypedDict):
    id: str
    query: str
    tipo: str                        # "rag" | "stats" | "comp"
    expected_agentes: list[str]      # agentes que deben correr
    expected_keywords: list[str]     # palabras que deben aparecer en el resultado
    expected_players: list[str]      # jugadores que deben mencionarse (vacío = no aplica)
    max_response_secs: float         # timeout para la sanity de performance


GOLDEN_SET: list[GoldenCase] = [
    # ── RAG queries ──────────────────────────────────────────────────────────
    {
        "id": "rag-001",
        "query": "quiero un mediocentro defensivo que recupere balones y distribuya bajo presión",
        "tipo": "rag",
        "expected_agentes": ["rag"],
        "expected_keywords": ["Midfielder", "midfield", "similitud"],
        "expected_players": [],
        "max_response_secs": 60.0,
    },
    {
        "id": "rag-002",
        "query": "delantero rápido con remate y desborde por banda",
        "tipo": "rag",
        "expected_agentes": ["rag"],
        "expected_keywords": ["Forward", "similitud"],
        "expected_players": [],
        "max_response_secs": 60.0,
    },
    {
        "id": "rag-003",
        "query": "central que sale jugando y construye desde atrás con pases largos",
        "tipo": "rag",
        "expected_agentes": ["rag"],
        "expected_keywords": ["Back", "similitud"],
        "expected_players": [],
        "max_response_secs": 60.0,
    },
    {
        "id": "rag-004",
        "query": "portero con buenos reflejos y que salga a cortar centros",
        "tipo": "rag",
        "expected_agentes": ["rag"],
        "expected_keywords": ["Goalkeeper", "similitud"],
        "expected_players": [],
        "max_response_secs": 60.0,
    },
    # ── Stats queries ─────────────────────────────────────────────────────────
    {
        "id": "stats-001",
        "query": "dame las stats de Messi",
        "tipo": "stats",
        "expected_agentes": ["stats"],
        "expected_keywords": ["Minutos jugados", "xG", "Pases completados"],
        "expected_players": ["Messi"],
        "max_response_secs": 60.0,
    },
    {
        "id": "stats-002",
        "query": "quiero ver el rendimiento de Busquets",
        "tipo": "stats",
        "expected_agentes": ["stats"],
        "expected_keywords": ["Minutos jugados", "xG", "Presiones"],
        "expected_players": ["Busquets"],
        "max_response_secs": 60.0,
    },
    {
        "id": "stats-003",
        "query": "estadísticas de Cristiano Ronaldo",
        "tipo": "stats",
        "expected_agentes": ["stats"],
        "expected_keywords": ["Minutos jugados", "xG"],
        "expected_players": ["Ronaldo", "Cristiano"],
        "max_response_secs": 60.0,
    },
    # ── Comp queries ──────────────────────────────────────────────────────────
    {
        "id": "comp-001",
        "query": "compará a Messi con Cristiano Ronaldo",
        "tipo": "comp",
        "expected_agentes": ["comp"],
        "expected_keywords": ["xG", "Pases"],
        "expected_players": ["Messi", "Ronaldo"],
        "max_response_secs": 90.0,
    },
    {
        "id": "comp-002",
        "query": "compara a Iniesta con Busquets",
        "tipo": "comp",
        "expected_agentes": ["comp"],
        "expected_keywords": ["xG", "Pases"],
        "expected_players": ["Iniesta", "Busquets"],
        "max_response_secs": 90.0,
    },
    {
        "id": "comp-003",
        "query": "diferencia entre Villa y Torres",
        "tipo": "comp",
        "expected_agentes": ["comp"],
        "expected_keywords": ["xG", "Tiros"],
        "expected_players": ["Villa", "Torres"],
        "max_response_secs": 90.0,
    },
]
