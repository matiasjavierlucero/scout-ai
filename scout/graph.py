"""
Paso 5: Orquestador con LangGraph.

Graph con tres nodos de agentes que pueden correr en paralelo via Send API.
El orquestador usa heurísticas para decidir qué agentes invocar según la query.

Flujo:
  START → orchestrator → [rag_node, stats_node, comp_node] (paralelo) → synthesis → END

Heurísticas de routing:
  - Palabras clave de comparación → needs_comp
  - Token de nombre de jugador en la query → needs_stats
  - Nada de lo anterior → needs_rag (búsqueda por perfil descriptivo)
"""

import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from scout.agents import run_comp_agent, run_rag_agent, run_stats_agent

# ── Stopwords que ignoramos al detectar nombres de jugadores ─────────────────

_STOPWORDS = {
    "las", "los", "del", "con", "para", "una", "uno", "que", "dame",
    "sus", "stats", "datos", "sobre", "entre", "como", "cuál", "cuales",
    "mejores", "peores", "este", "esta", "estos", "compara", "quiero",
    "busca", "encontrá", "dame", "mostrame", "versus", "juega", "juegan",
}

# ── Estado compartido del graph ──────────────────────────────────────────────

class ScoutState(TypedDict):
    query: str
    routing: dict  # {"needs_rag": bool, "needs_stats": bool, "needs_comp": bool}
    # Annotated con operator.add permite que nodos paralelos hagan append
    # sin pisarse — cada nodo agrega su resultado a la lista
    rag_results:   Annotated[list[str], operator.add]
    stats_results: Annotated[list[str], operator.add]
    comp_results:  Annotated[list[str], operator.add]
    final_report:  str


# ── Heurísticas de routing ───────────────────────────────────────────────────

_COMP_KEYWORDS = {
    "compará", "compara", "comparación", "versus", "vs", "contra",
    "diferencia", "mejor que", "peor que", "frente a",
}


def _load_index_names() -> list[str]:
    import json
    from pathlib import Path
    index_path = Path(__file__).parent.parent / "data" / "index.json"
    with open(index_path, encoding="utf-8") as f:
        return list(json.load(f).keys())


_INDEX_NAMES = _load_index_names()
print(f"[graph] Índice de nombres cargado — {len(_INDEX_NAMES)} jugadores")


def _detect_routing(query: str) -> dict[str, bool]:
    """
    Analiza la query con heurísticas para decidir qué agentes invocar.

    Orden de precedencia:
    1. Palabras clave de comparación → needs_comp + needs_stats
    2. Token de nombre de jugador detectado → needs_stats
    3. Sin coincidencias → needs_rag (query descriptiva de perfil)

    Returns:
        dict con needs_rag, needs_stats, needs_comp como booleans.
    """
    query_lower = query.lower()
    query_tokens = {
        t for t in query_lower.split()
        if len(t) > 3 and t not in _STOPWORDS
    }

    print(f"[routing] Tokens significativos: {query_tokens}")

    # 1. Detectar intención de comparación
    needs_comp = any(kw in query_lower for kw in _COMP_KEYWORDS)
    if needs_comp:
        print("[routing] → needs_comp (keyword de comparación detectada)")

    # 2. Detectar nombres de jugadores en la query
    matched_names = []
    for name in _INDEX_NAMES:
        name_tokens = set(name.lower().split())
        if query_tokens & name_tokens:
            matched_names.append(name)
        if len(matched_names) >= 2:
            break

    print(f"[routing] Nombres detectados: {matched_names}")

    needs_stats = len(matched_names) >= 1
    if len(matched_names) >= 2:
        needs_comp = True

    # 3. Si no hay nada concreto → búsqueda semántica
    needs_rag = not needs_stats and not needs_comp

    routing = {"needs_rag": needs_rag, "needs_stats": needs_stats, "needs_comp": needs_comp}
    print(f"[routing] Decisión final: {routing}")
    return routing


# ── Nodos del graph ──────────────────────────────────────────────────────────

def orchestrator_node(state: ScoutState) -> dict:
    """
    Nodo inicial. Analiza la query con heurísticas y guarda la decisión en el estado.
    La función de edge `route_to_agents` lee esa decisión y despacha los Send.
    En LangGraph 1.x los nodos siempre devuelven dict — los Send van en el edge.
    """
    print(f"\n[orchestrator] Query recibida: '{state['query']}'")
    routing = _detect_routing(state["query"])
    return {"routing": routing}


def route_to_agents(state: ScoutState) -> list[Send]:
    """
    Función de edge condicional. Lee la decisión de routing del estado
    y devuelve los Send objects — LangGraph los ejecuta en paralelo.
    """
    routing = state["routing"]
    sends = []
    if routing["needs_rag"]:
        print("[orchestrator] → despachando rag_node")
        sends.append(Send("rag_node", state))
    if routing["needs_stats"]:
        print("[orchestrator] → despachando stats_node")
        sends.append(Send("stats_node", state))
    if routing["needs_comp"]:
        print("[orchestrator] → despachando comp_node")
        sends.append(Send("comp_node", state))
    return sends


def rag_node(state: ScoutState) -> dict:
    """Nodo del agente RAG. Ejecuta búsqueda semántica y devuelve resultado."""
    print(f"\n[rag_node] Ejecutando...")
    result = run_rag_agent(state["query"])
    return {"rag_results": [result]}


def stats_node(state: ScoutState) -> dict:
    """Nodo del agente Stats. Busca y devuelve stats del jugador mencionado."""
    print(f"\n[stats_node] Ejecutando...")
    result = run_stats_agent(state["query"])
    return {"stats_results": [result]}


def comp_node(state: ScoutState) -> dict:
    """Nodo del agente Comp. Compara dos jugadores mencionados en la query."""
    print(f"\n[comp_node] Ejecutando...")
    result = run_comp_agent(state["query"])
    return {"comp_results": [result]}


def synthesis_node(state: ScoutState) -> dict:
    """
    Nodo final. Combina los resultados de todos los agentes que corrieron
    y genera el informe final estructurado en texto.
    """
    print(f"\n[synthesis] Combinando resultados...")

    sections = []

    if state.get("rag_results"):
        print("[synthesis] → incluyendo resultado RAG")
        sections.append(f"## Jugadores sugeridos por perfil\n\n{state['rag_results'][0]}")

    if state.get("stats_results"):
        print("[synthesis] → incluyendo resultado Stats")
        sections.append(f"## Estadísticas\n\n{state['stats_results'][0]}")

    if state.get("comp_results"):
        print("[synthesis] → incluyendo resultado Comparativa")
        sections.append(f"## Comparativa\n\n{state['comp_results'][0]}")

    final_report = "\n\n---\n\n".join(sections)
    print(f"[synthesis] Informe generado ({len(final_report)} chars)")

    return {"final_report": final_report}


# ── Compilación del graph ────────────────────────────────────────────────────

def build_graph():
    """Construye y compila el graph con MemorySaver checkpointer."""
    workflow = StateGraph(ScoutState)

    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("rag_node", rag_node)
    workflow.add_node("stats_node", stats_node)
    workflow.add_node("comp_node", comp_node)
    workflow.add_node("synthesis", synthesis_node)

    workflow.add_edge(START, "orchestrator")

    # route_to_agents lee el estado y devuelve Send objects → LangGraph los ejecuta en paralelo
    workflow.add_conditional_edges(
        "orchestrator",
        route_to_agents,
        ["rag_node", "stats_node", "comp_node"],
    )

    # cada agente converge en synthesis
    workflow.add_edge("rag_node", "synthesis")
    workflow.add_edge("stats_node", "synthesis")
    workflow.add_edge("comp_node", "synthesis")
    workflow.add_edge("synthesis", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


graph = build_graph()
print("[graph] Graph compilado y listo.\n")


# ── API pública ──────────────────────────────────────────────────────────────

def run(query: str, thread_id: str = "default") -> str:
    """
    Punto de entrada principal. Ejecuta el graph completo para una query.

    Args:
        query: Pregunta o descripción del usuario en lenguaje natural.
        thread_id: ID de conversación para MemorySaver (memoria entre queries).

    Returns:
        Informe final como string.
    """
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "query": query,
        "routing": {},
        "rag_results": [],
        "stats_results": [],
        "comp_results": [],
        "final_report": "",
    }

    print(f"\n{'='*60}")
    print(f"SCOUT AI — Query: {query}")
    print(f"{'='*60}")

    result = graph.invoke(initial_state, config=config)
    return result["final_report"]
