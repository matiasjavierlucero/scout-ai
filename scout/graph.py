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
import re
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send
from langgraph.graph import END, START, StateGraph

from scout.agents import run_comp_agent, run_rag_agent, run_stats_agent
from scout.llm import make_llm
from scout.tools import _name_to_minutes
from scout.schemas import InformeScouting

_conclusion_llm = make_llm(temperature=0.3)

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
    rag_results:        Annotated[list[str], operator.add]
    stats_results:      Annotated[list[str], operator.add]
    comp_results:       Annotated[list[str], operator.add]
    final_report:       str
    jugadores_detectados: list[str]  # jugadores resueltos en el orchestrator
    rag_players:          list[str]  # jugadores del resultado RAG en orden (para guardar en app)
    context_players:    list[str]    # jugadores del turno anterior (inyectado desde app)
    rag_context:        list[str]    # jugadores del último RAG del turno anterior (inyectado desde app)


# ── Heurísticas de routing ───────────────────────────────────────────────────

_COMP_KEYWORDS = {
    "compará", "compara", "comparación", "versus", "vs", "contra",
    "diferencia", "mejor que", "peor que", "frente a",
}

# Mapa de palabras ordinales → índice 0-based en la lista de resultados RAG.
# Permite resolver "el primer jugador que me ofreciste" → rag_context[0].
_ORDINAL_MAP: dict[str, int] = {
    "primero": 0, "primer": 0, "primera": 0,
    "segundo": 1, "segunda": 1,
    "tercero": 2, "tercera": 2,
    "cuarto":  3, "cuarta":  3,
    "quinto":  4, "quinta":  4,
}

# Palabras que indican intención de ver estadísticas de un jugador específico,
# usadas junto con la detección ordinal para evitar falsos positivos.
_STATS_INTENT_KEYWORDS = {
    "estadísticas", "estadisticas", "stats", "métricas", "metricas",
    "rendimiento", "datos", "números", "numeros", "información", "info",
    "háblame", "hablame", "cuéntame", "cuentame",
}


def _extract_rag_player_names(rag_text: str) -> list[str]:
    """
    Extrae los nombres de jugadores del output de buscar_jugadores en orden de similitud.

    El formato del output es:
      "  1. Ikechukwu Uche — Right Center Forward — similitud: 0.452"
      "  2. Ustaritz Aldekoaotalora Astarloa — Right Center Back — similitud: 0.450"

    Devuelve una lista de nombres ["Ikechukwu Uche", "Ustaritz Aldekoaotalora Astarloa", ...]
    para guardar como contexto y resolver referencias ordinales en el turno siguiente.
    """
    pattern = re.compile(r"^\s+\d+\.\s+(.+?)\s+—")
    names = []
    for line in rag_text.splitlines():
        match = pattern.match(line)
        if match:
            names.append(match.group(1).strip())
    return names


def _resolve_ordinal_reference(query: str, rag_context: list[str]) -> str | None:
    """
    Resuelve referencias del tipo "el primer jugador que me ofreciste" → nombre real.

    Condiciones para activar la resolución:
      1. La query contiene una palabra ordinal del _ORDINAL_MAP.
      2. La query contiene al menos una palabra de intención de stats (_STATS_INTENT_KEYWORDS).
      3. El índice ordinal existe dentro de rag_context.

    Exigir ambas condiciones evita falsos positivos: "el primero era muy rápido,
    dame otro similar" tiene ordinal pero NO intención de stats → sigue siendo RAG.

    Returns:
        Nombre del jugador resuelto, o None si no se cumplen las condiciones.
    """
    if not rag_context:
        return None

    query_lower = query.lower()

    has_stats_intent = any(kw in query_lower for kw in _STATS_INTENT_KEYWORDS)
    if not has_stats_intent:
        return None

    for term, idx in _ORDINAL_MAP.items():
        if term in query_lower and idx < len(rag_context):
            return rag_context[idx]

    return None


def _load_index_names() -> list[str]:
    import json
    from pathlib import Path
    index_path = Path(__file__).parent.parent / "data" / "index.json"
    with open(index_path, encoding="utf-8") as f:
        return list(json.load(f).keys())


_INDEX_NAMES = _load_index_names()
print(f"[graph] Índice de nombres cargado — {len(_INDEX_NAMES)} jugadores")


def _detect_routing(query: str) -> tuple[dict[str, bool], list[str]]:
    """
    Analiza la query con heurísticas para decidir qué agentes invocar.

    Returns:
        (routing_dict, distinct_players) — jugadores verdaderamente distintos detectados.

    Un jugador es "distinto" si aporta tokens únicos que ningún match anterior cubre.
    Esto evita que "Lionel Messi" detecte a Siviero (solo comparte "lionel") como
    segundo jugador y dispare una comparación falsa.
    """
    query_lower = query.lower()

    # Limpiar puntuación de cada token antes de filtrar (evita "ronaldo?" ≠ "ronaldo")
    query_tokens = set()
    for t in query_lower.split():
        clean = t.strip("?!.,;:'\"")
        if len(clean) > 3 and clean not in _STOPWORDS:
            query_tokens.add(clean)

    print(f"[routing] Tokens significativos: {query_tokens}")

    # 1. Detectar intención de comparación por palabras clave
    needs_comp = any(kw in query_lower for kw in _COMP_KEYWORDS)
    if needs_comp:
        print("[routing] → needs_comp (keyword de comparación detectada)")

    # 2. Recolectar matches con cuántos tokens coinciden
    scored_matches = []
    for name in _INDEX_NAMES:
        name_tokens = set(name.lower().split())
        overlapping = query_tokens & name_tokens
        if overlapping:
            scored_matches.append((name, overlapping))

    # Ordenar por: (tokens en común DESC, minutos jugados DESC)
    # El desempate por minutos resuelve "Cristiano" → Ronaldo > Biraghi
    scored_matches.sort(
        key=lambda x: (len(x[1]), _name_to_minutes.get(x[0], 0)),
        reverse=True,
    )

    print(f"[routing] Top matches: {[(n, t) for n, t in scored_matches[:3]]}")

    # 3. Extraer jugadores verdaderamente distintos (cada uno aporta tokens nuevos)
    distinct_players: list[str] = []
    covered_tokens: set[str] = set()
    for name, tokens in scored_matches:
        unique = tokens - covered_tokens
        if unique:  # solo agrega si tiene tokens que ningún match anterior cubre
            distinct_players.append(name)
            covered_tokens |= tokens
        if len(distinct_players) >= 2:
            break

    print(f"[routing] Jugadores distintos: {distinct_players}")

    needs_stats = len(distinct_players) >= 1
    if not needs_comp and len(distinct_players) >= 2:
        needs_comp = True
        print(f"[routing] → needs_comp (dos jugadores con tokens distintos)")

    # 4. Sin jugadores → búsqueda semántica
    needs_rag = not needs_stats and not needs_comp

    routing = {"needs_rag": needs_rag, "needs_stats": needs_stats, "needs_comp": needs_comp}
    print(f"[routing] Decisión final: {routing}")
    return routing, distinct_players


# ── Nodos del graph ──────────────────────────────────────────────────────────

def orchestrator_node(state: ScoutState) -> dict:
    """
    Nodo inicial. Analiza la query, resuelve jugadores y guarda la decisión en el estado.

    Resolución de referencias en orden de prioridad:

    1. Referencia ordinal + intención de stats ("el primer jugador que me ofreciste"):
       Si la query no contiene nombres pero tiene un ordinal ("primer", "segundo"...)
       y palabras de stats ("estadísticas", "info", etc.), se resuelve contra rag_context
       (los jugadores del último resultado RAG) y se redirige a stats_node.

    2. Comparación con contexto ("comparalo con Ronaldo", "comparalos"):
       Si needs_comp=True pero se detectó ≤1 jugador, se completa con context_players
       (jugadores del turno anterior, cualquier tipo de respuesta).
    """
    print(f"\n[orchestrator] Query recibida: '{state['query']}'")
    routing, players = _detect_routing(state["query"])

    rag_context = state.get("rag_context", [])
    context = state.get("context_players", [])

    # ── Resolución 1: referencia ordinal al resultado RAG anterior ────────────
    if routing["needs_rag"] and rag_context:
        resolved = _resolve_ordinal_reference(state["query"], rag_context)
        if resolved:
            players = [resolved]
            routing = {"needs_rag": False, "needs_stats": True, "needs_comp": False}
            print(f"[orchestrator] Referencia ordinal resuelta → '{resolved}'")

    # ── Resolución 2: comparación con contexto conversacional ─────────────────
    if routing["needs_comp"] and len(players) <= 1 and context:
        if len(players) == 1:
            players = [context[0]] + players
        else:
            players = context[:2]
        print(f"[orchestrator] Contexto inyectado: {context[:2]}")

    return {"routing": routing, "jugadores_detectados": players}


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
    if routing["needs_stats"] and not routing["needs_comp"]:
        # Cuando hay comparación, la tabla ya incluye las stats de ambos jugadores
        print("[orchestrator] → despachando stats_node")
        sends.append(Send("stats_node", state))
    if routing["needs_comp"]:
        print("[orchestrator] → despachando comp_node")
        sends.append(Send("comp_node", state))
    return sends


def rag_node(state: ScoutState) -> dict:
    """
    Nodo del agente RAG. Ejecuta búsqueda semántica y devuelve resultado.

    Además del texto, extrae los nombres de los jugadores en orden para guardarlos
    como rag_players en el estado. La app los persiste en last_rag_results para que
    el orchestrator pueda resolver referencias ordinales en el turno siguiente.
    """
    print(f"\n[rag_node] Ejecutando...")
    result = run_rag_agent(state["query"])
    player_names = _extract_rag_player_names(result)
    print(f"[rag_node] Jugadores extraídos del resultado: {player_names}")
    return {"rag_results": [result], "rag_players": player_names}


def stats_node(state: ScoutState) -> dict:
    """Nodo del agente Stats. Usa el jugador ya resuelto por el orchestrator."""
    print(f"\n[stats_node] Ejecutando...")
    players = state.get("jugadores_detectados", [])
    query = f"Dame las stats de {players[0]}" if players else state["query"]
    print(f"[stats_node] Query: '{query}'")
    result = run_stats_agent(query)
    return {"stats_results": [result]}


def comp_node(state: ScoutState) -> dict:
    """Nodo del agente Comp. Usa los dos jugadores ya resueltos por el orchestrator."""
    print(f"\n[comp_node] Ejecutando...")
    players = state.get("jugadores_detectados", [])
    if len(players) >= 2:
        query = f"Compará a {players[0]} con {players[1]}"
    else:
        query = state["query"]
    print(f"[comp_node] Query: '{query}'")
    result = run_comp_agent(query)
    return {"comp_results": [result]}


def _build_conclusion_messages(query: str, sections: dict) -> list:
    """
    Construye los mensajes para la llamada al LLM de conclusión.

    Extraído para ser reutilizable tanto por la versión bloqueante (_generate_conclusion)
    como por la versión streaming (stream_conclusion). El prompt es corto y directo
    para que llama3.2:3b lo siga sin inventar datos.
    """
    context_parts = []
    if sections.get("rag"):
        context_parts.append(f"Jugadores sugeridos:\n{sections['rag']}")
    if sections.get("stats"):
        context_parts.append(f"Estadísticas:\n{sections['stats']}")
    if sections.get("comp"):
        context_parts.append(f"Comparativa:\n{sections['comp']}")

    return [
        SystemMessage(
            "Sos un analista de scouting. Escribí una conclusión de 2 a 3 oraciones "
            "basada SOLO en los datos que se te dan. No inventes estadísticas. "
            "Sé directo y concreto."
        ),
        HumanMessage(
            f"Query del usuario: {query}\n\n"
            f"Datos disponibles:\n{'\n\n'.join(context_parts)}\n\n"
            "Conclusión:"
        ),
    ]


def stream_conclusion(query: str, informe: "InformeScouting", callbacks: list | None = None):
    """
    Generator que produce la conclusión token a token usando streaming del LLM.

    Diseñado para ser pasado directamente a st.write_stream() de Streamlit:

        conclusion_text = st.write_stream(stream_conclusion(query, informe))

    El LLM recibe el mismo contexto que tendría con invoke() — la diferencia es
    que los tokens llegan en tiempo real en vez de esperar a que la generación
    complete. Con llama3.2:3b, esto convierte ~8s de silencio en texto apareciendo
    palabra a palabra.

    Args:
        query: Query original del usuario (para contexto del prompt).
        informe: InformeScouting con los campos ya populados por el graph
                 (jugadores_sugeridos, estadisticas, comparativa).
        callbacks: Langfuse/LangChain callbacks para tracing. None = sin tracing.

    Yields:
        Fragmentos de texto (str) a medida que el LLM los genera.
    """
    sections = {
        "rag": informe.jugadores_sugeridos,
        "stats": informe.estadisticas,
        "comp": informe.comparativa,
    }
    messages = _build_conclusion_messages(query, sections)
    config = {"callbacks": callbacks} if callbacks else {}

    print("[conclusion] Iniciando streaming...")
    for chunk in _conclusion_llm.stream(messages, config=config):
        if chunk.content:
            yield chunk.content
    print("[conclusion] Streaming completado")


def synthesis_node(state: ScoutState) -> dict:
    """
    Nodo final. Construye un InformeScouting con los resultados de los agentes.

    La conclusión NO se genera aquí — queda vacía y es responsabilidad de la capa
    de presentación (app.py) generarla via stream_conclusion(). Esto permite que
    el graph termine rápido y la UI muestre stats/comp/rag inmediatamente, mientras
    la conclusión aparece token a token en paralelo.
    """
    print(f"\n[synthesis] Construyendo InformeScouting...")

    agentes = []
    sections = {}

    if state.get("rag_results"):
        print("[synthesis] → incluyendo resultado RAG")
        agentes.append("rag")
        sections["rag"] = state["rag_results"][0]

    if state.get("stats_results"):
        print("[synthesis] → incluyendo resultado Stats")
        agentes.append("stats")
        sections["stats"] = state["stats_results"][0]

    if state.get("comp_results"):
        print("[synthesis] → incluyendo resultado Comp")
        agentes.append("comp")
        sections["comp"] = state["comp_results"][0]

    informe = InformeScouting(
        query=state["query"],
        agentes_ejecutados=agentes,
        jugadores_sugeridos=sections.get("rag", ""),
        estadisticas=sections.get("stats", ""),
        comparativa=sections.get("comp", ""),
        conclusion="",  # populado por stream_conclusion() en app.py
        jugadores_detectados=state.get("jugadores_detectados", []),
        rag_players=state.get("rag_players", []),
    )

    print(f"[synthesis] InformeScouting construido (sin conclusión)")
    return {"final_report": informe.model_dump_json()}


# ── Compilación del graph ────────────────────────────────────────────────────

def build_graph():
    """
    Construye y compila el graph SIN checkpointer.

    No usamos MemorySaver porque cada query debe ser completamente stateless —
    el contexto multi-turno está manejado explícitamente vía context_players en
    initial_state. Con MemorySaver + operator.add, los resultados de queries
    anteriores (comp_results, stats_results) se acumulan y contaminan queries
    nuevas, haciendo aparecer comparativas donde no corresponde.
    """
    workflow = StateGraph(ScoutState)

    # add_node registra una función como nodo del graph, identificado por un nombre (string).
    # Acá solo DECLARAMOS qué nodos existen — todavía no definimos cómo se conectan entre sí.
    # Cada función nodo recibe el ScoutState y devuelve un dict parcial que se mergea al estado.
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("rag_node", rag_node)
    workflow.add_node("stats_node", stats_node)
    workflow.add_node("comp_node", comp_node)
    workflow.add_node("synthesis", synthesis_node)

    # add_edge conecta dos nodos con una transición FIJA e incondicional: A → B, siempre.
    # No hay lógica de por medio — apenas termina START, se ejecuta "orchestrator" sin excepción.
    workflow.add_edge(START, "orchestrator")

    # add_conditional_edges conecta un nodo a MÚLTIPLES destinos posibles, decididos en runtime
    # por una función de routing (acá route_to_agents). A diferencia de add_edge (transición
    # única y fija), esta función lee el estado y decide a qué nodo(s) ir — es el equivalente
    # a un branch condicional dentro del graph.
    # route_to_agents devuelve Send objects → LangGraph dispara esos nodos en PARALELO
    # (podés mandar el mismo nodo con distinto payload varias veces, o un subconjunto de ellos).
    # El tercer argumento (["rag_node", "stats_node", "comp_node"]) es la lista de destinos
    # posibles — LangGraph la usa para validar el graph y dibujarlo, no filtra el routing real.
    workflow.add_conditional_edges(
        "orchestrator",
        route_to_agents,
        ["rag_node", "stats_node", "comp_node"],
    )

    # cada agente converge en synthesis → de nuevo add_edge, porque acá no hay decisión:
    # sin importar cuál agente corrió, todos van a synthesis sí o sí.
    workflow.add_edge("rag_node", "synthesis")
    workflow.add_edge("stats_node", "synthesis")
    workflow.add_edge("comp_node", "synthesis")
    workflow.add_edge("synthesis", END)

    return workflow.compile()


graph = build_graph()
print("[graph] Graph compilado y listo.\n")


# ── API pública ──────────────────────────────────────────────────────────────

def run(
    query: str,
    context_players: list[str] | None = None,
    rag_context: list[str] | None = None,
    callbacks: list | None = None,
) -> InformeScouting:
    """
    Punto de entrada principal. Ejecuta el graph completo para una query.

    Cada invocación es completamente stateless — el graph no retiene nada entre
    llamadas. El contexto multi-turno (jugadores del turno anterior) se pasa
    explícitamente via context_players desde la capa de UI.

    Args:
        query: Pregunta o descripción del usuario en lenguaje natural.
        context_players: Jugadores del turno anterior para resolver follow-ups
                         como "comparalo con Ronaldo" o "y sus stats?".
        rag_context: Resultado RAG del turno anterior para resolver referencias
                     ordinales ("el primer jugador que me ofreciste").
        callbacks: Langfuse/LangChain callbacks para tracing. Cuando se pasa
                   un CallbackHandler de Langfuse, el trace incluye todos los
                   spans internos del graph (orchestrator, agentes, tools) de
                   forma automática — no hay instrumentación manual necesaria.

    Returns:
        InformeScouting con todos los campos estructurados.
    """
    initial_state = {
        "query": query,
        "routing": {},
        "rag_results": [],
        "stats_results": [],
        "comp_results": [],
        "final_report": "",
        "jugadores_detectados": [],
        "rag_players": [],
        "context_players": context_players or [],
        "rag_context": rag_context or [],
    }

    print(f"\n{'='*60}")
    print(f"SCOUT AI — Query: {query}")
    print(f"{'='*60}")

    config = {"callbacks": callbacks} if callbacks else {}
    result = graph.invoke(initial_state, config=config)
    return InformeScouting.model_validate_json(result["final_report"])
