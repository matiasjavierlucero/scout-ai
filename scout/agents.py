"""
Paso 4: Agentes de scouting.

Tres agentes construidos con create_react_agent de LangGraph:
  - rag_agent   — búsqueda semántica por perfil de juego
  - stats_agent — stats completas de un jugador por nombre
  - comp_agent  — comparativa lado a lado entre dos jugadores

Cada agente es un graph compilado que maneja el loop ReAct internamente:
  1. El LLM decide qué tool llamar (o si ya tiene suficiente info)
  2. Se ejecuta la tool
  3. El LLM procesa el resultado y decide si necesita más tools o responde
"""

from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from scout.tools import (
    buscar_jugadores,
    buscar_por_nombre,
    comparar_jugadores,
    stats_jugador,
)

MODEL = "llama3.2:3b"

print(f"[agents] Inicializando modelo {MODEL}...")
_llm = ChatOllama(model=MODEL, temperature=0)
print(f"[agents] Modelo listo.\n")


rag_agent = create_agent(
    model=_llm,
    tools=[buscar_jugadores],
    system_prompt=(
        "Sos un agente de scouting especializado en búsqueda semántica de jugadores. "
        "Recibís descripciones de perfiles de juego en lenguaje natural y usás la tool "
        "buscar_jugadores para encontrar los jugadores más similares en el dataset de La Liga. "
        "Siempre mostrá los resultados de la tool directamente, sin inventar información adicional. "
        "Si el resultado incluye advertencias de muestra pequeña, mencionálas."
    ),
)

stats_agent = create_agent(
    model=_llm,
    tools=[buscar_por_nombre, stats_jugador],
    system_prompt=(
        "Sos un agente de scouting especializado en estadísticas de jugadores. "
        "Cuando recibís un nombre, primero usá buscar_por_nombre para encontrar el nombre "
        "exacto en el dataset, luego usá stats_jugador para obtener sus estadísticas. "
        "Si hay múltiples candidatos, elegí el más relevante según contexto. "
        "Siempre mencioná la advertencia de muestra pequeña si el jugador tiene menos de 450 minutos."
    ),
)

comp_agent = create_agent(
    model=_llm,
    tools=[buscar_por_nombre, comparar_jugadores],
    system_prompt=(
        "Sos un agente de scouting especializado en comparación de jugadores. "
        "Cuando recibís dos nombres, primero resolvé cada nombre con buscar_por_nombre, "
        "luego usá comparar_jugadores con los nombres exactos del dataset. "
        "Presentá la tabla comparativa tal como viene de la tool y agregá una conclusión "
        "breve sobre cuál jugador destaca en cada aspecto. "
        "Mencioná advertencias de muestra pequeña si aplica."
    ),
)


def run_rag_agent(query: str) -> str:
    """Ejecuta el agente RAG con una query de perfil de juego."""
    print(f"\n[rag_agent] Ejecutando con query: '{query}'")
    result = rag_agent.invoke({"messages": [("user", query)]})
    response = result["messages"][-1].content
    print(f"[rag_agent] Respuesta generada ({len(response)} chars)")
    return response


def run_stats_agent(query: str) -> str:
    """Ejecuta el agente Stats con una query sobre un jugador."""
    print(f"\n[stats_agent] Ejecutando con query: '{query}'")
    result = stats_agent.invoke({"messages": [("user", query)]})
    response = result["messages"][-1].content
    print(f"[stats_agent] Respuesta generada ({len(response)} chars)")
    return response


def run_comp_agent(query: str) -> str:
    """Ejecuta el agente Comp con una query de comparación entre dos jugadores."""
    print(f"\n[comp_agent] Ejecutando con query: '{query}'")
    result = comp_agent.invoke({"messages": [("user", query)]})
    response = result["messages"][-1].content
    print(f"[comp_agent] Respuesta generada ({len(response)} chars)")
    return response
