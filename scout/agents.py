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
from langchain_core.messages import ToolMessage

from scout.llm import make_llm
from scout.prompts import get_prompt
from scout.tools import (
    buscar_jugadores,
    comparar_jugadores,
    stats_jugador,
)

print("[agents] Inicializando modelo...")
_llm = make_llm(temperature=0)
print(f"[agents] Modelo listo: {type(_llm).__name__}\n")


rag_agent = create_agent(
    model=_llm,
    tools=[buscar_jugadores],
    system_prompt=get_prompt("scout-rag-agent"),
)

stats_agent = create_agent(
    model=_llm,
    tools=[stats_jugador],
    system_prompt=get_prompt("scout-stats-agent"),
)

comp_agent = create_agent(
    model=_llm,
    tools=[comparar_jugadores],
    system_prompt=get_prompt("scout-comp-agent"),
)


def _extract_tool_output(messages: list) -> str:
    """
    Extrae el contenido del último ToolMessage de la conversación del agente.

    llama3.2:3b tiende a parafrasear el output de las tools en vez de copiarlo.
    Tomamos el ToolMessage directamente para garantizar que el dato real llegue
    al informe final, sin pasar por el LLM.

    Si no hay ToolMessage (el agente respondió sin llamar tools), devuelve
    el último mensaje del LLM como fallback.
    """
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            return msg.content
    return messages[-1].content


def run_rag_agent(query: str) -> str:
    """Ejecuta el agente RAG con una query de perfil de juego."""
    print(f"\n[rag_agent] Ejecutando con query: '{query}'")
    result = rag_agent.invoke({"messages": [("user", query)]})
    response = _extract_tool_output(result["messages"])
    print(f"[rag_agent] Respuesta generada ({len(response)} chars)")
    return response


def run_stats_agent(query: str) -> str:
    """Ejecuta el agente Stats con una query sobre un jugador."""
    print(f"\n[stats_agent] Ejecutando con query: '{query}'")
    result = stats_agent.invoke({"messages": [("user", query)]})
    response = _extract_tool_output(result["messages"])
    print(f"[stats_agent] Respuesta generada ({len(response)} chars)")
    return response


def run_comp_agent(query: str) -> str:
    """Ejecuta el agente Comp con una query de comparación entre dos jugadores."""
    print(f"\n[comp_agent] Ejecutando con query: '{query}'")
    result = comp_agent.invoke({"messages": [("user", query)]})
    response = _extract_tool_output(result["messages"])
    print(f"[comp_agent] Respuesta generada ({len(response)} chars)")
    return response
