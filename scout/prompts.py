"""
Gestión de prompts con Langfuse Prompt Management.

Intenta obtener cada prompt desde Langfuse (etiqueta "production").
Si Langfuse no está configurado o el prompt no existe, usa el fallback
hardcodeado — el sistema nunca falla por un prompt faltante.

Para registrar los prompts por primera vez en Langfuse:
  uv run python scripts/setup_prompts.py

Una vez en Langfuse, los prompts se pueden editar desde el dashboard
sin tocar el código ni hacer deploy. Cada edición crea una nueva versión.
"""

from scout.tracing import get_langfuse

# Fallbacks hardcodeados — fuente de verdad si Langfuse no está disponible.
# Estos son los valores que setup_prompts.py sube a Langfuse en la v1.
PROMPTS: dict[str, str] = {
    "scout-rag-agent": (
        "Sos un agente de scouting especializado en búsqueda semántica de jugadores. "
        "Recibís descripciones de perfiles de juego en lenguaje natural y usás la tool "
        "buscar_jugadores para encontrar los jugadores más similares en el dataset de La Liga. "
        "Siempre mostrá los resultados de la tool directamente, sin inventar información adicional. "
        "Si el resultado incluye advertencias de muestra pequeña, mencionálas."
    ),
    "scout-stats-agent": (
        "Sos un agente de scouting especializado en estadísticas de jugadores. "
        "Dado un nombre, usá directamente stats_jugador — la tool hace fuzzy matching "
        "internamente si el nombre no es exacto. "
        "Mencioná la advertencia de muestra pequeña si el jugador tiene menos de 450 minutos."
    ),
    "scout-comp-agent": (
        "Sos un agente de scouting especializado en comparación de jugadores. "
        "Dados dos nombres, usá comparar_jugadores directamente — la tool hace fuzzy matching "
        "internamente para cada jugador. "
        "Presentá la tabla comparativa tal como viene de la tool. "
        "Mencioná advertencias de muestra pequeña si aplica."
    ),
}


def get_prompt(name: str) -> str:
    """
    Obtiene un prompt desde Langfuse (label "production"), con fallback hardcodeado.

    Args:
        name: Nombre del prompt en Langfuse (ej: "scout-rag-agent").

    Returns:
        Texto del prompt listo para usar como system prompt.
    """
    langfuse = get_langfuse()
    if langfuse is not None:
        try:
            prompt = langfuse.get_prompt(name, label="production")
            text = prompt.compile()
            print(f"[prompts] '{name}' cargado desde Langfuse (v{prompt.version})")
            return text
        except Exception as e:
            print(f"[prompts] '{name}' no encontrado en Langfuse — usando fallback ({e})")

    return PROMPTS[name]
