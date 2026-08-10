"""
Factory de LLM con auto-detección del entorno.

En producción (Streamlit Cloud u otro hosting):
  - Si ANTHROPIC_API_KEY está presente → usa Claude claude-haiku-4-5-20251001

En desarrollo local:
  - Si no hay ANTHROPIC_API_KEY → usa Ollama (llama3.2:3b en localhost:11434)

Uso:
  from scout.llm import make_llm
  llm = make_llm(temperature=0)
"""

import os


def make_llm(temperature: float = 0):
    """
    Retorna un ChatModel configurado según el entorno.

    Args:
        temperature: Controla la aleatoriedad de las respuestas.
                     0 = determinista (para tools/razonamiento).
                     0.3 = levemente creativo (para conclusiones).
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if api_key:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            temperature=temperature,
            api_key=api_key,
        )

    # Fallback local — requiere Ollama corriendo en localhost:11434
    from langchain_ollama import ChatOllama

    return ChatOllama(model="llama3.2:3b", temperature=temperature)
