"""
Factory de LLM con auto-detección del entorno.

Prioridad:
  1. GROQ_API_KEY → ChatGroq (llama-3.1-8b-instant) — gratis, rápido, tool calling
  2. Sin API key  → ChatOllama (llama3.2:3b) — solo desarrollo local

Groq free tier: 30 RPM, 14.400 req/día — suficiente para un demo/portfolio.
Registrate en https://console.groq.com para obtener una API key gratis.

Uso:
  from scout.llm import make_llm
  llm = make_llm(temperature=0)
"""

import os


GROQ_MODEL = "llama-3.1-8b-instant"
OLLAMA_MODEL = "llama3.2:3b"


def make_llm(temperature: float = 0):
    """
    Retorna un ChatModel configurado según el entorno.

    Args:
        temperature: 0 = determinista (agents/tools), 0.3 = creativo (conclusiones).
    """
    groq_key = os.getenv("GROQ_API_KEY")

    if groq_key:
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=GROQ_MODEL,
            temperature=temperature,
            api_key=groq_key,
        )

    # Fallback local — requiere Ollama en localhost:11434
    from langchain_ollama import ChatOllama

    return ChatOllama(model=OLLAMA_MODEL, temperature=temperature)
