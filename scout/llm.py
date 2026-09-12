"""
Factory de LLM con routing por config.

Prioridad (un env var cambia el provider sin tocar el código del agente):

  1. LITELLM_MODEL → ChatLiteLLM — una clave cambia el provider:
       Dev local:    LITELLM_MODEL=ollama/llama3.2:3b
       Groq free:    LITELLM_MODEL=groq/llama-3.1-8b-instant  (+ GROQ_API_KEY)
       Producción:   LITELLM_MODEL=claude-3-5-sonnet-20241022 (+ ANTHROPIC_API_KEY)

  2. GROQ_API_KEY → ChatGroq (llama-3.1-8b-instant) — backward compat

  3. Sin API keys → ChatOllama (llama3.2:3b) — solo desarrollo local

Groq free tier: 30 RPM, 14.400 req/día — suficiente para un demo/portfolio.
Ollama requiere el modelo corrido en localhost:11434.
"""

import os

GROQ_MODEL = "openai/gpt-oss-20b"
OLLAMA_MODEL = "llama3.2:3b"


def make_llm(temperature: float = 0):
    """
    Retorna un ChatModel configurado según el entorno.

    Args:
        temperature: 0 = determinista (agents/tools), 0.3 = creativo (conclusiones).
    """
    litellm_model = os.getenv("LITELLM_MODEL")
    if litellm_model:
        # LiteLLM mode: un env var controla el provider completo.
        # LiteLLM entiende el prefijo del modelo:
        #   "ollama/llama3.2:3b"            → Ollama local
        #   "groq/llama-3.1-8b-instant"     → Groq API
        #   "claude-3-5-sonnet-20241022"     → Anthropic directa
        #   "openai/gpt-4o"                 → OpenAI
        from langchain_litellm import ChatLiteLLM
        return ChatLiteLLM(model=litellm_model, temperature=temperature)

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
