"""
Langfuse tracing — instrumentación del grafo LangGraph.

Inicializa un cliente Langfuse singleton y expone helpers para:
  - Inyectar el CallbackHandler en graph.invoke() y stream_conclusion()
  - Enviar eval scores asociados a un trace

Requiere en .env:
  LANGFUSE_PUBLIC_KEY=pk-lf-...
  LANGFUSE_SECRET_KEY=sk-lf-...
  LANGFUSE_BASE_URL=https://us.cloud.langfuse.com   # US region (o cloud.langfuse.com para EU)

Acepta cualquiera de estos nombres de env var para el host (en orden de prioridad):
  LANGFUSE_HOST, LANGFUSE_BASE_URL, LANGFUSE_BASEURL

Si las keys no están configuradas, get_langfuse() retorna None y el sistema
funciona normalmente sin tracing — fail-safe, nunca rompe el flujo principal.
"""

import os
from typing import Optional


_langfuse = None


def get_langfuse():
    """
    Retorna el cliente Langfuse singleton.
    None si las keys no están configuradas.
    """
    global _langfuse
    if _langfuse is None:
        pk = os.getenv("LANGFUSE_PUBLIC_KEY")
        sk = os.getenv("LANGFUSE_SECRET_KEY")
        if pk and sk:
            from langfuse import Langfuse
            host = (
                os.getenv("LANGFUSE_HOST")
                or os.getenv("LANGFUSE_BASE_URL")
                or os.getenv("LANGFUSE_BASEURL")
            )
            kwargs = {"public_key": pk, "secret_key": sk}
            if host:
                kwargs["host"] = host
            _langfuse = Langfuse(**kwargs)
    return _langfuse


def get_callback_handler():
    """
    Retorna un nuevo CallbackHandler de Langfuse por llamada.

    Cada instancia crea un trace independiente en Langfuse —
    úsalo una vez por invocación del grafo/stream.
    Retorna None si Langfuse no está configurado.
    """
    if get_langfuse() is None:
        return None
    from langfuse.langchain import CallbackHandler
    return CallbackHandler()


def send_eval_scores(trace_id: str, scores: dict[str, float], comment: str = "") -> None:
    """
    Envía scores de evaluación a Langfuse asociados a un trace.

    Args:
        trace_id: ID del trace (obtenido con langfuse.get_current_trace_id()).
        scores: Dict nombre → valor (e.g. {"relevance": 0.8, "sanity": 1.0}).
        comment: Texto libre para contexto adicional.
    """
    langfuse = get_langfuse()
    if langfuse is None:
        return
    for name, value in scores.items():
        langfuse.create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment or None,
        )
