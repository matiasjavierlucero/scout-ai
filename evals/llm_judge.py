"""
LLM-as-judge — evaluación de calidad con modelo de lenguaje.

El juez usa un modelo INDEPENDIENTE al agente cuando GROQ_API_KEY está
configurada (llama-3.1-8b-instant via Groq). Sin la key, cae a Ollama local
con el mismo modelo que el agente — self-serving bias activo, scores no
confiables para decisiones de deploy.

Prioridad del juez:
  1. GROQ_API_KEY → ChatGroq llama-3.1-8b-instant  (independiente, más capaz)
  2. Sin key      → ChatOllama llama3.2:3b          (mismo modelo — ⚠️ bias)

Por qué Groq en lugar del mismo modelo:
  - Familia diferente de pesos: el juez no reconoce su propio estilo
  - Mayor capacidad (8B vs 3B): puede detectar errores que el agente cometería
  - Gratuito (30 RPM, 14.400 req/día): no agrega costo al pipeline de evals
"""

import json
import os
import re
from dataclasses import dataclass

from scout.llm import make_llm

_GROQ_JUDGE_MODEL = "llama-3.1-8b-instant"


@dataclass
class JudgeResult:
    relevance: float      # 0.0 – 1.0: ¿responde la query?
    grounding: float      # 0.0 – 1.0: ¿afirmaciones respaldadas por datos?
    quality: float        # 0.0 – 1.0: ¿coherente y útil?
    score: float          # promedio de las tres dimensiones
    reasoning: str        # explicación breve del juez
    model_used: str       # modelo del juez (para trazabilidad)
    is_same_model_as_agent: bool  # True = self-serving bias activo


def make_judge_llm():
    """
    Retorna el LLM independiente que actuará como juez.

    Prioriza Groq (modelo diferente al agente). Si no hay GROQ_API_KEY,
    cae a Ollama — is_same_model_as_agent quedará True en ese caso.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        from langchain_groq import ChatGroq
        return ChatGroq(model=_GROQ_JUDGE_MODEL, temperature=0, api_key=groq_key)
    return make_llm(temperature=0)


def _judge_is_independent() -> bool:
    """True si el juez usa un modelo diferente al agente."""
    return bool(os.getenv("GROQ_API_KEY"))


_JUDGE_LLM = make_judge_llm()

_JUDGE_PROMPT = """\
Evaluá el siguiente informe de scouting en 3 dimensiones.

Query del usuario: {query}

Informe generado:
---
{informe_text}
---

Respondé ÚNICAMENTE con un JSON válido, sin texto adicional, con este formato:
{{
  "relevance": <float 0.0-1.0>,
  "grounding": <float 0.0-1.0>,
  "quality": <float 0.0-1.0>,
  "reasoning": "<explicación breve en una oración>"
}}

Definiciones:
- relevance: ¿El informe responde la query del usuario?
  0.0 = completamente irrelevante, 1.0 = responde perfectamente
- grounding: ¿Las afirmaciones están basadas en datos concretos del informe?
  0.0 = hallucinations o sin datos, 1.0 = todo respaldado por stats reales
- quality: ¿El informe es coherente, estructurado y útil para un scout?
  0.0 = incoherente o vacío, 1.0 = excelente calidad

Solo JSON, nada más:"""


def _extract_json(text: str) -> dict | None:
    """
    Extrae el primer objeto JSON del texto del juez.

    llama3.2:3b a veces rodea el JSON con texto o markdown.
    Intentamos parsear directamente, y si falla, buscamos el bloque.
    """
    text = text.strip()

    # Intento directo
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Buscar bloque JSON entre llaves
    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def judge(query: str, informe) -> JudgeResult:
    """
    Evalúa la calidad del informe usando el LLM como juez.

    Args:
        query: Query original del usuario.
        informe: InformeScouting con los resultados.

    Returns:
        JudgeResult con scores y razonamiento.
    """
    informe_text = "\n".join(filter(None, [
        f"Agentes: {', '.join(informe.agentes_ejecutados)}",
        f"Jugadores sugeridos:\n{informe.jugadores_sugeridos}" if informe.jugadores_sugeridos else "",
        f"Estadísticas:\n{informe.estadisticas}" if informe.estadisticas else "",
        f"Comparativa:\n{informe.comparativa}" if informe.comparativa else "",
        f"Conclusión:\n{informe.conclusion}" if informe.conclusion else "",
    ]))

    prompt = _JUDGE_PROMPT.format(query=query, informe_text=informe_text)

    try:
        response = _JUDGE_LLM.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        parsed = _extract_json(content)
    except Exception as e:
        return JudgeResult(
            relevance=0.0,
            grounding=0.0,
            quality=0.0,
            score=0.0,
            reasoning=f"Error al invocar el juez: {e}",
            model_used=type(_JUDGE_LLM).__name__,
            is_same_model_as_agent=True,
        )

    if parsed is None:
        return JudgeResult(
            relevance=0.0,
            grounding=0.0,
            quality=0.0,
            score=0.0,
            reasoning=f"El juez no devolvió JSON válido. Raw: {content[:200]}",
            model_used=type(_JUDGE_LLM).__name__,
            is_same_model_as_agent=True,
        )

    relevance = float(parsed.get("relevance", 0.0))
    grounding = float(parsed.get("grounding", 0.0))
    quality   = float(parsed.get("quality", 0.0))
    score = (relevance + grounding + quality) / 3

    return JudgeResult(
        relevance=relevance,
        grounding=grounding,
        quality=quality,
        score=score,
        reasoning=str(parsed.get("reasoning", "")),
        model_used=type(_JUDGE_LLM).__name__,
        is_same_model_as_agent=not _judge_is_independent(),
    )
