"""
LLM-as-judge — evaluación de calidad con modelo de lenguaje.

⚠️  RED FLAG DE PRODUCCIÓN — LEER ANTES DE USAR ⚠️

Este módulo usa llama3.2:3b como juez de calidad de las respuestas generadas
por el mismo llama3.2:3b. Esto es una MALA PRÁCTICA en producción por dos
razones fundamentales:

  1. Self-serving bias: un modelo tiende a calificarse bien a sí mismo
     independientemente de la calidad real de sus respuestas. No puede
     evaluar objetivamente su propio output.

  2. Capacidad insuficiente: un modelo de 3B parámetros no tiene el
     razonamiento suficiente para ser árbitro confiable de sí mismo.

En producción, el juez DEBE ser:
  - Un modelo independiente y más capaz: Claude Sonnet, GPT-4o
  - O al menos una familia diferente: si el agente es llama3.2, el juez
    debería ser llama3.1:8b o superior (más capacidad + pesos diferentes)

Usamos llama3.2:3b aquí ÚNICAMENTE porque:
  a) Es el único modelo instalado localmente
  b) El objetivo es aprender el PATRÓN de LLM-as-judge, no obtener scores
     confiables

Los scores de este módulo NO son válidos para decisiones de deployment.
Son útiles para detectar regresiones muy obvias (respuesta completamente
irrelevante o vacía), pero no para medir calidad real.

Para producción real, reemplazá make_judge_llm() para que devuelva
un modelo independiente. El resto del código no cambia.
"""

import json
import re
from dataclasses import dataclass

from scout.llm import make_llm


@dataclass
class JudgeResult:
    relevance: float      # 0.0 – 1.0: ¿responde la query?
    grounding: float      # 0.0 – 1.0: ¿afirmaciones respaldadas por datos?
    quality: float        # 0.0 – 1.0: ¿coherente y útil?
    score: float          # promedio de las tres dimensiones
    reasoning: str        # explicación breve del juez
    model_used: str       # modelo del juez (documentar para trazabilidad)
    is_same_model_as_agent: bool  # True = red flag activo


def make_judge_llm():
    """
    Retorna el LLM que actuará como juez.

    ⚠️ PRODUCCIÓN: reemplazá este método para devolver un modelo
    independiente (Claude Sonnet, GPT-4o, etc.) antes de usar los
    scores para decisiones reales.
    """
    # En prod: return ChatAnthropic(model="claude-sonnet-4-6") o similar
    return make_llm(temperature=0)


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
        is_same_model_as_agent=True,  # siempre True hasta que se reemplace make_judge_llm()
    )
