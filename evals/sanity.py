"""
Sanity checks — evaluaciones rule-based, determinísticas, sin LLM.

Estas checks son baratas (microsegundos), 100% reproducibles y nunca fallan
por latencia de red ni comportamiento no-determinista del LLM.
Son la primera línea de defensa en un pipeline de evals.

Cada check retorna un EvalResult con:
  - passed: bool
  - score: float (1.0 = pasó, 0.0 = falló, parcial si aplica)
  - reason: string explicando el resultado
"""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scout.schemas import InformeScouting
    from evals.golden_set import GoldenCase


@dataclass
class EvalResult:
    name: str
    passed: bool
    score: float          # 0.0 – 1.0
    reason: str


def check_has_content(informe: "InformeScouting") -> EvalResult:
    """El informe tiene al menos un campo de contenido no vacío."""
    has_content = any([
        informe.jugadores_sugeridos.strip(),
        informe.estadisticas.strip(),
        informe.comparativa.strip(),
    ])
    return EvalResult(
        name="has_content",
        passed=has_content,
        score=1.0 if has_content else 0.0,
        reason="El informe tiene contenido" if has_content else "Todos los campos de contenido están vacíos",
    )


def check_agentes_correctos(case: "GoldenCase", informe: "InformeScouting") -> EvalResult:
    """Los agentes ejecutados coinciden con los esperados para este tipo de query."""
    expected = set(case["expected_agentes"])
    actual = set(informe.agentes_ejecutados)
    match = expected.issubset(actual)  # permite agentes extra, exige los esperados
    return EvalResult(
        name="agentes_correctos",
        passed=match,
        score=1.0 if match else 0.0,
        reason=(
            f"Agentes correctos: {sorted(actual)}"
            if match
            else f"Esperaba {sorted(expected)}, ejecutó {sorted(actual)}"
        ),
    )


def check_keywords_en_resultado(case: "GoldenCase", informe: "InformeScouting") -> EvalResult:
    """Las keywords esperadas aparecen en algún campo del informe."""
    full_text = " ".join([
        informe.jugadores_sugeridos,
        informe.estadisticas,
        informe.comparativa,
    ]).lower()

    found = [kw for kw in case["expected_keywords"] if kw.lower() in full_text]
    missing = [kw for kw in case["expected_keywords"] if kw.lower() not in full_text]
    total = len(case["expected_keywords"])
    score = len(found) / total if total > 0 else 1.0

    return EvalResult(
        name="keywords_presentes",
        passed=score >= 0.7,  # pasa si ≥70% de keywords están
        score=score,
        reason=(
            f"Keywords presentes: {found}" if not missing
            else f"Presentes: {found} | Ausentes: {missing}"
        ),
    )


def check_jugadores_mencionados(case: "GoldenCase", informe: "InformeScouting") -> EvalResult:
    """Los jugadores esperados están mencionados en el resultado."""
    expected_players = case["expected_players"]
    if not expected_players:
        return EvalResult(
            name="jugadores_mencionados",
            passed=True,
            score=1.0,
            reason="No se esperaban jugadores específicos (query RAG)",
        )

    full_text = " ".join([
        informe.jugadores_sugeridos,
        informe.estadisticas,
        informe.comparativa,
        " ".join(informe.jugadores_detectados),
    ]).lower()

    found = [p for p in expected_players if p.lower() in full_text]
    score = len(found) / len(expected_players)

    return EvalResult(
        name="jugadores_mencionados",
        passed=score >= 1.0,
        score=score,
        reason=(
            f"Todos los jugadores mencionados: {found}"
            if score == 1.0
            else f"Encontrados: {found} | Faltantes: {[p for p in expected_players if p.lower() not in full_text]}"
        ),
    )


def check_stats_en_rango(informe: "InformeScouting") -> EvalResult:
    """
    Las estadísticas en el texto están en rangos físicamente posibles.

    Los rangos son conservadores para eliminar valores claramente corruptos
    (e.g., xG/90 = 50, pases/90 = 1000). No son restricciones de calidad.
    """
    import re

    if not informe.estadisticas:
        return EvalResult(
            name="stats_en_rango",
            passed=True,
            score=1.0,
            reason="Sin campo de estadísticas — no aplica",
        )

    issues = []

    # xG/90 razonable: 0 – 2.0 (el mejor de la historia rara vez supera 1.5)
    for match in re.finditer(r"xG/90[:\s]+([0-9.]+)", informe.estadisticas):
        val = float(match.group(1))
        if val > 2.0:
            issues.append(f"xG/90={val} fuera de rango (>2.0)")

    # Pases completados/90: 0 – 150 (Busquets ~90)
    for match in re.finditer(r"[Pp]ases completados/90[:\s]+([0-9.]+)", informe.estadisticas):
        val = float(match.group(1))
        if val > 150:
            issues.append(f"Pases/90={val} fuera de rango (>150)")

    # Presiones/90: 0 – 50 (jugadores muy presionadores ~25-30)
    for match in re.finditer(r"[Pp]resiones/90[:\s]+([0-9.]+)", informe.estadisticas):
        val = float(match.group(1))
        if val > 50:
            issues.append(f"Presiones/90={val} fuera de rango (>50)")

    passed = len(issues) == 0
    return EvalResult(
        name="stats_en_rango",
        passed=passed,
        score=1.0 if passed else 0.0,
        reason="Todos los valores en rango válido" if passed else f"Valores fuera de rango: {issues}",
    )


def check_schema_completo(informe: "InformeScouting") -> EvalResult:
    """El InformeScouting tiene todos los campos requeridos populados."""
    issues = []
    if not informe.query:
        issues.append("query vacío")
    if not informe.agentes_ejecutados:
        issues.append("agentes_ejecutados vacío")

    passed = len(issues) == 0
    return EvalResult(
        name="schema_completo",
        passed=passed,
        score=1.0 if passed else 0.0,
        reason="Schema completo" if passed else f"Campos faltantes: {issues}",
    )


def check_response_time(elapsed_secs: float, max_secs: float) -> EvalResult:
    """El tiempo de respuesta está dentro del límite aceptable."""
    passed = elapsed_secs <= max_secs
    return EvalResult(
        name="response_time",
        passed=passed,
        score=1.0 if passed else max(0.0, 1.0 - (elapsed_secs - max_secs) / max_secs),
        reason=f"{elapsed_secs:.1f}s ({'OK' if passed else f'excede límite de {max_secs}s'})",
    )


def run_all_sanity_checks(
    case: "GoldenCase",
    informe: "InformeScouting",
    elapsed_secs: float,
) -> list[EvalResult]:
    """
    Corre todos los sanity checks para un caso del golden set.

    Args:
        case: GoldenCase con las expectativas.
        informe: Respuesta del sistema para evaluar.
        elapsed_secs: Tiempo que tardó el sistema en responder.

    Returns:
        Lista de EvalResult, uno por check.
    """
    return [
        check_has_content(informe),
        check_agentes_correctos(case, informe),
        check_keywords_en_resultado(case, informe),
        check_jugadores_mencionados(case, informe),
        check_stats_en_rango(informe),
        check_schema_completo(informe),
        check_response_time(elapsed_secs, case["max_response_secs"]),
    ]
