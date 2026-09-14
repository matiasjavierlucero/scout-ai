#!/usr/bin/env python3
"""
Script de evaluación de Scout AI.

Corre los 10 casos del golden set, aplica sanity checks + LLM-as-judge (opcional),
reporta un resumen de scores y los envía a Langfuse si las API keys están configuradas.

Si el score promedio de sanity está por debajo del threshold, el script termina
con código de salida 1 — útil para bloquear un deploy en CI.

⚠️ LLM-as-judge usa el mismo modelo que el agente (llama3.2:3b) — ver llm_judge.py
   para la explicación completa del red flag. Los scores del juez NO son confiables
   para decisiones de producción.

Uso:
  uv run python evals/run_evals.py                # solo sanity checks
  uv run python evals/run_evals.py --judge        # sanity + LLM-as-judge
  uv run python evals/run_evals.py --threshold 0.8
  uv run python evals/run_evals.py --case rag-001  # un caso específico
"""

import argparse
import sys
import time
from pathlib import Path

# Agregar root al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


def _print_header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _print_section(title: str) -> None:
    print(f"\n── {title} {'─' * (55 - len(title))}")


def _score_bar(score: float, width: int = 20) -> str:
    filled = int(score * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {score:.2f}"


def run_eval_case(case, use_judge: bool) -> dict:
    """
    Corre un caso del golden set y retorna el resultado completo.

    Returns:
        Dict con el caso, el informe, los resultados de sanity y (opcional) el juicio.
    """
    from scout.graph import run
    from evals.sanity import run_all_sanity_checks
    from evals.golden_set import GoldenCase

    print(f"\n  [{case['id']}] {case['query'][:70]}...")

    t0 = time.time()
    try:
        informe = run(case["query"])
        elapsed = time.time() - t0
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ❌ Error al ejecutar: {e}")
        return {
            "case": case,
            "informe": None,
            "sanity": [],
            "judge": None,
            "elapsed": elapsed,
            "error": str(e),
        }

    sanity_results = run_all_sanity_checks(case, informe, elapsed)

    judge_result = None
    if use_judge:
        from evals.llm_judge import judge
        judge_result = judge(case["query"], informe)

    return {
        "case": case,
        "informe": informe,
        "sanity": sanity_results,
        "judge": judge_result,
        "elapsed": elapsed,
        "error": None,
    }


def print_case_results(result: dict) -> None:
    """Imprime los resultados de un caso con formato."""
    case = result["case"]
    sanity = result["sanity"]
    judge = result["judge"]

    _print_section(f"{case['id']} ({case['tipo']})")
    print(f"  Query: {case['query'][:70]}")
    print(f"  Tiempo: {result['elapsed']:.1f}s")

    if result["error"]:
        print(f"  ❌ ERROR: {result['error']}")
        return

    print(f"\n  Sanity checks:")
    for r in sanity:
        icon = "✓" if r.passed else "✗"
        print(f"    {icon} {r.name:<25} {_score_bar(r.score, 12)}  {r.reason[:60]}")

    if judge is not None:
        bias_tag = "⚠️  mismo modelo → bias activo" if judge.is_same_model_as_agent else "✓ juez independiente"
        print(f"\n  LLM Judge ({bias_tag}):")
        print(f"    relevance:  {_score_bar(judge.relevance, 12)}")
        print(f"    grounding:  {_score_bar(judge.grounding, 12)}")
        print(f"    quality:    {_score_bar(judge.quality, 12)}")
        print(f"    score avg:  {_score_bar(judge.score, 12)}")
        print(f"    reasoning:  {judge.reasoning[:80]}")


def print_summary(results: list[dict], threshold: float) -> float:
    """Imprime el resumen global y retorna el score promedio de sanity."""
    _print_header("RESUMEN DE EVALUACIÓN")

    total_cases = len(results)
    error_cases = sum(1 for r in results if r["error"])
    ok_cases = total_cases - error_cases

    # Score promedio de sanity (solo casos sin error)
    sanity_scores = []
    for r in results:
        if not r["error"] and r["sanity"]:
            case_score = sum(s.score for s in r["sanity"]) / len(r["sanity"])
            sanity_scores.append(case_score)

    avg_sanity = sum(sanity_scores) / len(sanity_scores) if sanity_scores else 0.0

    # Score promedio de judge (si aplica)
    judge_scores = [r["judge"].score for r in results if r["judge"] is not None]
    avg_judge = sum(judge_scores) / len(judge_scores) if judge_scores else None

    print(f"\n  Casos ejecutados: {ok_cases}/{total_cases}")
    if error_cases > 0:
        print(f"  Casos con error:  {error_cases}")

    print(f"\n  Score sanity (rule-based):   {_score_bar(avg_sanity)}")
    if avg_judge is not None:
        print(f"  Score judge  (⚠️  LLM bias): {_score_bar(avg_judge)}")

    print(f"\n  Threshold configurado: {threshold:.2f}")

    passed = avg_sanity >= threshold
    if passed:
        print(f"  ✅ PASSED — score {avg_sanity:.3f} ≥ threshold {threshold}")
    else:
        print(f"  ❌ FAILED — score {avg_sanity:.3f} < threshold {threshold}")

    # Casos que fallaron
    failed_cases = [r for r in results if not r["error"] and r["sanity"]]
    failed_cases = [
        r for r in failed_cases
        if sum(s.score for s in r["sanity"]) / len(r["sanity"]) < threshold
    ]
    if failed_cases:
        print(f"\n  Casos por debajo del threshold:")
        for r in failed_cases:
            case_score = sum(s.score for s in r["sanity"]) / len(r["sanity"])
            print(f"    - {r['case']['id']}: {case_score:.3f}")

    return avg_sanity


def send_to_langfuse(results: list[dict], avg_sanity: float, run_id: str) -> None:
    """Envía los scores de evaluación a Langfuse como un trace de eval."""
    from scout.tracing import get_langfuse
    langfuse = get_langfuse()
    if langfuse is None:
        print("\n  (Langfuse no configurado — scores no enviados)")
        return

    print(f"\n  Enviando scores a Langfuse (run_id: {run_id})...")

    # En Langfuse v4, create_score requiere trace_id.
    # Creamos un trace que agrupa todos los scores del run.
    trace = langfuse.trace(
        name="eval-run",
        input={"run_id": run_id, "n_cases": len(results)},
        output={"avg_sanity": avg_sanity},
        tags=["eval"],
        metadata={"run_id": run_id},
    )

    for r in results:
        if r["error"] or not r["sanity"]:
            continue

        case_score = sum(s.score for s in r["sanity"]) / len(r["sanity"])
        case_id = r["case"]["id"]

        langfuse.create_score(
            trace_id=trace.id,
            name="sanity_score",
            value=case_score,
            comment=f"case={case_id}",
        )

        if r["judge"] is not None:
            bias = "bias" if r["judge"].is_same_model_as_agent else "independent"
            langfuse.create_score(
                trace_id=trace.id,
                name="judge_score",
                value=r["judge"].score,
                comment=f"case={case_id} judge={bias}",
            )

    langfuse.create_score(
        trace_id=trace.id,
        name="eval_avg_sanity",
        value=avg_sanity,
        comment=f"n={len(results)} casos",
    )

    langfuse.flush()
    print("  ✓ Scores enviados a Langfuse")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa Scout AI contra el golden set")
    parser.add_argument("--judge", action="store_true", help="Activar LLM-as-judge (lento)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Score mínimo (default: 0.7)")
    parser.add_argument("--case", type=str, default=None, help="Correr solo un caso por id")
    args = parser.parse_args()

    from evals.golden_set import GOLDEN_SET

    cases_to_run = GOLDEN_SET
    if args.case:
        cases_to_run = [c for c in GOLDEN_SET if c["id"] == args.case]
        if not cases_to_run:
            print(f"Error: caso '{args.case}' no encontrado")
            sys.exit(1)

    _print_header(f"SCOUT AI — EVALUACIÓN ({len(cases_to_run)} casos)")
    if args.judge:
        from evals.llm_judge import _judge_is_independent
        if _judge_is_independent():
            print("\n  ✓  LLM-as-judge ACTIVO — juez independiente (GROQ_API_KEY configurada)")
        else:
            print("\n  ⚠️  LLM-as-judge ACTIVO")
            print("  ⚠️  Usando el mismo modelo como juez y agente (self-serving bias)")
            print("  ⚠️  Los scores del juez NO son válidos para decisiones de deploy")

    results = []
    for case in cases_to_run:
        result = run_eval_case(case, use_judge=args.judge)
        results.append(result)
        print_case_results(result)

    avg_sanity = print_summary(results, threshold=args.threshold)

    # Enviar a Langfuse con un run_id único
    import uuid
    run_id = str(uuid.uuid4())[:8]
    send_to_langfuse(results, avg_sanity, run_id)

    # Exit code para CI: 0 si pasa el threshold, 1 si no
    sys.exit(0 if avg_sanity >= args.threshold else 1)


if __name__ == "__main__":
    main()
