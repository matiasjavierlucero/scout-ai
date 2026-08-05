"""
Paso 3: Tools del agente de scouting.

Cuatro tools con @tool decorator de LangChain:
  - buscar_por_nombre   — fuzzy match por nombre, devuelve candidatos para que el usuario elija
  - stats_jugador       — stats completas de un jugador por nombre exacto
  - comparar_jugadores  — tabla comparativa lado a lado
  - buscar_jugadores    — búsqueda semántica por perfil de juego
"""

import json
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MINUTES_WARNING_THRESHOLD = 450  # menos de 5 partidos completos

print("[tools] Cargando datos en memoria...")

_stats_df = pd.read_csv(DATA_DIR / "player_stats.csv")
print(f"[tools]   player_stats.csv cargado — {len(_stats_df)} jugadores")

with open(DATA_DIR / "player_profiles.json", encoding="utf-8") as f:
    _profiles = json.load(f)
print(f"[tools]   player_profiles.json cargado — {len(_profiles)} perfiles")

with open(DATA_DIR / "index.json", encoding="utf-8") as f:
    _index = json.load(f)
_index_names = list(_index.keys())
print(f"[tools]   index.json cargado — {len(_index_names)} entradas")

_embeddings = np.load(DATA_DIR / "embeddings.npy")
print(f"[tools]   embeddings.npy cargado — shape {_embeddings.shape}")

print(f"[tools] Cargando modelo de embeddings {MODEL_NAME}...")
_model = SentenceTransformer(MODEL_NAME)
print(f"[tools] Modelo listo. Todo cargado.\n")


def _fuzzy_score(a: str, b: str) -> float:
    """
    Calcula qué tan similares son dos strings como un número entre 0.0 y 1.0.
    Usa SequenceMatcher de difflib, que mide la proporción de caracteres en común
    sobre el total de caracteres de ambos strings (algoritmo Ratcliff/Obershelp).

    Ejemplos:
      "Messi"  vs "Lionel Messi"  → ~0.59
      "Ronaldo" vs "Cristiano Ronaldo" → ~0.54
      "Xavi"   vs "Xavi Hernández" → ~0.62
      "Xavi"   vs "Xavi"           → 1.0  (igual)
      "Messi"  vs "Ronaldo"        → ~0.17 (muy distinto)
    """
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _find_candidates(nombre: str, k: int = 5) -> list[dict]:
    """
    Devuelve hasta k jugadores cuyo nombre se parece a `nombre`.

    Estrategia de scoring en dos niveles:
    1. Token match: si alguna palabra del query aparece exactamente en el nombre
       del candidato, score alto (0.5 + proporción de tokens que matchean).
       Esto resuelve "Messi" → "Lionel Andrés Messi Cuccittini".
    2. Fallback a _fuzzy_score (caracteres) si ningún token matchea.
       Esto cubre typos y nombres parciales sin palabras exactas.
    """
    query_tokens = set(nombre.lower().split())

    def score(name: str) -> float:
        name_tokens = set(name.lower().split())
        overlap = len(query_tokens & name_tokens)
        if overlap > 0:
            return 0.5 + 0.5 * (overlap / max(len(query_tokens), 1))
        return _fuzzy_score(nombre, name)

    scored = [(name, score(name)) for name in _index_names]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:k]

    candidates = []
    for name, score in top:
        row = _stats_df[_stats_df["name"] == name]
        if row.empty:
            continue
        row = row.iloc[0]
        candidates.append({
            "name": name,
            "position": row["position"],
            "minutes_played": int(row["minutes_played"]),
            "similarity_score": round(score, 3),
        })

    return candidates


def _cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Calcula la similitud coseno entre un vector query y cada fila de una matriz.
    Devuelve un array de scores entre -1.0 y 1.0 (en la práctica 0.0 a 1.0 con embeddings).

    La similitud coseno mide el ángulo entre dos vectores, no su distancia euclidiana.
    Dos vectores apuntan en la misma dirección → score 1.0 (muy similares).
    Dos vectores perpendiculares → score 0.0 (sin relación semántica).

    El +1e-10 evita división por cero si algún vector es todo ceros.

    Ejemplo:
      query_vec shape: (384,)         ← embedding de la query del usuario
      matrix shape:    (1811, 384)    ← embeddings de los 1811 jugadores
      resultado shape: (1811,)        ← un score por jugador
    """
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return matrix_norm @ query_norm


@tool
def buscar_por_nombre(nombre: str) -> str:
    """
    Busca jugadores en el dataset por nombre o apellido usando fuzzy matching.
    Útil cuando el usuario sabe quién busca pero no conoce el nombre exacto.
    Devuelve hasta 5 candidatos con nombre, posición y minutos jugados para
    que el usuario elija el jugador correcto antes de pedir sus stats.

    Args:
        nombre: Nombre o apellido del jugador (puede ser parcial o con typos).

    Returns:
        Lista de hasta 5 candidatos con nombre, posición y minutos jugados.
    """
    print(f"\n[buscar_por_nombre] Query: '{nombre}'")
    candidates = _find_candidates(nombre, k=5)

    if not candidates:
        print(f"[buscar_por_nombre] Sin resultados para '{nombre}'")
        return f"No encontré jugadores similares a '{nombre}'."

    print(f"[buscar_por_nombre] {len(candidates)} candidatos encontrados:")
    lines = [f"Encontré {len(candidates)} jugadores similares a '{nombre}':\n"]
    for i, c in enumerate(candidates, 1):
        warning = " ⚠️ muestra pequeña" if c["minutes_played"] < MINUTES_WARNING_THRESHOLD else ""
        line = (
            f"  {i}. {c['name']}"
            f" — {c['position']}"
            f" — {c['minutes_played']} min{warning}"
        )
        print(f"[buscar_por_nombre]  {line.strip()}")
        lines.append(line)

    return "\n".join(lines)


@tool
def stats_jugador(nombre: str) -> str:
    """
    Devuelve las estadísticas completas por 90 minutos de un jugador dado su nombre exacto.
    Usar después de buscar_por_nombre para obtener el nombre exacto del jugador.
    Incluye advertencia si el jugador tiene pocos minutos (muestra poco representativa).

    Args:
        nombre: Nombre exacto del jugador tal como aparece en el dataset StatsBomb.

    Returns:
        Stats completas del jugador por 90 minutos.
    """
    print(f"\n[stats_jugador] Buscando stats de '{nombre}'")
    row = _stats_df[_stats_df["name"] == nombre]

    if row.empty:
        print(f"[stats_jugador] Jugador '{nombre}' no encontrado — intentando fuzzy match")
        candidates = _find_candidates(nombre, k=1)
        if not candidates:
            return f"No encontré al jugador '{nombre}'. Usá buscar_por_nombre para encontrar el nombre exacto."
        best = candidates[0]
        print(f"[stats_jugador] Mejor match: '{best['name']}' (score={best['similarity_score']})")
        row = _stats_df[_stats_df["name"] == best["name"]]

    row = row.iloc[0]
    minutes = int(row["minutes_played"])
    print(f"[stats_jugador] Jugador encontrado: {row['name']} — {minutes} minutos")

    warning = ""
    if minutes < MINUTES_WARNING_THRESHOLD:
        warning = (
            f"\n⚠️  ADVERTENCIA: Solo {minutes} minutos jugados "
            f"(menos de 5 partidos). Las stats por 90 pueden no ser representativas."
        )
        print(f"[stats_jugador] ⚠️  Muestra pequeña: {minutes} min")

    result = (
        f"📊 Stats de {row['name']} ({row['position']}){warning}\n"
        f"{'─' * 45}\n"
        f"  Minutos jugados:        {minutes}\n"
        f"  Pases completados/90:   {row['passes_completed_p90']}\n"
        f"  Pases totales/90:       {row['passes_total_p90']}\n"
        f"  Tiros/90:               {row['shots_p90']}\n"
        f"  xG/90:                  {row['xg_p90']:.2f}\n"
        f"  Recuperaciones/90:      {row['ball_recoveries_p90']}\n"
        f"  Presiones/90:           {row['pressures_p90']}\n"
        f"  Regates completados/90: {row['dribbles_won_p90']}\n"
    )

    print(f"[stats_jugador] Stats generadas correctamente")
    return result


@tool
def comparar_jugadores(nombre_a: str, nombre_b: str) -> str:
    """
    Compara dos jugadores lado a lado en una tabla de métricas clave por 90 minutos.
    Usar después de confirmar los nombres exactos con buscar_por_nombre.
    Incluye advertencia si alguno de los dos tiene pocos minutos.

    Args:
        nombre_a: Nombre exacto del primer jugador.
        nombre_b: Nombre exacto del segundo jugador.

    Returns:
        Tabla comparativa con las métricas clave de ambos jugadores.
    """
    print(f"\n[comparar_jugadores] Comparando '{nombre_a}' vs '{nombre_b}'")

    def get_row(nombre: str):
        row = _stats_df[_stats_df["name"] == nombre]
        if row.empty:
            candidates = _find_candidates(nombre, k=1)
            if not candidates:
                return None
            row = _stats_df[_stats_df["name"] == candidates[0]["name"]]
        return row.iloc[0]

    row_a = get_row(nombre_a)
    row_b = get_row(nombre_b)

    if row_a is None:
        return f"No encontré al jugador '{nombre_a}'. Usá buscar_por_nombre."
    if row_b is None:
        return f"No encontré al jugador '{nombre_b}'. Usá buscar_por_nombre."

    print(f"[comparar_jugadores] Jugadores encontrados: {row_a['name']} vs {row_b['name']}")

    warnings = []
    for row in [row_a, row_b]:
        if int(row["minutes_played"]) < MINUTES_WARNING_THRESHOLD:
            warnings.append(
                f"⚠️  {row['name']}: solo {int(row['minutes_played'])} min — muestra pequeña"
            )

    metrics = [
        ("Posición",              "position",              False),
        ("Minutos jugados",       "minutes_played",        False),
        ("Pases completados/90",  "passes_completed_p90",  True),
        ("Pases totales/90",      "passes_total_p90",      True),
        ("Tiros/90",              "shots_p90",             True),
        ("xG/90",                 "xg_p90",                True),
        ("Recuperaciones/90",     "ball_recoveries_p90",   True),
        ("Presiones/90",          "pressures_p90",         True),
        ("Regates completados/90","dribbles_won_p90",      True),
    ]

    col_w = 26
    name_a = row_a["name"][:16]
    name_b = row_b["name"][:16]

    header = f"{'Métrica':<{col_w}}  {name_a:<18}  {name_b:<18}"
    separator = "─" * len(header)

    rows = []
    for label, col, highlight in metrics:
        val_a = row_a[col]
        val_b = row_b[col]

        if highlight and isinstance(val_a, float):
            marker_a = " ◀" if val_a > val_b else "  "
            marker_b = " ◀" if val_b > val_a else "  "
            rows.append(f"{label:<{col_w}}  {val_a:<18.2f}{marker_a}  {val_b:<18.2f}{marker_b}")
        else:
            rows.append(f"{label:<{col_w}}  {str(val_a):<18}  {str(val_b):<18}")

    warning_block = ("\n" + "\n".join(warnings)) if warnings else ""

    result = (
        f"⚖️  Comparativa: {row_a['name']} vs {row_b['name']}\n"
        f"{separator}\n"
        f"{header}\n"
        f"{separator}\n"
        + "\n".join(rows)
        + f"\n{separator}"
        + warning_block
    )

    print(f"[comparar_jugadores] Tabla generada")
    return result


@tool
def buscar_jugadores(query: str, k: int = 5) -> str:
    """
    Busca jugadores por perfil de juego usando búsqueda semántica (embeddings).
    Útil cuando el usuario describe características deseadas en vez de un nombre.
    Ejemplos: 'mediocentro defensivo que recupera balones', 'delantero con alto xG'.

    Args:
        query: Descripción del perfil de juego buscado en lenguaje natural.
        k: Número de jugadores a devolver (default: 5).

    Returns:
        Lista de los k jugadores más similares al perfil descrito.
    """
    print(f"\n[buscar_jugadores] Query semántica: '{query}' (k={k})")

    print(f"[buscar_jugadores] Generando embedding de la query...")
    query_vec = _model.encode([query], convert_to_numpy=True)[0]

    print(f"[buscar_jugadores] Calculando similitud coseno contra {len(_embeddings)} jugadores...")
    similarities = _cosine_similarity(query_vec, _embeddings)
    top_indices = np.argsort(similarities)[::-1][:k]

    print(f"[buscar_jugadores] Top {k} resultados:")
    lines = [f"🔍 Jugadores más similares al perfil '{query}':\n"]
    for rank, idx in enumerate(top_indices, 1):
        name = _index_names[idx]
        score = similarities[idx]
        row = _stats_df[_stats_df["name"] == name]
        if row.empty:
            continue
        row = row.iloc[0]
        warning = " ⚠️ muestra pequeña" if int(row["minutes_played"]) < MINUTES_WARNING_THRESHOLD else ""
        line = (
            f"  {rank}. {name}"
            f" — {row['position']}"
            f" — similitud: {score:.3f}{warning}"
        )
        print(f"[buscar_jugadores]  {line.strip()}")
        lines.append(line)

    return "\n".join(lines)
