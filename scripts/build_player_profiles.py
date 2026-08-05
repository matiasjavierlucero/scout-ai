"""
Paso 1: Pipeline de datos
Carga eventos de La Liga (StatsBomb Open Data), agrega stats por jugador
por 90 minutos y genera perfiles textuales para el índice RAG.

Salida:
  data/player_stats.csv       — stats agregadas por jugador
  data/player_profiles.json   — perfil textual por jugador
"""

import json
import os
import warnings
from pathlib import Path

import pandas as pd
from statsbombpy import sb

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

COMPETITION_ID = 11  # La Liga


def load_all_matches() -> pd.DataFrame:
    competitions = sb.competitions()
    la_liga = competitions[competitions["competition_id"] == COMPETITION_ID]
    print(f"Temporadas disponibles: {len(la_liga)}")

    all_matches = []
    for _, row in la_liga.iterrows():
        matches = sb.matches(
            competition_id=COMPETITION_ID, season_id=int(row["season_id"])
        )
        all_matches.append(matches)
        print(f"  {row['season_name']}: {len(matches)} partidos")

    return pd.concat(all_matches, ignore_index=True)


def extract_events(match_ids: list[int]) -> pd.DataFrame:
    all_events = []
    total = len(match_ids)

    for i, match_id in enumerate(match_ids, 1):
        if i % 50 == 0 or i == total:
            print(f"  Procesando partido {i}/{total}...")
        events = sb.events(match_id=match_id)
        events["match_id"] = match_id
        all_events.append(events)

    return pd.concat(all_events, ignore_index=True)


def compute_minutes_played(events: pd.DataFrame) -> pd.Series:
    """
    Minutos jugados totales por jugador, sumando el minuto máximo por partido.

    Lógica: para cada (jugador, partido) tomamos el minuto más alto en que
    tuvo un evento — proxy de cuánto jugó en ese partido. Luego sumamos
    esos valores across todos sus partidos para obtener el total acumulado.

    Esto corrige el bug de usar max() global (que devolvía ~90 min sin importar
    cuántos partidos jugó el jugador, inflando todas las stats por 90).
    """
    per_match = events.groupby(["player", "match_id"])["minute"].max()
    return per_match.groupby("player").sum().rename("minutes_played")


def aggregate_stats(events: pd.DataFrame) -> pd.DataFrame:
    players = events[events["player"].notna()].copy()

    # Posición más frecuente
    positions = (
        players[players["position"].notna()]
        .groupby("player")["position"]
        .agg(lambda x: x.value_counts().index[0])
        .rename("position")
    )

    minutes = compute_minutes_played(players)

    # Pases completados
    passes = players[players["type"] == "Pass"]
    pass_complete = (
        passes[passes["pass_outcome"].isna()]  # NaN = completado en StatsBomb
        .groupby("player")
        .size()
        .rename("passes_completed")
    )
    pass_total = passes.groupby("player").size().rename("passes_total")

    # Tiros y xG
    shots = players[players["type"] == "Shot"]
    shot_count = shots.groupby("player").size().rename("shots")
    xg = shots.groupby("player")["shot_statsbomb_xg"].sum().rename("xg")

    # Recuperaciones de balón
    recoveries = (
        players[players["type"] == "Ball Recovery"]
        .groupby("player")
        .size()
        .rename("ball_recoveries")
    )

    # Presiones
    pressures = (
        players[players["type"] == "Pressure"]
        .groupby("player")
        .size()
        .rename("pressures")
    )

    # Regates completados
    dribbles = players[players["type"] == "Dribble"]
    dribbles_won = (
        dribbles[dribbles["dribble_outcome"] == "Complete"]
        .groupby("player")
        .size()
        .rename("dribbles_won")
    )

    stats = pd.concat(
        [
            positions,
            minutes,
            pass_complete,
            pass_total,
            shot_count,
            xg,
            recoveries,
            pressures,
            dribbles_won,
        ],
        axis=1,
    ).fillna(0)

    # Filtrar jugadores con muy pocos minutos (datos poco representativos)
    stats = stats[stats["minutes_played"] >= 90]

    # Normalizar a por 90 minutos
    per90_cols = [
        "passes_completed",
        "passes_total",
        "shots",
        "xg",
        "ball_recoveries",
        "pressures",
        "dribbles_won",
    ]
    for col in per90_cols:
        stats[f"{col}_p90"] = (stats[col] / stats["minutes_played"] * 90).round(2)

    return stats.reset_index().rename(columns={"player": "name"})


def build_profile(row: pd.Series) -> str:
    return (
        f"{row['name']} es un jugador de {row['position']} con {int(row['minutes_played'])} "
        f"minutos jugados. Stats por 90 min: "
        f"{row['passes_completed_p90']} pases completados, "
        f"{row['shots_p90']} tiros, "
        f"{row['xg_p90']:.2f} xG, "
        f"{row['ball_recoveries_p90']} recuperaciones, "
        f"{row['pressures_p90']} presiones, "
        f"{row['dribbles_won_p90']} regates completados."
    )


def main():
    print("=== Paso 1: Pipeline de datos ===\n")

    print("1. Cargando partidos de La Liga...")
    matches = load_all_matches()
    print(f"   Total: {len(matches)} partidos\n")

    print("2. Descargando eventos (esto tarda varios minutos)...")
    events = extract_events(matches["match_id"].tolist())
    print(f"   Total eventos: {len(events):,}\n")

    print("3. Agregando stats por jugador...")
    stats = aggregate_stats(events)
    print(f"   Jugadores con datos suficientes: {len(stats)}\n")

    print("4. Guardando player_stats.csv...")
    stats.to_csv(DATA_DIR / "player_stats.csv", index=False)

    print("5. Generando perfiles textuales...")
    profiles = {row["name"]: build_profile(row) for _, row in stats.iterrows()}
    with open(DATA_DIR / "player_profiles.json", "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

    print(f"\n✓ {len(profiles)} perfiles generados")
    print(f"✓ data/player_stats.csv")
    print(f"✓ data/player_profiles.json")
    print("\nEjemplo de perfil:")
    sample = next(iter(profiles.values()))
    print(f"  {sample}")


if __name__ == "__main__":
    main()
