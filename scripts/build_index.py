"""
Paso 2: Índice RAG
Genera embeddings de los perfiles textuales y los guarda en disco.

Salida:
  data/embeddings.npy   — matriz (N jugadores * 384 dimensiones)
  data/index.json       — mapeo nombre → índice de fila
"""

import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    print("=== Paso 2: Índice RAG ===\n")

    print("1. Cargando perfiles textuales...")
    with open(DATA_DIR / "player_profiles.json", encoding="utf-8") as f:
        profiles = json.load(f)
    names = list(profiles.keys())
    texts = list(profiles.values())
    print(f"   {len(names)} jugadores cargados\n")

    print(f"2. Cargando modelo {MODEL_NAME}...")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"   Modelo listo en {time.time() - t0:.1f}s")
    print(f"   Dimensiones del vector: {model.get_embedding_dimension()}\n")

    print("3. Generando embeddings...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"   Embeddings generados en {time.time() - t0:.1f}s")
    print(f"   Shape de la matriz: {embeddings.shape}\n")

    print("4. Guardando embeddings.npy...")
    np.save(DATA_DIR / "embeddings.npy", embeddings)
    size_mb = (DATA_DIR / "embeddings.npy").stat().st_size / 1024 / 1024
    print(f"   Guardado ({size_mb:.2f} MB)\n")

    print("5. Guardando index.json...")
    index = {name: i for i, name in enumerate(names)}
    with open(DATA_DIR / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"   {len(index)} entradas guardadas\n")

    print("=== Listo ===")
    print(f"✓ data/embeddings.npy  — {embeddings.shape[0]} vectores de {embeddings.shape[1]} dims")
    print(f"✓ data/index.json      — mapeo nombre → fila")


if __name__ == "__main__":
    main()
