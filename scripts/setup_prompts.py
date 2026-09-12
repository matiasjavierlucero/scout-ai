"""
Registra los system prompts de los agentes en Langfuse Prompt Management.

Corre una sola vez para subir los prompts iniciales. Después los editás
directamente desde el dashboard de Langfuse — sin tocar código ni hacer deploy.
Cada edición en el dashboard crea una nueva versión con rollback disponible.

Uso:
  uv run python scripts/setup_prompts.py
"""

from dotenv import load_dotenv
load_dotenv()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scout.tracing import get_langfuse
from scout.prompts import PROMPTS


def main() -> None:
    langfuse = get_langfuse()
    if langfuse is None:
        print("ERROR: Langfuse no configurado.")
        print("Verificá que LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY y LANGFUSE_BASE_URL estén en el .env")
        sys.exit(1)

    print(f"Registrando {len(PROMPTS)} prompts en Langfuse...\n")

    for name, text in PROMPTS.items():
        langfuse.create_prompt(
            name=name,
            type="text",
            prompt=text,
            labels=["production"],
        )
        print(f"  ✓ '{name}' — {len(text)} caracteres")

    langfuse.flush()
    print(f"\nListo. Abrí Langfuse → Prompt Management para verlos y editarlos.")
    print("Cada edición desde el dashboard crea una nueva versión automáticamente.")


if __name__ == "__main__":
    main()
