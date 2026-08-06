# Scout AI

Agente de scouting táctico sobre datos de **StatsBomb Open Data** (La Liga).
Recibe queries en lenguaje natural y devuelve informes estructurados con búsqueda semántica, estadísticas por 90 minutos y comparativas entre jugadores.

## Demo

```
"quiero un mediocentro defensivo que presione y recupere balones"
"dame las stats de Busquets"
"compará a Messi con Cristiano Ronaldo"
"jugador argentino con capacidad de regate"
"comparalo con Xavi"
```

## Arquitectura

```
                    Orquestador (LangGraph)
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼       ← paralelo (Send API)
     Agente RAG      Agente Stats     Agente Comp
   (búsqueda por    (stats por 90    (comparativa
    perfil de       del jugador)     lado a lado)
      juego)
          │                │                │
          └────────────────┴────────────────┘
                           │
                    InformeScouting (Pydantic)
                           │
                      Streamlit UI
```

**Routing automático** — el orquestador analiza la query con heurísticas (tokens de nombres + keywords de comparación) y decide qué agentes invocar sin que el usuario especifique.

**Contexto conversacional** — queries de seguimiento como "comparalo con Ronaldo" o "y sus stats?" funcionan porque el sistema recuerda los jugadores del turno anterior.

## Stack

| Capa | Tecnología |
|---|---|
| LLM local | Ollama (`llama3.2:3b`) |
| Orquestación | LangGraph (`StateGraph` + `Send` API) |
| Embeddings | `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`) |
| Datos | StatsBomb Open Data via `statsbombpy` |
| Enriquecimiento | Wikipedia API (MediaWiki) |
| UI | Streamlit |
| Output estructurado | Pydantic |

## Requisitos

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) con el modelo `llama3.2:3b`

```bash
ollama pull llama3.2:3b
```

## Instalación

```bash
git clone <repo>
cd scout-ai
uv sync
```

## Setup del pipeline de datos

Los tres pasos son **one-time** — generan los archivos en `data/` que el app consume.

### Paso 1 — Pipeline de datos (~20-30 min)

Descarga todos los partidos de La Liga (15 temporadas, 2004-2021), extrae eventos y calcula stats agregadas por 90 minutos para cada jugador.

```bash
uv run python scripts/build_player_profiles.py
```

Genera:
- `data/player_stats.csv` — stats agregadas (1885 jugadores)
- `data/player_profiles.json` — perfiles textuales por jugador

### Paso 2 — Enriquecimiento Wikipedia (~35-40 min)

Agrega resúmenes biográficos de Wikipedia a cada perfil para mejorar la búsqueda semántica. Permite queries como "jugador argentino" o "delantero portugués que jugó en España".

Incluye cache en `data/wikipedia_cache.json` — si se interrumpe, retoma desde donde paró.

```bash
uv run python scripts/enrich_profiles_wikipedia.py
```

### Paso 3 — Índice RAG (~22 seg)

Genera embeddings multililngüe sobre los perfiles enriquecidos y los guarda como matriz numpy para búsqueda por similitud coseno.

```bash
uv run python scripts/build_index.py
```

Genera:
- `data/embeddings.npy` — matriz (1885 × 384)
- `data/index.json` — mapeo nombre → fila

## Correr el app

```bash
uv run streamlit run app.py
```

## Estructura del proyecto

```
scout-ai/
├── app.py                              # Streamlit UI + manejo de sesión
├── scout/
│   ├── agents.py                       # Agentes RAG, Stats y Comp (ReAct loop)
│   ├── graph.py                        # Orquestador LangGraph + routing
│   ├── tools.py                        # Tools con @tool decorator
│   └── schemas.py                      # InformeScouting (Pydantic)
├── scripts/
│   ├── build_player_profiles.py        # Paso 1: ETL StatsBomb → CSV/JSON
│   ├── enrich_profiles_wikipedia.py    # Paso 2: enriquecimiento Wikipedia
│   └── build_index.py                  # Paso 3: embeddings + índice
└── data/                               # Generado por los scripts (no committear)
    ├── player_stats.csv
    ├── player_profiles.json
    ├── embeddings.npy
    ├── index.json
    └── wikipedia_cache.json
```

## Dataset

**StatsBomb Open Data — La Liga**
- 15 temporadas (2004/05 → 2020/21)
- ~580 partidos
- 1885 jugadores con al menos 90 minutos jugados
- Métricas: pases completados, tiros, xG, recuperaciones, presiones, regates — normalizadas a por 90 minutos

Acceso via [`statsbombpy`](https://github.com/statsbomb/statsbombpy) — sin API key, datos abiertos.

## Cómo funciona el RAG

Cada jugador tiene un perfil textual con sus stats + contexto biográfico de Wikipedia:

```
"Lionel Andrés Messi Cuccittini es un jugador de Right Wing con 46189 minutos jugados.
Stats por 90 min: 44.58 pases completados, 4.58 tiros, 0.63 xG, 3.70 recuperaciones,
8.93 presiones, 5.74 regates completados. Contexto: Lionel Andrés Messi Cuccittini,
nacido el 24 de junio de 1987 en Rosario, Argentina, es un futbolista argentino que
juega como delantero..."
```

Al recibir una query descriptiva, se genera su embedding y se buscan los jugadores con mayor similitud coseno contra la matriz de 1885 vectores.

## Decisiones de diseño notables

**`operator.add` sin `MemorySaver`** — los campos de resultados usan `Annotated[list, operator.add]` para que los nodos paralelos puedan acumular resultados sin pisarse. El checkpointer está removido intencionalmente: con `MemorySaver` los resultados de queries anteriores se acumulan (el reducer es aditivo, no replace) y contaminan la query siguiente.

**`_extract_tool_output`** — `llama3.2:3b` parafrasea el output de las tools en vez de copiarlo. Los agentes toman el último `ToolMessage` directamente, saltando el mensaje final del LLM.

**Distinct-token algorithm** — para evitar falsos positivos en el routing (e.g., "Lionel Messi" detectando a otro jugador que comparte el token "lionel"), cada match solo cuenta si aporta tokens que ningún match anterior cubre. Tiebreaker por minutos jugados.
