"""
Paso 2b: Enriquecimiento de perfiles con Wikipedia.

Toma los perfiles textuales generados por build_player_profiles.py y los enriquece
con un resumen biográfico desde Wikipedia. El resultado hace que el agente RAG sea
dramáticamente más potente para queries semánticas que dependen de contexto biográfico:

  Antes:  "jugador argentino que revolucionó el fútbol"     → 0 resultados relevantes
  Después: query anterior → encuentra a Messi, Di María, Higuaín, etc.

  Antes:  "delantero portugués que jugó en España"          → 0 resultados relevantes
  Después: query anterior → encuentra a Ronaldo, Nani, etc.

La razón es que los perfiles originales solo contienen stats agregadas — sin nombre de país,
sin historia del jugador, sin contexto narrativo. Wikipedia aporta exactamente eso.

Estrategia de enriquecimiento:
  1. Para cada jugador, buscar en Wikipedia ES (español) con su nombre oficial.
  2. Si no hay resultado en ES o el resumen es muy corto, fallback a Wikipedia EN.
  3. Si tampoco hay resultado en EN, el perfil queda sin enriquecer (sin error — no todos
     los jugadores de La Liga tienen artículo en Wikipedia).
  4. El resumen se concatena al perfil existente con el prefijo "Contexto: ".

API usada:
  MediaWiki API — acción "query" con generator=search + prop=extracts.
  Combina búsqueda y extracción en una sola petición HTTP (vs. dos peticiones separadas).
  No requiere autenticación. Límite real: ~200 req/s, nosotros usamos 0.3s entre requests.

Cache:
  Resultados guardados en data/wikipedia_cache.json para que el script pueda interrumpirse
  y reanudarse sin repetir peticiones. Si ya procesaste 1000 jugadores y el script se cae,
  al relanzar reanuda desde el 1001.

Uso:
  uv run python scripts/enrich_profiles_wikipedia.py

  El script modifica data/player_profiles.json in-place.
  Después de terminar, corré scripts/build_index.py para regenerar los embeddings.

Tiempo estimado:
  ~1885 jugadores × 0.3s de delay = ~9-12 minutos.
  (Variable según latencia de red a Wikipedia.)
"""

import json
import time
from pathlib import Path

import requests

# ── Configuración ─────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_FILE = DATA_DIR / "wikipedia_cache.json"

LANG_PRIMARY = "es"
LANG_FALLBACK = "en"

# Cantidad de oraciones del intro de Wikipedia a incluir en el perfil.
# 3 oraciones captura: qué es el jugador, nacionalidad y posición, club/carrera principal.
EXTRACT_SENTENCES = 3

# Si el resumen tiene menos de MIN_EXTRACT_LEN chars, se considera insuficiente
# y se intenta con el idioma de fallback.
MIN_EXTRACT_LEN = 50

# Delay entre peticiones HTTP para ser respetuosos con la API de Wikipedia.
# 1.0s garantiza ~1 req/s, bien dentro del límite real observado.
REQUEST_DELAY_SECONDS = 1.0

# Reintentos en caso de 429 Too Many Requests — la API de Wikipedia aplica
# rate limiting por ráfagas (burst), no solo por req/s. Con backoff exponencial
# de 2^intento segundos: 2s, 4s, 8s antes de rendirse.
MAX_RETRIES = 3

# User-Agent descriptivo — MediaWiki pide que los bots se identifiquen.
USER_AGENT = (
    "ScoutAI/1.0 "
    "(proyecto educativo de análisis de fútbol sobre StatsBomb Open Data; "
    "sin uso comercial)"
)


# ── Funciones de acceso a Wikipedia ──────────────────────────────────────────

def _fetch_from_wikipedia(name: str, lang: str) -> str | None:
    """
    Realiza una búsqueda en Wikipedia y devuelve el extracto introductorio del primer resultado.

    Usa el endpoint de MediaWiki API con generator=search combinado con prop=extracts,
    lo que nos permite obtener búsqueda + texto en una sola petición HTTP.

    Parámetros de la API:
      - generator=search + gsrsearch: busca artículos cuyo título o texto coincide con `name`.
      - gsrlimit=1: solo queremos el resultado más relevante.
      - prop=extracts: pide el texto del artículo.
      - exintro=true: solo la sección introductoria (antes del primer == Sección ==).
      - explaintext=true: texto plano sin markup de wikitext ni HTML.
      - exsentences=3: limita a las primeras N oraciones (evita respuestas gigantes).

    Args:
        name: Nombre oficial del jugador (puede ser nombre completo con apellidos).
        lang: Código de idioma — "es" para español, "en" para inglés.

    Returns:
        Texto del extracto introductorio (str), o None si:
        - No se encontró ningún artículo para `name`.
        - El extracto encontrado tiene menos de MIN_EXTRACT_LEN caracteres (artículo stub).
        - Ocurrió cualquier error HTTP, de red o de parseo.
    """
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": name,
        "gsrlimit": 1,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "exsentences": EXTRACT_SENTENCES,
        "format": "json",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=10,
                headers={"User-Agent": USER_AGENT},
            )

            if response.status_code == 429:
                # La API puede incluir Retry-After en segundos; si no, usamos backoff exponencial.
                retry_after = int(response.headers.get("Retry-After", 2 ** attempt))
                print(f"    [wikipedia] 429 para '{name}' ({lang}) — esperando {retry_after}s (intento {attempt}/{MAX_RETRIES})")
                time.sleep(retry_after)
                continue

            response.raise_for_status()
            data = response.json()
            break  # petición exitosa — salir del loop de reintentos

        except requests.exceptions.HTTPError as exc:
            if attempt == MAX_RETRIES:
                print(f"    [wikipedia] Error HTTP para '{name}' ({lang}): {exc}")
                return None
        except Exception as exc:
            # Errores de red, timeout, JSON malformado — no propagamos para que el script
            # continúe con el siguiente jugador. El cache guardará found=False.
            print(f"    [wikipedia] Error para '{name}' ({lang}): {exc}")
            return None
    else:
        # Agotamos los reintentos sin éxito
        return None

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return None

    # El resultado viene como dict {page_id: page_obj} — tomamos el primero y único.
    page = next(iter(pages.values()))

    # Si el artículo fue redirigido a -1 (missing), pages contiene {"-1": {...}}
    if str(list(pages.keys())[0]) == "-1":
        return None

    extract = page.get("extract", "").strip()

    # Extracts muy cortos indican artículos stub o redirecciones sin contenido.
    if len(extract) < MIN_EXTRACT_LEN:
        return None

    return extract


def _get_wikipedia_summary(player_name: str) -> tuple[str | None, str | None]:
    """
    Obtiene el resumen de Wikipedia para un jugador, probando ES y luego EN.

    Hace dos intentos en orden:
      1. Wikipedia en español (más informativa para jugadores de La Liga).
      2. Wikipedia en inglés (fallback para jugadores menos conocidos en ES).

    Después de cada petición, espera REQUEST_DELAY_SECONDS para respetar el rate limit.

    Args:
        player_name: Nombre oficial del jugador (e.g. "Lionel Andrés Messi Cuccittini").

    Returns:
        Tupla (summary, lang) donde:
          - summary: texto del extracto si se encontró, None si no.
          - lang: "es" o "en" indicando de qué Wikipedia viene, None si no se encontró.
    """
    # Intentar en español primero
    summary = _fetch_from_wikipedia(player_name, LANG_PRIMARY)
    time.sleep(REQUEST_DELAY_SECONDS)

    if summary:
        return summary, LANG_PRIMARY

    # Fallback a inglés
    summary = _fetch_from_wikipedia(player_name, LANG_FALLBACK)
    time.sleep(REQUEST_DELAY_SECONDS)

    if summary:
        return summary, LANG_FALLBACK

    return None, None


# ── Cache ─────────────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    """
    Carga el cache de Wikipedia desde disco si existe.

    El cache tiene la forma:
      {
        "Lionel Andrés Messi Cuccittini": {
          "summary": "Lionel Andrés Messi es un futbolista...",
          "lang": "es",
          "found": true
        },
        "Juan Pérez": {
          "summary": null,
          "lang": null,
          "found": false
        },
        ...
      }

    Returns:
        Dict con el cache existente, o dict vacío si el archivo no existe.
    """
    if CACHE_FILE.exists():
        with open(CACHE_FILE, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"[cache] Cache cargado — {len(cache)} entradas")
        return cache
    print("[cache] Cache nuevo — empezando desde cero")
    return {}


def _save_cache(cache: dict) -> None:
    """
    Persiste el cache a disco.

    Se llama periódicamente (cada 50 jugadores) y al finalizar el script.
    Si el proceso se interrumpe, los datos ya guardados no se pierden.

    Args:
        cache: Dict con todos los resultados procesados hasta ahora.
    """
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ── Enriquecimiento de perfiles ───────────────────────────────────────────────

def enrich_profile(original_profile: str, wikipedia_summary: str) -> str:
    """
    Concatena el resumen de Wikipedia al perfil estadístico existente.

    Formato resultante:
      "{stats_profile}. Contexto: {wikipedia_summary}"

    El prefijo "Contexto:" ayuda al modelo de embeddings a distinguir semánticamente
    qué parte del texto es datos estructurados y qué parte es contexto narrativo.
    Aunque sentence-transformers no interpreta el prefijo literalmente, el cambio de
    registro lingüístico (números → prosa) ya captura esa distinción en el espacio vectorial.

    Args:
        original_profile: Perfil generado por build_player_profiles.py.
        wikipedia_summary: Resumen biográfico de Wikipedia.

    Returns:
        Perfil enriquecido listo para reindexar.
    """
    # Asegurar que el perfil original no termine en punto doble al concatenar
    base = original_profile.rstrip(".")
    return f"{base}. Contexto: {wikipedia_summary}"


# ── Script principal ──────────────────────────────────────────────────────────

def main() -> None:
    """
    Pipeline completo de enriquecimiento:
      1. Carga player_profiles.json (generado por build_player_profiles.py).
      2. Carga el cache existente (si el script se corrió antes parcialmente).
      3. Para cada jugador sin entrada en cache, busca en Wikipedia.
      4. Guarda el cache cada 50 jugadores (checkpoint para recuperación).
      5. Genera nuevos perfiles enriquecidos y los guarda en player_profiles.json.

    El script NO modifica player_stats.csv ni el índice — después de correrlo,
    hay que regenerar embeddings con: uv run python scripts/build_index.py
    """
    print("=== Enriquecimiento de perfiles con Wikipedia ===\n")

    # 1. Cargar perfiles originales
    profiles_path = DATA_DIR / "player_profiles.json"
    print(f"Cargando perfiles desde {profiles_path}...")
    with open(profiles_path, encoding="utf-8") as f:
        profiles: dict[str, str] = json.load(f)
    print(f"   {len(profiles)} perfiles cargados\n")

    # 2. Cargar cache
    cache = _load_cache()

    player_names = list(profiles.keys())
    total = len(player_names)

    # Estadísticas de progreso
    stats = {"found_es": 0, "found_en": 0, "not_found": 0, "cached": 0}

    # 3. Buscar en Wikipedia para jugadores no cacheados
    print(f"Buscando en Wikipedia ({total} jugadores)...\n")

    for i, name in enumerate(player_names, 1):
        if name in cache:
            stats["cached"] += 1
            if cache[name]["found"]:
                if cache[name]["lang"] == "es":
                    stats["found_es"] += 1
                else:
                    stats["found_en"] += 1
            else:
                stats["not_found"] += 1
            continue

        # Buscar en Wikipedia
        summary, lang = _get_wikipedia_summary(name)

        if summary:
            cache[name] = {"summary": summary, "lang": lang, "found": True}
            if lang == "es":
                stats["found_es"] += 1
            else:
                stats["found_en"] += 1
            print(f"  [{i}/{total}] ✓ {name[:40]:<40} ({lang})")
        else:
            cache[name] = {"summary": None, "lang": None, "found": False}
            stats["not_found"] += 1
            print(f"  [{i}/{total}] ✗ {name[:40]:<40} (sin artículo)")

        # Checkpoint cada 50 jugadores para no perder progreso si hay un corte
        if i % 50 == 0:
            _save_cache(cache)
            found_total = stats["found_es"] + stats["found_en"]
            print(
                f"\n  [checkpoint {i}/{total}] "
                f"Encontrados: {found_total} | "
                f"Sin artículo: {stats['not_found']} | "
                f"Cacheados: {stats['cached']}\n"
            )

    # Guardar cache final
    _save_cache(cache)

    # 4. Generar perfiles enriquecidos
    print("\nGenerando perfiles enriquecidos...")
    enriched_profiles: dict[str, str] = {}
    enriched_count = 0

    for name, original in profiles.items():
        entry = cache.get(name, {})
        if entry.get("found") and entry.get("summary"):
            enriched_profiles[name] = enrich_profile(original, entry["summary"])
            enriched_count += 1
        else:
            # Jugador sin artículo en Wikipedia — perfil original intacto
            enriched_profiles[name] = original

    # 5. Guardar perfiles enriquecidos
    print(f"Guardando {len(enriched_profiles)} perfiles en {profiles_path}...")
    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(enriched_profiles, f, ensure_ascii=False, indent=2)

    # ── Resumen final ─────────────────────────────────────────────────────────
    found_total = stats["found_es"] + stats["found_en"]
    coverage_pct = found_total / total * 100

    print(f"""
╔══════════════════════════════════════════════════════╗
║              Enriquecimiento completado              ║
╠══════════════════════════════════════════════════════╣
║  Total jugadores:       {total:<6}                      ║
║  Enriquecidos (ES):     {stats['found_es']:<6}                      ║
║  Enriquecidos (EN):     {stats['found_en']:<6}                      ║
║  Sin artículo Wikipedia:{stats['not_found']:<6}                      ║
║  Cobertura:             {coverage_pct:.1f}%                      ║
╠══════════════════════════════════════════════════════╣
║  Archivos generados:                                 ║
║    data/player_profiles.json  (enriquecido)          ║
║    data/wikipedia_cache.json  (cache para re-uso)    ║
╚══════════════════════════════════════════════════════╝

Próximo paso:
  uv run python scripts/build_index.py
  (regenera embeddings con los perfiles enriquecidos)
""")

    # Mostrar ejemplo de perfil enriquecido
    sample_name = next(
        (n for n in player_names if cache.get(n, {}).get("found")),
        player_names[0],
    )
    print(f"Ejemplo de perfil enriquecido ({sample_name}):")
    print(f"  {enriched_profiles[sample_name][:300]}...")


if __name__ == "__main__":
    main()
