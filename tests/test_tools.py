"""
Tests de las funciones puras y tools de scout/tools.py.

Hay dos niveles:
  - Funciones puras (_fuzzy_score, _cosine_similarity): sin dependencias externas,
    rápidas y deterministas. Se pueden correr sin datos ni modelos.
  - Tools con datos (_find_candidates, stats_jugador, comparar_jugadores): requieren
    player_stats.csv, player_profiles.json y embeddings.npy en data/.

Si el pipeline de datos no se corrió todavía, los tests que requieren datos
son saltados automáticamente.
"""

from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).parent.parent / "data"
_data_available = (DATA_DIR / "player_stats.csv").exists()

requires_data = pytest.mark.skipif(
    not _data_available,
    reason="player_stats.csv no encontrado — corré scripts/build_player_profiles.py primero",
)


# ── Funciones puras ───────────────────────────────────────────────────────────

class TestFuzzyScore:
    """
    _fuzzy_score mide similitud entre strings (0.0 a 1.0).
    Usa SequenceMatcher (Ratcliff/Obershelp) — sensible al largo relativo de los strings.
    """

    def test_strings_identicos(self):
        from scout.tools import _fuzzy_score
        assert _fuzzy_score("Messi", "Messi") == 1.0

    def test_case_insensitive(self):
        from scout.tools import _fuzzy_score
        assert _fuzzy_score("messi", "MESSI") == 1.0

    def test_substring_tiene_score_alto(self):
        from scout.tools import _fuzzy_score
        score = _fuzzy_score("Messi", "Lionel Messi")
        assert score > 0.5

    def test_strings_sin_relacion_tienen_score_bajo(self):
        from scout.tools import _fuzzy_score
        score = _fuzzy_score("Messi", "Ronaldo")
        assert score < 0.3

    def test_strings_vacios(self):
        from scout.tools import _fuzzy_score
        assert _fuzzy_score("", "") == 1.0
        assert _fuzzy_score("Messi", "") == 0.0


class TestCosineSimilarity:
    """
    _cosine_similarity calcula el ángulo entre un vector query y cada fila de una matriz.
    Score 1.0 = misma dirección, 0.0 = perpendiculares, -1.0 = opuestos.
    """

    def test_vectores_identicos(self):
        from scout.tools import _cosine_similarity
        v = np.array([1.0, 0.0, 0.0])
        m = np.array([[1.0, 0.0, 0.0]])
        assert _cosine_similarity(v, m)[0] == pytest.approx(1.0)

    def test_vectores_ortogonales(self):
        from scout.tools import _cosine_similarity
        v = np.array([1.0, 0.0])
        m = np.array([[0.0, 1.0]])
        assert _cosine_similarity(v, m)[0] == pytest.approx(0.0)

    def test_vectores_opuestos(self):
        from scout.tools import _cosine_similarity
        v = np.array([1.0, 0.0])
        m = np.array([[-1.0, 0.0]])
        assert _cosine_similarity(v, m)[0] == pytest.approx(-1.0)

    def test_multiples_filas(self):
        from scout.tools import _cosine_similarity
        v = np.array([1.0, 0.0])
        m = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        scores = _cosine_similarity(v, m)
        assert len(scores) == 3
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.0)
        assert scores[2] == pytest.approx(-1.0)

    def test_vector_cero_no_divide_por_cero(self):
        from scout.tools import _cosine_similarity
        v = np.array([0.0, 0.0])
        m = np.array([[1.0, 0.0]])
        scores = _cosine_similarity(v, m)
        assert np.isfinite(scores[0])


# ── Tools con datos ───────────────────────────────────────────────────────────

@requires_data
class TestFindCandidates:
    """
    _find_candidates hace fuzzy matching sobre el índice real de jugadores.
    """

    def test_nombre_exacto_es_primer_resultado(self):
        from scout.tools import _find_candidates
        candidates = _find_candidates("Lionel Andrés Messi Cuccittini", k=5)
        assert candidates[0]["name"] == "Lionel Andrés Messi Cuccittini"

    def test_apellido_parcial_encuentra_jugador(self):
        from scout.tools import _find_candidates
        candidates = _find_candidates("Busquets", k=5)
        assert any("Busquets" in c["name"] for c in candidates)

    def test_respeta_limite_k(self):
        from scout.tools import _find_candidates
        candidates = _find_candidates("Garcia", k=3)
        assert len(candidates) <= 3

    def test_candidato_tiene_campos_requeridos(self):
        from scout.tools import _find_candidates
        candidates = _find_candidates("Messi", k=1)
        assert len(candidates) >= 1
        c = candidates[0]
        assert "name" in c
        assert "position" in c
        assert "minutes_played" in c
        assert "similarity_score" in c
        assert 0.0 <= c["similarity_score"] <= 1.0


@requires_data
class TestStatsJugador:
    """
    stats_jugador devuelve las métricas por 90 de un jugador dado su nombre.
    Hace fuzzy matching interno cuando el nombre no es exacto.
    """

    def test_nombre_exacto(self):
        from scout.tools import stats_jugador
        result = stats_jugador.invoke("Lionel Andrés Messi Cuccittini")
        assert "Messi" in result
        assert "Minutos jugados" in result
        assert "xG" in result

    def test_nombre_parcial_via_fuzzy(self):
        from scout.tools import stats_jugador
        result = stats_jugador.invoke("Busquets")
        assert "Busquets" in result
        assert "Minutos jugados" in result

    def test_fuzzy_siempre_devuelve_algo(self):
        """
        El fuzzy matching nunca devuelve vacío — siempre encuentra el candidato
        más similar, incluso con un score bajo. Documentamos este comportamiento
        porque afecta el UX: si el usuario tipea mal un nombre, igual obtiene
        un resultado (posiblemente incorrecto).
        """
        from scout.tools import stats_jugador
        result = stats_jugador.invoke("ZZZ_jugador_que_no_existe_XYZ")
        # Siempre devuelve stats de alguien, no "No encontré"
        assert "Minutos jugados" in result


@requires_data
class TestCompararJugadores:
    """
    comparar_jugadores genera una tabla comparativa entre dos jugadores.
    """

    def test_comparacion_valida(self):
        from scout.tools import comparar_jugadores
        result = comparar_jugadores.invoke({
            "nombre_a": "Lionel Andrés Messi Cuccittini",
            "nombre_b": "Cristiano Ronaldo dos Santos Aveiro",
        })
        assert "Messi" in result
        assert "Ronaldo" in result
        assert "xG" in result
        assert "◀" in result  # marcador del mejor en cada métrica

    def test_fuzzy_resuelve_nombre_desconocido(self):
        """
        Mismo comportamiento que stats_jugador — el fuzzy siempre encuentra algo.
        Un nombre inventado termina comparándose con el jugador más "parecido"
        fonéticamente, lo que puede sorprender al usuario.
        """
        from scout.tools import comparar_jugadores
        result = comparar_jugadores.invoke({
            "nombre_a": "ZZZ_no_existe",
            "nombre_b": "Lionel Andrés Messi Cuccittini",
        })
        # Encuentra a alguien para comparar, aunque no sea el esperado
        assert "Messi" in result
        assert "◀" in result
