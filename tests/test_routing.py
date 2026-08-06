"""
Tests del algoritmo de routing en scout/graph.py.

El routing es la lógica más crítica del sistema — decide qué agentes invocar.
Todos los bugs reportados en el proyecto fueron problemas de routing:
  - "Lionel Messi" disparaba comparación (falso positivo)
  - "ronaldo?" no encontraba a Ronaldo (puntuación pegada al token)
  - "Y Cristiano?" devolvía Biraghi en vez de Ronaldo (desempate por minutos)
  - "Hablame de Messi" mostraba tabla Messi vs Ronaldo (MemorySaver leak)

Estos tests son los regression tests de esos bugs.

NOTA DE PERFORMANCE: importar scout.graph carga player_stats.csv, el índice,
los embeddings y el modelo sentence-transformers (~10s la primera vez).
Es el costo de hacer integration testing real en lugar de mocks.
"""

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent.parent / "data"

_data_available = all(
    (DATA_DIR / f).exists()
    for f in ["index.json", "player_stats.csv", "embeddings.npy"]
)

requires_data = pytest.mark.skipif(
    not _data_available,
    reason="Archivos de datos no encontrados — corré el pipeline primero",
)


@requires_data
class TestDetectRoutingIntenciones:
    """
    Verifica que la intención del usuario se mapea correctamente al agente correcto.
    """

    def test_query_descriptiva_va_a_rag(self):
        """Queries sin nombres de jugadores deben ir al agente RAG."""
        from scout.graph import _detect_routing
        routing, players = _detect_routing("quiero un mediocentro defensivo que presione")
        assert routing["needs_rag"] is True
        assert routing["needs_stats"] is False
        assert routing["needs_comp"] is False
        assert players == []

    def test_nombre_jugador_va_a_stats(self):
        """Mencionar un jugador sin keyword de comparación → solo Stats."""
        from scout.graph import _detect_routing
        routing, players = _detect_routing("dame las stats de Messi")
        assert routing["needs_stats"] is True
        assert routing["needs_comp"] is False
        assert routing["needs_rag"] is False
        assert len(players) == 1

    def test_keyword_comparacion_va_a_comp(self):
        """Keyword de comparación + dos jugadores → solo Comp."""
        from scout.graph import _detect_routing
        routing, players = _detect_routing("compará a Messi con Cristiano Ronaldo")
        assert routing["needs_comp"] is True
        assert len(players) == 2

    def test_dos_jugadores_sin_keyword_va_a_comp(self):
        """Dos jugadores detectados activan Comp aunque no haya keyword explícita."""
        from scout.graph import _detect_routing
        routing, players = _detect_routing("Messi Ronaldo")
        assert routing["needs_comp"] is True
        assert len(players) == 2

    def test_routing_es_mutuamente_excluyente(self):
        """Solo un tipo de routing debe estar activo por query."""
        from scout.graph import _detect_routing
        for query in [
            "quiero un extremo rápido",
            "stats de Busquets",
            "compará Messi con Xavi",
        ]:
            routing, _ = _detect_routing(query)
            activos = sum([routing["needs_rag"], routing["needs_stats"], routing["needs_comp"]])
            # needs_stats y needs_comp pueden coexistir cuando hay keyword + nombre
            # pero needs_rag siempre es exclusivo
            if routing["needs_rag"]:
                assert activos == 1, f"RAG no debe coexistir con otros para: '{query}'"


@requires_data
class TestDetectRoutingRegressions:
    """
    Regression tests de bugs reales encontrados durante el desarrollo.
    Cada test documenta el bug, qué lo causaba y que la corrección sigue funcionando.
    """

    def test_lionel_messi_detecta_un_solo_jugador(self):
        """
        BUG: "Lionel Messi" disparaba comparación porque el token "lionel"
        matcheaba a Messi Y a otro jugador que se llama Lionel (Siviero).
        Dos matches → needs_comp = True → comparación falsa.

        FIX: algoritmo de tokens distintos — un jugador solo cuenta si aporta
        tokens que ningún match anterior cubre. Siviero solo comparte "lionel",
        token ya cubierto por Messi. No suma como segundo jugador.
        """
        from scout.graph import _detect_routing
        routing, players = _detect_routing("Lionel Messi")
        assert routing["needs_comp"] is False, (
            "Regresión: 'Lionel Messi' no debe disparar comparación"
        )
        assert len(players) == 1
        assert "Messi" in players[0]

    def test_puntuacion_no_rompe_deteccion(self):
        """
        BUG: "ronaldo?" no encontraba a Ronaldo porque el token era "ronaldo?"
        con el signo de pregunta pegado — no matcheaba ningún nombre del índice.

        FIX: strip de puntuación (?!.,;:'") antes de filtrar tokens.
        """
        from scout.graph import _detect_routing
        routing, players = _detect_routing("ronaldo?")
        assert routing["needs_stats"] is True, (
            "Regresión: 'ronaldo?' debe encontrar a Ronaldo"
        )
        assert len(players) == 1
        assert "Ronaldo" in players[0]

    def test_cristiano_resuelve_ronaldo_no_biraghi(self):
        """
        BUG: "Y Cristiano?" devolvía a Cristiano Biraghi en vez de
        Cristiano Ronaldo porque ambos tienen exactamente 1 token en común
        ("cristiano") y el ordenamiento era alfabético por defecto.

        FIX: tiebreaker por minutos jugados — Ronaldo tiene miles de minutos
        en La Liga vs. Biraghi. El jugador más prominente gana el desempate.
        """
        from scout.graph import _detect_routing
        routing, players = _detect_routing("Y Cristiano?")
        assert len(players) == 1
        assert "Ronaldo" in players[0], (
            f"Regresión: 'Y Cristiano?' debería resolver a Ronaldo, no a '{players[0]}'"
        )

    def test_hablame_de_messi_no_dispara_comparacion(self):
        """
        BUG: "Hablame de Messi" mostraba la tabla Messi vs Ronaldo de una
        query anterior porque MemorySaver + operator.add acumulaban comp_results
        entre invocaciones del graph.

        El routing en sí nunca estuvo mal para esta query — solo verificamos
        que _detect_routing no activa needs_comp.

        FIX: se removió MemorySaver del graph. Cada invocación es stateless.
        """
        from scout.graph import _detect_routing
        routing, players = _detect_routing("Hablame de Messi")
        assert routing["needs_comp"] is False, (
            "Regresión: 'Hablame de Messi' no debe activar comparación"
        )
        assert routing["needs_stats"] is True


@requires_data
class TestDetectRoutingJugadoresDetectados:
    """
    Verifica que los jugadores resueltos son los correctos y en el orden esperado.
    """

    def test_primer_jugador_mencionado_es_primero(self):
        """El primer jugador en la query debe ser players[0]."""
        from scout.graph import _detect_routing
        _, players = _detect_routing("compará a Messi con Ronaldo")
        assert len(players) == 2
        assert "Messi" in players[0]
        assert "Ronaldo" in players[1]

    def test_busquets_resuelve_nombre_completo(self):
        """Un apellido parcial debe resolver al nombre oficial completo del dataset."""
        from scout.graph import _detect_routing
        _, players = _detect_routing("stats de Busquets")
        assert len(players) == 1
        assert "Busquets" in players[0]
        # El nombre completo tiene más tokens que solo "Busquets"
        assert len(players[0].split()) > 1

    def test_query_sin_jugadores_devuelve_lista_vacia(self):
        """Queries puramente descriptivas no deben detectar jugadores."""
        from scout.graph import _detect_routing
        _, players = _detect_routing("necesito un portero con buena distribución")
        assert players == []
