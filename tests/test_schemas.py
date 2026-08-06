"""
Tests del modelo de output estructurado InformeScouting.

Estos tests son completamente independientes — no requieren datos ni modelos.
Verifican que el modelo Pydantic serializa, deserializa y expone sus campos
correctamente. Son los más rápidos del suite.
"""

import pytest
from scout.schemas import InformeScouting


class TestInformeScouting:
    def test_round_trip_json(self):
        """Serializar a JSON y deserializar debe reproducir el objeto exactamente."""
        original = InformeScouting(
            query="dame las stats de Busquets",
            agentes_ejecutados=["stats"],
            estadisticas="📊 Stats de Sergio Busquets...",
            conclusion="Busquets es un mediocampista de élite.",
        )
        recovered = InformeScouting.model_validate_json(original.model_dump_json())

        assert recovered.query == original.query
        assert recovered.agentes_ejecutados == original.agentes_ejecutados
        assert recovered.estadisticas == original.estadisticas
        assert recovered.conclusion == original.conclusion

    def test_campos_vacios_por_defecto(self):
        """Los campos de resultado son strings vacíos por defecto, no None."""
        informe = InformeScouting(query="test", agentes_ejecutados=[])

        assert informe.jugadores_sugeridos == ""
        assert informe.estadisticas == ""
        assert informe.comparativa == ""
        assert informe.conclusion == ""
        assert informe.jugadores_detectados == []

    def test_jugadores_detectados_en_json(self):
        """La lista de jugadores detectados sobrevive el round-trip JSON."""
        informe = InformeScouting(
            query="compará a Messi con Ronaldo",
            agentes_ejecutados=["comp"],
            jugadores_detectados=[
                "Lionel Andrés Messi Cuccittini",
                "Cristiano Ronaldo dos Santos Aveiro",
            ],
        )
        recovered = InformeScouting.model_validate_json(informe.model_dump_json())
        assert len(recovered.jugadores_detectados) == 2
        assert "Messi" in recovered.jugadores_detectados[0]

    def test_display_incluye_todas_las_secciones(self):
        """display() debe incluir el contenido de cada sección no vacía."""
        informe = InformeScouting(
            query="test",
            agentes_ejecutados=["rag", "stats", "comp"],
            jugadores_sugeridos="Jugador A",
            estadisticas="Stats B",
            comparativa="Tabla C",
            conclusion="Conclusión D",
        )
        texto = informe.display()

        assert "Jugador A" in texto
        assert "Stats B" in texto
        assert "Tabla C" in texto
        assert "Conclusión D" in texto

    def test_display_omite_secciones_vacias(self):
        """display() no debe incluir headers de secciones que están vacías."""
        informe = InformeScouting(
            query="test",
            agentes_ejecutados=["stats"],
            estadisticas="Solo stats",
        )
        texto = informe.display()

        assert "Solo stats" in texto
        assert "Jugadores sugeridos" not in texto
        assert "Comparativa" not in texto

    def test_multiples_agentes_en_lista(self):
        """agentes_ejecutados acepta cualquier combinación válida."""
        for agentes in [["rag"], ["stats"], ["comp"], ["rag", "stats"], ["stats", "comp"]]:
            informe = InformeScouting(query="x", agentes_ejecutados=agentes)
            assert informe.agentes_ejecutados == agentes
