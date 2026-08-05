"""
Paso 6: Output estructurado con Pydantic.

InformeScouting es el modelo de salida final del sistema.
Cada campo corresponde al resultado de un agente que corrió.
Los campos de agentes no ejecutados quedan como string vacío.
"""

from pydantic import BaseModel, Field


class InformeScouting(BaseModel):
    query: str = Field(description="Query original del usuario")
    agentes_ejecutados: list[str] = Field(description="Agentes que corrieron: rag, stats, comp")
    jugadores_sugeridos: str = Field(default="", description="Resultado del agente RAG")
    estadisticas: str = Field(default="", description="Resultado del agente Stats")
    comparativa: str = Field(default="", description="Resultado del agente Comp")
    conclusion: str = Field(default="", description="Conclusión generada por el LLM")
    jugadores_detectados: list[str] = Field(default_factory=list, description="Jugadores resueltos en esta query")

    def display(self) -> str:
        """Formatea el informe completo para mostrar en terminal."""
        sections = [
            f"🔎 Query: {self.query}",
            f"⚙️  Agentes ejecutados: {', '.join(self.agentes_ejecutados)}",
        ]

        if self.jugadores_sugeridos:
            sections.append(f"\n## Jugadores sugeridos por perfil\n\n{self.jugadores_sugeridos}")

        if self.estadisticas:
            sections.append(f"\n## Estadísticas\n\n{self.estadisticas}")

        if self.comparativa:
            sections.append(f"\n## Comparativa\n\n{self.comparativa}")

        if self.conclusion:
            sections.append(f"\n## Conclusión\n\n{self.conclusion}")

        return "\n".join(sections)
