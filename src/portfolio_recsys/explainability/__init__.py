"""Modulo de explicabilidad de las decisiones de inversion.

Componente independiente y de SOLO LECTURA sobre los artefactos ya generados
(predicciones de test, resultados de backtesting y checkpoints de modelo). No
interviene en el entrenamiento ni altera el flujo de decision del sistema.

Implementa el enfoque minimo de explicabilidad acordado, distribuido segun la
naturaleza de cada etapa:

- Trazabilidad de la cadena de decision (etapas interpretables por diseno:
  seleccion top-K y ordenacion): `traceability`.
- Descomposicion de la optimizacion de Markowitz (etapa interpretable por
  diseno): `markowitz_decomposition`.
- Atribucion de importancia de variables de la red recurrente mediante
  Integrated Gradients (unico componente opaco): `attribution`.
"""

from __future__ import annotations

__all__ = [
    "traceability",
    "markowitz_decomposition",
    "attribution",
]
