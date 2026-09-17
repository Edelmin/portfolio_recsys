"""Nodos del pipeline stratified_random_sampling.

Aplica muestreo estratificado sobre el universo de empresas, seleccionando
un subconjunto estadisticamente representativo por sector.
"""

import math
from dataclasses import dataclass

import polars as pl


@dataclass
class SampleData:
    population_size: int
    confidence_z: float = 1.96
    p: float = 0.5
    epsilon: float = 0.05

    @property
    def q(self) -> float:
        return 1 - self.p

    def compute(self) -> float:
        numerator = self.population_size * self.p * self.q * (self.confidence_z**2)
        denominator = (self.epsilon**2) * (
            self.population_size - 1
        ) + self.p * self.q * (self.confidence_z**2)
        return numerator / denominator

    @property
    def sample_size(self) -> int:
        return math.ceil(self.compute())


def get_sample(dataframe: pl.DataFrame, extra_samples: int = 0) -> pl.DataFrame:
    """Selecciona una muestra representativa del DataFrame de entrada.

    Calcula el tamano de muestra necesario para confianza 95% y error 5%,
    suma las muestras extra solicitadas, y devuelve una muestra aleatoria
    con semilla fija para reproducibilidad.

    Args:
        dataframe: Universo completo del sector.
        extra_samples: Empresas adicionales sobre el n estadistico.
                       Se aplica un cap al tamano de la poblacion.
    """
    seed = 42
    n_population = dataframe.shape[0]
    sample_info = SampleData(n_population)
    sample_size = min(sample_info.sample_size + extra_samples, n_population)

    return dataframe.sample(n=sample_size, seed=seed)
