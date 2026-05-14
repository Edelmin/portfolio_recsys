from dataclasses import dataclass
from pathlib import Path
import math
from pandas import DataFrame
import pandas as pd


@dataclass
class SampleData:
    population_size: int
    confidence_z: float = 1.96
    p: float = 0.5
    epsilon: float = 0.05
    path: Path | None = None

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

    def summary(self) -> str:
        n = self.sample_size
        return f"El tamaño de la muestra será {n} de {self.population_size}" + (
            f", en {self.path}" if self.path else ""
        )


def get_sample(dataframe: DataFrame, aditional_margin=0) -> DataFrame:

    seed = 42

    sample_info = SampleData(dataframe.shape[0])

    sample_size = sample_info.sample_size

    dataframe = dataframe.sample(n=sample_size + aditional_margin, random_state=seed)

    return dataframe
