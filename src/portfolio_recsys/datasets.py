"""Datasets personalizados de Kedro para el proyecto.

Contiene ``NpzDataset``, un dataset que serializa un diccionario de arrays
NumPy (y metadata JSON-compatible) a un fichero ``.npz`` comprimido. Se usa
para persistir el tensor de entrada DIRECTA a los modelos (las ventanas
temporales ya materializadas), de modo que el dato que consume PyTorch quede
guardado de forma inmutable y reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from kedro.io import AbstractDataset


class NpzDataset(AbstractDataset):
    """Persiste un dict de arrays NumPy + metadata en un fichero ``.npz``.

    El diccionario puede contener:
      - Arrays NumPy (se guardan como tales).
      - Valores JSON-compatibles (listas, int, str, dict): se serializan en una
        entrada auxiliar ``__meta__`` y se restauran al cargar.

    Al cargar, devuelve un dict con los mismos arrays y la metadata restaurada.
    """

    def __init__(self, filepath: str) -> None:
        self._filepath = Path(filepath)

    def _save(self, data: dict[str, Any]) -> None:
        self._filepath.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, np.ndarray] = {}
        meta: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                arrays[key] = value
            else:
                meta[key] = value

        # La metadata no-array se guarda como un array de un unico string JSON.
        arrays["__meta__"] = np.array(json.dumps(meta), dtype=object)
        np.savez_compressed(self._filepath, **arrays)

    def _load(self) -> dict[str, Any]:
        with np.load(self._filepath, allow_pickle=True) as handle:
            result: dict[str, Any] = {}
            meta: dict[str, Any] = {}
            for key in handle.files:
                if key == "__meta__":
                    meta = json.loads(str(handle[key].item()))
                    continue
                result[key] = handle[key]
        result.update(meta)
        return result

    def _describe(self) -> dict[str, Any]:
        return {"filepath": str(self._filepath)}

    def _exists(self) -> bool:
        return self._filepath.exists()
