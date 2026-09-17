"""Ventanas temporales y datasets PyTorch para modelos recurrentes.

Genera ventanas deslizantes de longitud fija sobre series temporales
financieras, detectando y descartando ventanas que cruzan gaps
excesivos (suspensiones, huecos de datos).

Produce:
- Arrays NumPy por ticker (features y targets).
- Indice Polars con metadata de cada ventana.
- Tensores materializados por split y loaders rapidos (GpuBatchLoader).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import torch

from portfolio_recsys.models.config import WindowConfig


def build_temporal_window_store(
    dataframe: pl.DataFrame,
    feature_columns: list[str],
    target_scaled_column: str = "target_scaled",
    target_original_column: str = "target_log_return",
    window_config: WindowConfig | None = None,
) -> dict[str, Any]:
    """Construye el almacen de ventanas temporales para entrenamiento.

    Para cada ticker, genera ventanas deslizantes de sequence_length
    sesiones consecutivas. Descarta ventanas donde algun gap entre
    sesiones consecutivas supera max_calendar_gap_days.

    Args:
        dataframe: DataFrame preprocesado con features escaladas, targets,
            split, ticker, date, trade_entry_date, trade_exit_date.
        feature_columns: Lista de columnas de features a incluir.
        target_scaled_column: Nombre de la columna con target escalado.
        target_original_column: Nombre de la columna con target original.
        window_config: Configuracion de ventanas. Si None, usa defaults.

    Returns:
        Diccionario con:
        - feature_arrays: dict[ticker, np.ndarray] con shape [T, F].
        - target_arrays: dict[ticker, np.ndarray] con shape [T].
        - window_index: pl.DataFrame con metadata de cada ventana.
        - sequence_length, feature_columns, y demas metadata.
    """
    if window_config is None:
        window_config = WindowConfig()

    sequence_length = window_config.sequence_length
    max_calendar_gap_days = window_config.max_calendar_gap_days

    # Filtro de ventanas anomalas (Medida 4 de control de calidad): solo se activa
    # si esta habilitado en la config Y la columna de marca existe en el dataframe.
    # Asi las llamadas que no proporcionan esa columna conservan su comportamiento.
    anomaly_column = getattr(window_config, "anomaly_column", "is_anomalous_return")
    filter_anomalous = (
        getattr(window_config, "filter_anomalous_windows", False)
        and anomaly_column in dataframe.columns
    )

    required_columns = set(feature_columns) | {
        target_scaled_column,
        target_original_column,
        "ticker",
        "date",
        "split",
        "sample_eligible",
        "trade_entry_date",
        "trade_exit_date",
    }

    missing = required_columns - set(dataframe.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el DataFrame: {sorted(missing)}")

    dataframe = dataframe.sort(["ticker", "date"])
    tickers = dataframe.get_column("ticker").unique().sort().to_list()

    feature_arrays: dict[str, np.ndarray] = {}
    target_arrays: dict[str, np.ndarray] = {}
    window_records: list[dict[str, Any]] = []
    sample_id_counter = 0

    for ticker in tickers:
        ticker_df = dataframe.filter(pl.col("ticker") == ticker).sort("date")

        if ticker_df.height < sequence_length:
            continue

        # Extraer arrays
        features = ticker_df.select(feature_columns).to_numpy().astype(np.float32)
        targets = ticker_df.select(target_scaled_column).to_numpy().flatten().astype(np.float32)

        feature_arrays[ticker] = features
        target_arrays[ticker] = targets

        # Calcular gaps de calendario entre sesiones consecutivas
        dates = ticker_df.get_column("date")
        calendar_gaps = (
            ticker_df.with_columns(
                pl.col("date").diff().dt.total_days().alias("_gap_days")
            )
            .get_column("_gap_days")
            .to_numpy()
        )

        # Extraer metadata
        splits = ticker_df.get_column("split").to_list()
        sample_eligible = ticker_df.get_column("sample_eligible").to_list()
        dates_list = dates.to_list()
        trade_entry_dates = ticker_df.get_column("trade_entry_date").to_list()
        trade_exit_dates = ticker_df.get_column("trade_exit_date").to_list()
        target_originals = (
            ticker_df.get_column(target_original_column).to_numpy().astype(np.float64)
        )

        # Marca de retorno anomalo por fila (solo si el filtro esta activo).
        if filter_anomalous:
            anomaly_flags = (
                ticker_df.get_column(anomaly_column)
                .fill_null(False)
                .to_numpy()
                .astype(bool)
            )
        else:
            anomaly_flags = None

        # Generar ventanas
        for end_pos in range(sequence_length - 1, ticker_df.height):
            start_pos = end_pos - sequence_length + 1

            # Verificar que la fila final es elegible
            if not sample_eligible[end_pos]:
                continue

            # Cuarentena de ventanas anomalas: en train/validation, descartar la
            # ventana si alguna de sus filas (features o etiqueta) esta marcada
            # como retorno anomalo. Las ventanas de test NO se filtran, para poder
            # analizar sus retornos extremos en el backtesting.
            if filter_anomalous and splits[end_pos] in ("train", "validation"):
                if anomaly_flags[start_pos: end_pos + 1].any():
                    continue

            # Verificar que el target escalado es finito
            if not np.isfinite(targets[end_pos]):
                continue

            # Verificar que no hay gaps excesivos dentro de la ventana
            window_gaps = calendar_gaps[start_pos + 1: end_pos + 1]
            has_excessive_gap = False
            for gap in window_gaps:
                if gap is not None and not np.isnan(gap) and gap > max_calendar_gap_days:
                    has_excessive_gap = True
                    break

            if has_excessive_gap:
                continue

            window_records.append({
                "sample_id": sample_id_counter,
                "ticker": ticker,
                "start_position": start_pos,
                "end_position": end_pos,
                "signal_date": dates_list[end_pos],
                "trade_entry_date": trade_entry_dates[end_pos],
                "trade_exit_date": trade_exit_dates[end_pos],
                "split": splits[end_pos],
                "target_log_return": float(target_originals[end_pos]),
            })
            sample_id_counter += 1

    window_index = pl.DataFrame(window_records)

    return {
        "feature_arrays": feature_arrays,
        "target_arrays": target_arrays,
        "window_index": window_index,
        "sequence_length": sequence_length,
        "feature_columns": list(feature_columns),
        "target_scaled_column": target_scaled_column,
        "target_original_column": target_original_column,
        "max_calendar_gap_days": max_calendar_gap_days,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader rapido: ventanas pre-materializadas en un unico tensor contiguo
# ─────────────────────────────────────────────────────────────────────────────


def _materialize_split_tensors(
    window_store: dict[str, Any],
    split: str,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Apila TODAS las ventanas de un split en un unico tensor contiguo.

    En lugar de cortar la ventana por muestra en __getitem__ (lento, en Python
    y en un solo hilo), se construye de una vez:
      - X: tensor [N, sequence_length, num_features] (float32)
      - y: tensor [N] (float32)
      - sample_ids: np.ndarray [N]

    Esto elimina el cuello de botella de carga de datos: el bucle de
    entrenamiento solo tiene que indexar y transferir lotes grandes a GPU.
    """
    seq_len = int(window_store["sequence_length"])
    n_features = len(window_store["feature_columns"])
    feature_arrays = window_store["feature_arrays"]
    target_arrays = window_store["target_arrays"]

    idx = (
        window_store["window_index"]
        .filter(pl.col("split") == split)
        .sort(["signal_date", "ticker"])
    )
    if idx.height == 0:
        raise ValueError(f"No hay ventanas para el split '{split}'.")

    tickers = idx.get_column("ticker").to_list()
    starts = idx.get_column("start_position").to_numpy().astype(np.int64)
    ends = idx.get_column("end_position").to_numpy().astype(np.int64)
    sample_ids = idx.get_column("sample_id").to_numpy().astype(np.int64)

    n = len(sample_ids)
    x = np.empty((n, seq_len, n_features), dtype=np.float32)
    y = np.empty((n,), dtype=np.float32)

    # Copia vectorizada por muestra (numpy, rapido; sin overhead de PyTorch/Python
    # por elemento). Se recorre una sola vez al inicio de cada entrenamiento.
    for i in range(n):
        tk = tickers[i]
        s = int(starts[i])
        e = int(ends[i])
        x[i] = feature_arrays[tk][s: e + 1]
        y[i] = target_arrays[tk][e]

    return torch.from_numpy(x), torch.from_numpy(y), sample_ids


class GpuBatchLoader:
    """Iterador de lotes sobre ventanas pre-materializadas (sin DataLoader/workers).

    Mantiene todo el split en un unico tensor contiguo en `storage_device`
    (por defecto CPU con pin_memory; para datasets que quepan en GPU se puede
    usar 'cuda' y evitar transferencias). En cada iteracion baraja indices
    (si shuffle) y entrega lotes ya en `compute_device`.

    Es compatible con el bucle de entrenamiento: cada iteracion produce
    (batch_x, batch_y, batch_sample_ids).
    """

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sample_ids: np.ndarray,
        batch_size: int,
        shuffle: bool,
        compute_device: torch.device,
        storage_device: torch.device | None = None,
        seed: int = 42,
    ) -> None:
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.compute_device = compute_device
        self.seed = int(seed)
        self._epoch = 0

        storage_device = storage_device or torch.device("cpu")
        self.x = x.to(storage_device)
        self.y = y.to(storage_device)
        # pin_memory acelera la transferencia CPU->GPU (solo si almacenado en CPU)
        if storage_device.type == "cpu" and compute_device.type == "cuda":
            self.x = self.x.pin_memory()
            self.y = self.y.pin_memory()
        self.sample_ids = torch.from_numpy(sample_ids)
        self.n = self.x.shape[0]

    def __len__(self) -> int:
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self._epoch)
            order = torch.randperm(self.n, generator=g)
            self._epoch += 1
        else:
            order = torch.arange(self.n)

        non_blocking = self.compute_device.type == "cuda"
        for start in range(0, self.n, self.batch_size):
            sel = order[start: start + self.batch_size]
            bx = self.x[sel].to(self.compute_device, non_blocking=non_blocking)
            by = self.y[sel].to(self.compute_device, non_blocking=non_blocking)
            bids = self.sample_ids[sel]
            yield bx, by, bids


def create_fast_dataloaders_from_arrays(
    window_arrays: dict[str, Any],
    batch_size: int = 512,
    random_seed: int = 42,
    compute_device: torch.device | None = None,
    store_on_gpu: bool = False,
) -> tuple[GpuBatchLoader | None, GpuBatchLoader | None, GpuBatchLoader | None]:
    """Crea loaders rapidos consumiendo DIRECTAMENTE las ventanas persistidas (.npz).

    Usa el tensor de entrada DIRECTA ya materializado y persistido por
    ``materialize_persistable_windows``
    (dataset ``{market,enriched}_window`` del catalogo). Asi el modelo entrena con
    EXACTAMENTE las mismas ventanas que se calcularon y guardaron de antemano, sin
    recalcularlas en cada entrenamiento.

    Args:
        window_arrays: Dict cargado del ``.npz`` con claves ``{split}_X``,
            ``{split}_y`` y ``{split}_sample_ids`` por split disponible, mas
            ``feature_columns``, ``sequence_length`` y ``n_features``.
        batch_size: Tamano de lote.
        random_seed: Semilla para el barajado del train.
        compute_device: Dispositivo de computo (por defecto cuda si disponible).
        store_on_gpu: Si True y hay GPU, mantiene los tensores en la GPU.

    Returns:
        (train_loader, validation_loader, test_loader); cada uno None si el split
        no esta presente en el artefacto.
    """
    if compute_device is None:
        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    storage_device = (
        compute_device if (store_on_gpu and compute_device.type == "cuda")
        else torch.device("cpu")
    )

    def _build(split: str, shuffle: bool) -> GpuBatchLoader | None:
        x_key = f"{split}_X"
        if x_key not in window_arrays:
            return None
        x = torch.from_numpy(np.asarray(window_arrays[x_key], dtype=np.float32))
        y = torch.from_numpy(np.asarray(window_arrays[f"{split}_y"], dtype=np.float32))
        ids = np.asarray(window_arrays[f"{split}_sample_ids"], dtype=np.int64)
        return GpuBatchLoader(
            x=x, y=y, sample_ids=ids, batch_size=batch_size, shuffle=shuffle,
            compute_device=compute_device, storage_device=storage_device,
            seed=random_seed,
        )

    train_loader = _build("train", shuffle=True)
    validation_loader = _build("validation", shuffle=False)
    test_loader = _build("test", shuffle=False)
    return train_loader, validation_loader, test_loader


def materialize_persistable_windows(
    window_store: dict[str, Any],
    splits: tuple[str, ...] = ("train", "validation", "test"),
) -> dict[str, Any]:
    """Materializa los tensores densos por split para persistir en disco.

    Produce el dataset de entrada DIRECTA al modelo: para cada split disponible,
    apila las ventanas en un unico tensor ``X`` de forma
    ``[N, sequence_length, n_features]`` (float32), junto con el target ``y``
    ([N]) y los ``sample_ids`` ([N]). Se acompana del orden EXACTO de las
    columnas de features y de metadata minima, de modo que el artefacto sea
    autosuficiente y reproducible (el tensor que ve PyTorch, congelado).

    Los splits vacios se omiten sin error (p. ej. si un split no tiene ventanas).

    Returns:
        Dict serializable con, por cada split presente, las claves
        ``{split}_X`` (np.float32), ``{split}_y`` (np.float32) y
        ``{split}_sample_ids`` (np.int64), mas ``feature_columns`` (lista),
        ``sequence_length`` (int) y ``n_features`` (int).
    """
    feature_columns = list(window_store["feature_columns"])
    payload: dict[str, Any] = {
        "feature_columns": feature_columns,
        "sequence_length": int(window_store["sequence_length"]),
        "n_features": len(feature_columns),
    }

    window_index = window_store["window_index"]
    available = set(window_index.get_column("split").unique().to_list())

    for split in splits:
        if split not in available:
            continue
        x, y, sample_ids = _materialize_split_tensors(window_store, split)
        payload[f"{split}_X"] = x.numpy().astype(np.float32)
        payload[f"{split}_y"] = y.numpy().astype(np.float32)
        payload[f"{split}_sample_ids"] = np.asarray(sample_ids, dtype=np.int64)

    return payload
