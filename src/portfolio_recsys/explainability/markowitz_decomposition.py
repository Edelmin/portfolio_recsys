"""Descomposicion de la optimizacion de Markowitz.

La asignacion de pesos de Markowitz es transparente por construccion: dado el
vector de retornos esperados, la matriz de covarianza y las restricciones, la
solucion se explica por el compromiso entre retorno esperado y riesgo de cada
activo. Este modulo descompone, para un rebalanceo concreto, la contribucion de
cada activo a ambos terminos, tomando los pesos y retornos que el propio
backtesting registro (SOLO LECTURA de artefactos).

Para el termino de riesgo se reestima la matriz de covarianza en la fecha del
rebalanceo reutilizando la misma logica que el backtesting (ventana historica de
retornos diarios anualizada), de modo que la descomposicion sea coherente con la
optimizacion original.

Definiciones (todo en escala anualizada, coherente con la optimizacion):
  - Contribucion al retorno esperado:  RC_i = w_i * mu_i.
  - Contribucion al riesgo (varianza): VC_i = w_i * (Sigma w)_i, de modo que
    sum_i VC_i = w^T Sigma w (varianza total de la cartera).
  - Contribucion marginal al riesgo:   MRC_i = (Sigma w)_i.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import polars as pl

from portfolio_recsys.explainability.traceability import _find_rebalance_record


def _to_date(value: Any) -> date:
    """Convierte una fecha del JSON (string ISO) a ``datetime.date``."""
    if isinstance(value, date):
        return value
    text = str(value)
    # Formatos habituales: '2022-07-05 04:00:00+00:00' o '2022-07-05'.
    text = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return datetime.fromisoformat(text[:10]).date()


def decompose_rebalance(
    results: dict[str, Any],
    config_name: str,
    rebalance_date: str,
    prices_eur: pl.DataFrame,
    lookback_sessions: int = 252,
    trading_sessions_per_year: int = 252,
) -> pl.DataFrame:
    """Descompone la asignacion de Markowitz de un rebalanceo por activo.

    Args:
        results: Dict de resultados de backtesting.
        config_name: Configuracion a inspeccionar (p. ej. ``top10_long_only``).
        rebalance_date: Fecha del rebalanceo.
        prices_eur: DataFrame de precios en EUR (columnas: ``ticker``, ``date``,
            ``close_eur``), para reestimar la covarianza en la fecha.
        lookback_sessions: Ventana historica para la covarianza (252 por defecto,
            igual que el backtesting).
        trading_sessions_per_year: Sesiones de negociacion por ano (252).

    Returns:
        DataFrame por activo con:
          - ``ticker``, ``weight``, ``weight_equal`` (1/K de referencia),
          - ``expected_return`` (mu_i predicho, anualizado),
          - ``return_contribution`` (w_i * mu_i),
          - ``marginal_risk`` ((Sigma w)_i),
          - ``variance_contribution`` (w_i * (Sigma w)_i),
          - ``risk_contribution_pct`` (VC_i / suma, en %).
    """
    record = _find_rebalance_record(results["configs"][config_name], rebalance_date)
    tickers = list(record.get("tickers", []))
    weights = np.asarray(record.get("weights", []), dtype=np.float64)
    mus = np.asarray(record.get("predicted_returns", []), dtype=np.float64)

    if len(tickers) == 0:
        return pl.DataFrame()

    as_of = _to_date(rebalance_date)
    cov = _estimate_covariance_matrix(
        prices_eur, tickers, as_of, lookback_sessions, trading_sessions_per_year
    )

    # (Sigma w) y contribuciones al riesgo.
    sigma_w = cov @ weights if cov is not None else np.full(len(tickers), np.nan)
    variance_contribution = weights * sigma_w
    return_contribution = weights * mus

    k = len(tickers)
    weight_equal = np.full(k, 1.0 / k)

    total_vc = float(np.nansum(variance_contribution))
    risk_pct = (
        variance_contribution / total_vc * 100.0
        if total_vc not in (0.0, np.nan) and np.isfinite(total_vc) and total_vc != 0
        else np.full(k, np.nan)
    )

    frame = pl.DataFrame({
        "ticker": tickers,
        "weight": weights,
        "weight_equal": weight_equal,
        "expected_return": mus,
        "return_contribution": return_contribution,
        "marginal_risk": sigma_w,
        "variance_contribution": variance_contribution,
        "risk_contribution_pct": risk_pct,
    }).sort("weight", descending=True)

    return frame


def _estimate_covariance_matrix(
    prices_eur: pl.DataFrame,
    tickers: list[str],
    as_of_date: date,
    lookback_sessions: int,
    trading_sessions_per_year: int,
) -> np.ndarray | None:
    """Estima la matriz de covarianza anualizada de los log-retornos diarios.

    Replica la logica del backtesting: toma los ultimos ``lookback_sessions``
    log-retornos diarios anteriores o iguales a la fecha, calcula la covarianza
    muestral diaria y la anualiza multiplicando por las sesiones por ano. Aplica
    una regularizacion minima para garantizar definicion positiva.
    """
    # Normalizar la columna de fecha a tipo Date para comparar con as_of_date
    # (la fuente suele venir como Datetime con zona horaria).
    prices_eur = prices_eur.with_columns(pl.col("date").cast(pl.Date).alias("_date_only"))

    return_series: dict[str, np.ndarray] = {}
    for ticker in tickers:
        hist = (
            prices_eur.filter(
                (pl.col("ticker") == ticker) & (pl.col("_date_only") <= as_of_date)
            )
            .sort("_date_only")
            .tail(lookback_sessions + 1)
        )
        prices = hist.get_column("close_eur").to_numpy().astype(np.float64)
        prices = prices[prices > 0]
        if prices.shape[0] < 2:
            return None
        return_series[ticker] = np.diff(np.log(prices))

    min_len = min(len(s) for s in return_series.values())
    if min_len < 2:
        return None
    matrix = np.vstack([return_series[t][-min_len:] for t in tickers])
    daily_cov = np.cov(matrix)
    if daily_cov.ndim == 0:  # un solo activo
        daily_cov = daily_cov.reshape(1, 1)
    annual_cov = daily_cov * trading_sessions_per_year
    annual_cov = annual_cov + 1e-8 * np.eye(annual_cov.shape[0])
    return annual_cov


def build_full_decomposition(
    results: dict[str, Any],
    prices_eur: pl.DataFrame,
    lookback_sessions: int = 252,
    trading_sessions_per_year: int = 252,
) -> pl.DataFrame:
    """Descomposicion exhaustiva de una combinacion: todas las configs del modelo
    y todos sus rebalanceos, apilados en un unico DataFrame.

    Reutiliza una cache de matrices de covarianza indexada por (fecha, conjunto
    de tickers), de modo que configuraciones distintas que comparten la misma
    seleccion en la misma fecha no recalculan la covarianza.

    Columns: config, rebalance_date, ticker, weight, weight_equal,
    expected_return, return_contribution, marginal_risk, variance_contribution,
    risk_contribution_pct.
    """
    from portfolio_recsys.explainability.traceability import model_config_names

    # Pre-normalizar fechas de precios una sola vez (coste evitado por ticker).
    prices_eur = prices_eur.with_columns(pl.col("date").cast(pl.Date).alias("_date_only"))

    cov_cache: dict[tuple, np.ndarray | None] = {}
    frames: list[pl.DataFrame] = []

    for config_name in model_config_names(results):
        cfg = results["configs"][config_name]
        for record in cfg.get("rebalance_details", []):
            tickers = list(record.get("tickers", []))
            if not tickers:
                continue
            weights = np.asarray(record.get("weights", []), dtype=np.float64)
            mus = np.asarray(record.get("predicted_returns", []), dtype=np.float64)
            as_of = _to_date(record.get("date"))

            cache_key = (as_of, tuple(tickers))
            if cache_key not in cov_cache:
                cov_cache[cache_key] = _estimate_covariance_matrix_prenorm(
                    prices_eur, tickers, as_of, lookback_sessions,
                    trading_sessions_per_year,
                )
            cov = cov_cache[cache_key]

            sigma_w = cov @ weights if cov is not None else np.full(len(tickers), np.nan)
            variance_contribution = weights * sigma_w
            return_contribution = weights * mus
            k = len(tickers)
            total_vc = float(np.nansum(variance_contribution))
            risk_pct = (
                variance_contribution / total_vc * 100.0
                if np.isfinite(total_vc) and total_vc != 0
                else np.full(k, np.nan)
            )

            frames.append(pl.DataFrame({
                "config": [config_name] * k,
                "rebalance_date": [str(record.get("date"))] * k,
                "ticker": tickers,
                "weight": weights,
                "weight_equal": np.full(k, 1.0 / k),
                "expected_return": mus,
                "return_contribution": return_contribution,
                "marginal_risk": sigma_w,
                "variance_contribution": variance_contribution,
                "risk_contribution_pct": risk_pct,
            }))

    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")


def _estimate_covariance_matrix_prenorm(
    prices_eur: pl.DataFrame,
    tickers: list[str],
    as_of_date: date,
    lookback_sessions: int,
    trading_sessions_per_year: int,
) -> np.ndarray | None:
    """Como ``_estimate_covariance_matrix`` pero asume que ``prices_eur`` ya trae
    la columna auxiliar ``_date_only`` (evita recastear en cada llamada)."""
    return_series: dict[str, np.ndarray] = {}
    for ticker in tickers:
        hist = (
            prices_eur.filter(
                (pl.col("ticker") == ticker) & (pl.col("_date_only") <= as_of_date)
            )
            .sort("_date_only")
            .tail(lookback_sessions + 1)
        )
        prices = hist.get_column("close_eur").to_numpy().astype(np.float64)
        prices = prices[prices > 0]
        if prices.shape[0] < 2:
            return None
        return_series[ticker] = np.diff(np.log(prices))

    min_len = min(len(s) for s in return_series.values())
    if min_len < 2:
        return None
    matrix = np.vstack([return_series[t][-min_len:] for t in tickers])
    daily_cov = np.cov(matrix)
    if daily_cov.ndim == 0:
        daily_cov = daily_cov.reshape(1, 1)
    annual_cov = daily_cov * trading_sessions_per_year + 1e-8 * np.eye(len(tickers))
    return annual_cov
