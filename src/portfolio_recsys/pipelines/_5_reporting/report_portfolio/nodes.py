"""Nodos del pipeline report_portfolio.

Construccion y evaluacion de carteras Markowitz con backtesting.

Flujo:
1. Generar fechas de rebalanceo segun el horizonte de prediccion.
2. En cada fecha, seleccionar top-K tickers por predicted_log_return.
3. Estimar la matriz de covarianza con retornos historicos.
4. Optimizar pesos (long-only y long-short) via cvxpy.
5. Simular rendimiento real hasta el siguiente rebalanceo.
6. Comparar vs oracle (tickers que realmente rindieron mejor).
7. Generar informe con equity curves y metricas.
"""

from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import date
from typing import Any

import numpy as np
import polars as pl

from portfolio_recsys.pipelines._1_ingest.fetch_benchmark_prices.nodes import (
    RISK_FREE_TICKER,
)

logger = logging.getLogger(__name__)

# Estado compartido con los procesos worker del backtest (se inicializa via
# initializer del ProcessPoolExecutor para NO picklear los DataFrames por tarea).
_WORKER_STATE: dict[str, Any] = {}


def _resolve_num_workers(num_workers, reserved_cores: int) -> int:
    """Resuelve el nº de procesos: explicito o (nucleos_logicos - reserva)."""
    if num_workers is not None and int(num_workers) > 0:
        return int(num_workers)
    total = os.cpu_count() or 1
    return max(1, total - max(0, int(reserved_cores)))


def _backtest_pool_initializer(val_predictions_ipc, prices_ipc) -> None:
    """Inicializa cada proceso worker: reconstruye los DataFrames polars una vez.

    Se pasan los datos serializados en formato Arrow IPC (bytes) una sola vez por
    worker (no por tarea), evitando el coste de picklear DataFrames grandes en
    cada configuracion. Limita ademas los hilos BLAS a 1 por proceso para evitar
    sobre-suscripcion (N procesos x M hilos).
    """
    import io
    # Limitar hilos de BLAS y de polars a 1 por proceso worker: con N procesos,
    # dejar que cada uno use multiples hilos sobre-suscribiria la CPU.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("POLARS_MAX_THREADS", "1")
    _WORKER_STATE["val_predictions"] = pl.read_ipc(io.BytesIO(val_predictions_ipc))
    _WORKER_STATE["prices_df"] = pl.read_ipc(io.BytesIO(prices_ipc))


def _backtest_config_task(task: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Ejecuta una configuracion de backtest en un proceso worker.

    Usa los DataFrames compartidos por el initializer. Devuelve (config_key, res).
    """
    config_key = task.pop("config_key")
    result = _backtest_single_config(
        val_predictions=_WORKER_STATE["val_predictions"],
        prices_df=_WORKER_STATE["prices_df"],
        **task,
    )
    return config_key, result


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Optimizacion Markowitz
# ─────────────────────────────────────────────────────────────────────────────


def _solve_markowitz_long_only(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    max_risk: float,
    max_weight: float | None = None,
) -> np.ndarray | None:
    """Optimiza cartera long-only (w >= 0, sum(w) = 1, risk <= max_risk).

    Args:
        expected_returns: Vector de log-retornos esperados EN ESCALA DE PERIODO
            de holding (K,).
        cov_matrix: Matriz de covarianza de log-retornos EN ESCALA DE PERIODO
            de holding (K, K).
        max_risk: Volatilidad maxima de la cartera EN ESCALA DE PERIODO de
            holding (desviacion tipica).
        max_weight: Peso maximo por activo (limite de concentracion). Si es None
            no se aplica limite. Markowitz por si solo NO diversifica: al
            maximizar retorno con un tope de riesgo, la solucion optima puede
            concentrar todo el peso en un unico activo. Este limite fuerza el
            reparto.

    Returns:
        Vector de pesos optimos o None si no hay solucion factible.
    """
    import cvxpy as cp

    n = len(expected_returns)
    w = cp.Variable(n)

    objective = cp.Maximize(expected_returns @ w)
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        cp.quad_form(w, cov_matrix) <= max_risk**2,
    ]
    if max_weight is not None:
        constraints.append(w <= max_weight)

    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.OSQP, warm_start=True)
        if problem.status in ("optimal", "optimal_inaccurate"):
            return np.array(w.value).flatten()
    except cp.SolverError:
        pass

    # Fallback: intentar con SCS
    try:
        problem.solve(solver=cp.SCS)
        if problem.status in ("optimal", "optimal_inaccurate"):
            return np.array(w.value).flatten()
    except cp.SolverError:
        pass

    return None


def _solve_markowitz_long_short(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    max_risk: float,
    max_leverage: float,
    max_weight: float | None = None,
) -> np.ndarray | None:
    """Optimiza cartera long-short (sum(w)=1, sum(|w|)<=leverage, risk<=max_risk).

    Args:
        expected_returns: Vector de log-retornos esperados EN ESCALA DE PERIODO
            de holding (K,).
        cov_matrix: Matriz de covarianza de log-retornos EN ESCALA DE PERIODO
            de holding (K, K).
        max_risk: Volatilidad maxima de la cartera EN ESCALA DE PERIODO de
            holding (desviacion tipica).
        max_leverage: Limite de leverage (suma de valores absolutos).
        max_weight: Limite de concentracion por activo aplicado en valor
            absoluto (|w_i| <= max_weight). Si es None no se aplica.

    Returns:
        Vector de pesos optimos o None si no hay solucion factible.
    """
    import cvxpy as cp

    n = len(expected_returns)
    w = cp.Variable(n)

    objective = cp.Maximize(expected_returns @ w)
    constraints = [
        cp.sum(w) == 1,
        cp.norm(w, 1) <= max_leverage,
        cp.quad_form(w, cov_matrix) <= max_risk**2,
    ]
    if max_weight is not None:
        constraints.append(cp.abs(w) <= max_weight)

    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.OSQP, warm_start=True)
        if problem.status in ("optimal", "optimal_inaccurate"):
            return np.array(w.value).flatten()
    except cp.SolverError:
        pass

    try:
        problem.solve(solver=cp.SCS)
        if problem.status in ("optimal", "optimal_inaccurate"):
            return np.array(w.value).flatten()
    except cp.SolverError:
        pass

    return None


def _solve_cvxpy(problem, w):
    """Resuelve un problema cvxpy con OSQP y fallback a SCS. Devuelve pesos o None."""
    import cvxpy as cp

    for solver in (cp.OSQP, cp.SCS):
        try:
            kwargs = {"warm_start": True} if solver is cp.OSQP else {}
            problem.solve(solver=solver, **kwargs)
            if problem.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                return np.array(w.value).flatten()
        except cp.SolverError:
            continue
    return None


def _base_constraints(w, max_leverage, max_weight):
    """Restricciones comunes: presupuesto pleno + long-only o long-short.

    Si max_leverage es None -> long-only (w>=0). Si no -> long-short con
    ||w||_1 <= max_leverage. max_weight limita la concentracion por activo.
    """
    import cvxpy as cp

    constraints = [cp.sum(w) == 1]
    if max_leverage is None:
        constraints.append(w >= 0)
        if max_weight is not None:
            constraints.append(w <= max_weight)
    else:
        constraints.append(cp.norm(w, 1) <= max_leverage)
        if max_weight is not None:
            constraints.append(cp.abs(w) <= max_weight)
    return constraints


def _solve_min_variance(
    cov_matrix: np.ndarray,
    max_weight: float | None = None,
    max_leverage: float | None = None,
) -> np.ndarray | None:
    """Cartera de MINIMA VARIANZA: min wᵀΣw s.a. sum(w)=1 (+ concentracion).

    NO usa los retornos esperados (mu) en la ponderacion: reparte el peso para
    minimizar el riesgo conjunto, aprovechando activos poco correlacionados. Por
    ello diversifica de forma natural. Tampoco usa el tope max_risk: busca el
    minimo riesgo alcanzable (lo que evita la infactibilidad a horizontes largos).

    Args:
        cov_matrix: Covarianza EN ESCALA DE PERIODO de holding (K, K).
        max_weight: Limite de concentracion por activo.
        max_leverage: None -> long-only; valor -> long-short.
    """
    import cvxpy as cp

    n = cov_matrix.shape[0]
    w = cp.Variable(n)
    problem = cp.Problem(cp.Minimize(cp.quad_form(w, cov_matrix)),
                         _base_constraints(w, max_leverage, max_weight))
    return _solve_cvxpy(problem, w)


def _solve_mean_variance(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float,
    max_weight: float | None = None,
    max_leverage: float | None = None,
) -> np.ndarray | None:
    """Cartera MEDIA-VARIANZA: max muᵀw - lambda·wᵀΣw s.a. sum(w)=1.

    Formulacion canonica de Markowitz con aversion al riesgo `lambda`
    (risk_aversion). Usa mu tanto en seleccion (top-K, previa) como en la
    ponderacion, pero penaliza el riesgo en el objetivo, por lo que diversifica
    mas que el enfoque de maximizar retorno con tope de riesgo. `mu` y `Sigma`
    deben estar en la MISMA escala temporal (aqui, escala de periodo de holding),
    de modo que un mismo `lambda` es comparable entre horizontes.

    Args:
        expected_returns: mu EN ESCALA DE PERIODO (K,).
        cov_matrix: Covarianza EN ESCALA DE PERIODO (K, K).
        risk_aversion: Coeficiente lambda (>0). Mayor -> mas aversion al riesgo
            y mas diversificacion.
        max_weight: Limite de concentracion por activo.
        max_leverage: None -> long-only; valor -> long-short.
    """
    import cvxpy as cp

    n = len(expected_returns)
    w = cp.Variable(n)
    objective = cp.Maximize(expected_returns @ w - risk_aversion * cp.quad_form(w, cov_matrix))
    problem = cp.Problem(objective, _base_constraints(w, max_leverage, max_weight))
    return _solve_cvxpy(problem, w)


def _solve_target_return(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    target_return: float,
    max_weight: float | None = None,
    max_leverage: float | None = None,
) -> np.ndarray | None:
    """Cartera de MINIMA VARIANZA con retorno objetivo (dual del enfoque actual):

        min wᵀΣw  s.a.  muᵀw >= target_return, sum(w)=1 (+ concentracion).

    Fija un retorno esperado minimo y minimiza el riesgo para alcanzarlo. Si es
    infactible (ningun reparto alcanza el objetivo), el llamante debe recurrir a
    minima varianza como fallback.

    Args:
        expected_returns: mu EN ESCALA DE PERIODO (K,).
        cov_matrix: Covarianza EN ESCALA DE PERIODO (K, K).
        target_return: Retorno esperado minimo EN ESCALA DE PERIODO.
        max_weight: Limite de concentracion por activo.
        max_leverage: None -> long-only; valor -> long-short.
    """
    import cvxpy as cp

    n = len(expected_returns)
    w = cp.Variable(n)
    constraints = _base_constraints(w, max_leverage, max_weight)
    constraints.append(expected_returns @ w >= target_return)
    problem = cp.Problem(cp.Minimize(cp.quad_form(w, cov_matrix)), constraints)
    return _solve_cvxpy(problem, w)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Metricas de cartera
# ─────────────────────────────────────────────────────────────────────────────


def _compute_portfolio_metrics(
    cumulative_returns: list[float],
    period_returns: list[float],
    holding_sessions: int,
) -> dict[str, float | None]:
    """Calcula metricas de rendimiento de una cartera.

    Args:
        cumulative_returns: Lista de valores de la equity curve (1.0 = inicio).
        period_returns: Lista de retornos simples por periodo de rebalanceo.
        holding_sessions: Numero de sesiones por periodo de holding.

    Returns:
        Diccionario con metricas: total_return, annualized_return,
        annualized_volatility, sharpe_ratio, sortino_ratio, max_drawdown.
    """
    if not cumulative_returns or len(cumulative_returns) < 2:
        return {
            "total_return": None,
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe_ratio": None,
            "sortino_ratio": None,
            "max_drawdown": None,
        }

    total_return = cumulative_returns[-1] / cumulative_returns[0] - 1.0

    # Anualizar
    periods_per_year = 252.0 / holding_sessions
    n_periods = len(period_returns)

    if n_periods > 0 and total_return > -1.0:
        annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    else:
        annualized_return = None

    if n_periods > 1:
        vol = float(np.std(period_returns, ddof=1))
        annualized_vol = vol * math.sqrt(periods_per_year)
    else:
        annualized_vol = None

    if annualized_return is not None and annualized_vol and annualized_vol > 0:
        sharpe = annualized_return / annualized_vol
    else:
        sharpe = None

    # Ratio de Sortino: penaliza solo la volatilidad a la baja
    if n_periods > 1 and annualized_return is not None:
        downside_returns = [r for r in period_returns if r < 0]
        if downside_returns:
            downside_std = float(np.std(downside_returns, ddof=1))
            annualized_downside = downside_std * math.sqrt(periods_per_year)
            if annualized_downside > 0:
                sortino = annualized_return / annualized_downside
            else:
                sortino = None
        else:
            # No hay retornos negativos: Sortino es infinito en teoria
            sortino = None
    else:
        sortino = None

    # Max drawdown
    peak = cumulative_returns[0]
    max_dd = 0.0
    for val in cumulative_returns:
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        if dd > max_dd:
            max_dd = dd

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Nodo principal: run_markowitz_backtest
# ─────────────────────────────────────────────────────────────────────────────


def run_markowitz_backtest(
    prediction_table: pl.DataFrame,
    stock_prices_eur: pl.DataFrame,
    benchmark_prices_eur: pl.DataFrame,
    split_configuration: dict[str, Any],
    markowitz_params: dict[str, Any],
    horizon_key: str,
    horizon_config: dict[str, Any],
    evaluation_split: str = "test",
) -> dict[str, Any]:
    """Ejecuta el backtest completo de carteras Markowitz para un horizonte.

    Para cada configuracion top-K y variante (long-only / long-short),
    genera rebalanceos sobre el periodo de validacion y calcula el
    rendimiento realizado.

    Args:
        prediction_table: DataFrame con predicciones del modelo
            (signal_date, ticker, predicted_log_return, target_log_return, split).
        stock_prices_eur: DataFrame con precios diarios en EUR
            (date, ticker, close_eur).
        benchmark_prices_eur: DataFrame con precios diarios en EUR del indice
            de mercado (S&P 500) y del activo seguro (date, ticker, close_eur).
            Se usan solo como comparadores, restringidos al periodo de validacion.
        split_configuration: Dict con train_end_date, validation_end_date, etc.
        markowitz_params: Parametros del pipeline markowitz_portfolio.
        horizon_key: Clave del horizonte (ej: "1d", "3m", "1y").
        horizon_config: Dict con holding_period_sessions y label.

    Returns:
        Diccionario con resultados del backtest para todas las configuraciones.
    """
    holding_sessions = horizon_config["holding_period_sessions"]
    horizon_label = horizon_config["label"]
    top_k_configs = markowitz_params["top_k_configs"]
    max_risk = markowitz_params["max_risk"]
    cov_lookback = markowitz_params["covariance_lookback_sessions"]
    allow_short = markowitz_params["allow_short_selling"]
    max_leverage = markowitz_params.get("max_leverage", 1.5)
    min_history = markowitz_params.get("min_price_history_sessions", 60)
    max_weight = markowitz_params.get("max_weight", None)
    # Estrategias de optimizacion a evaluar y sus parametros
    strategies = markowitz_params.get("strategies", ["max_return"])
    risk_aversion_lambdas = markowitz_params.get("risk_aversion_lambdas", [5.0])
    target_annual_return = markowitz_params.get("target_annual_return", 0.15)

    logger.info(
        "Backtest Markowitz [%s] — split=%s, holding=%d sesiones, top_k=%s, "
        "max_risk(anual)=%.0f%%, max_weight=%s, strategies=%s",
        horizon_key, evaluation_split, holding_sessions, top_k_configs,
        max_risk * 100, max_weight, strategies,
    )

    # Filtrar predicciones del split de evaluacion (por defecto TEST: periodo
    # out-of-sample en el que el modelo esta congelado y no intervino ni en el
    # ajuste de pesos ni en la seleccion de arquitectura/hiperparametros).
    val_predictions = prediction_table.filter(pl.col("split") == evaluation_split)

    # Exclusion manual de tickers del universo ponderable. La prediccion ya se
    # calculo (siguen en prediction_table para analisis aparte), pero se apartan
    # del backtest para que no entren en la seleccion top-K ni en la optimizacion
    # de Markowitz. Al filtrar aqui, la exclusion afecta por igual a todas las
    # estrategias (modelo, oracle y equiponderada), preservando un universo comun.
    excluded_tickers = markowitz_params.get("excluded_tickers", []) or []
    if excluded_tickers:
        n_before = val_predictions.height
        val_predictions = val_predictions.filter(
            ~pl.col("ticker").is_in(excluded_tickers)
        )
        logger.info(
            "Backtest: excluidos %d tickers del universo ponderable (%s); "
            "%d -> %d filas de prediccion.",
            len(excluded_tickers), ", ".join(excluded_tickers),
            n_before, val_predictions.height,
        )

    if val_predictions.is_empty():
        logger.warning(
            "No hay predicciones del split '%s' para horizonte %s",
            evaluation_split, horizon_key,
        )
        return {"horizon": horizon_key, "error": f"no_{evaluation_split}_predictions"}

    # Obtener fechas de señal unicas en validacion (ordenadas)
    signal_dates = (
        val_predictions.select("signal_date")
        .unique()
        .sort("signal_date")
        .get_column("signal_date")
        .to_list()
    )

    # Generar fechas de rebalanceo: cada holding_sessions fechas de señal
    rebalance_dates = signal_dates[::holding_sessions]
    logger.info(
        "Fechas de rebalanceo generadas: %d (de %d signal_dates)",
        len(rebalance_dates), len(signal_dates),
    )

    if len(rebalance_dates) < 2:
        logger.warning("Insuficientes fechas de rebalanceo para horizonte %s", horizon_key)
        return {"horizon": horizon_key, "error": "insufficient_rebalance_dates"}

    # Preparar precios como dict para busqueda rapida
    prices_df = stock_prices_eur.select(["date", "ticker", "close_eur"]).sort(["ticker", "date"])

    # Resultados por configuracion
    all_results = {}
    directions = ["long_only"]
    if allow_short:
        directions.append("long_short")

    # Prefijo de estrategia para las claves de configuracion. max_return mantiene
    # la clave historica (top{k}_{direction}) por compatibilidad con informes.
    def _strategy_variants(strategy: str):
        """Genera (config_suffix, kwargs) por cada variante de una estrategia."""
        if strategy == "max_return":
            for d in directions:
                yield d, {"strategy": "max_return", "direction": d}
        elif strategy == "min_variance":
            for d in directions:
                yield f"minvar_{d}", {"strategy": "min_variance", "direction": d}
        elif strategy == "target_return":
            for d in directions:
                yield f"target_{d}", {"strategy": "target_return", "direction": d,
                                      "target_annual_return": target_annual_return}
        elif strategy == "mean_variance":
            for lam in risk_aversion_lambdas:
                lam_tag = f"{lam:g}".replace(".", "p")
                for d in directions:
                    yield (f"meanvar_l{lam_tag}_{d}",
                           {"strategy": "mean_variance", "direction": d,
                            "risk_aversion": lam})
        else:
            raise ValueError(f"Estrategia desconocida: {strategy}")

    # Construir la lista de tareas de configuracion (cada una independiente).
    tasks: list[dict[str, Any]] = []
    for top_k in top_k_configs:
        for strategy in strategies:
            for suffix, kwargs in _strategy_variants(strategy):
                tasks.append({
                    "config_key": f"top{top_k}_{suffix}",
                    "rebalance_dates": rebalance_dates,
                    "top_k": top_k,
                    "variant": suffix,
                    "max_risk": max_risk,
                    "max_leverage": max_leverage,
                    "cov_lookback": cov_lookback,
                    "min_history": min_history,
                    "holding_sessions": holding_sessions,
                    "max_weight": max_weight,
                    **kwargs,
                })

    n_workers = _resolve_num_workers(
        markowitz_params.get("backtest_num_workers"),
        markowitz_params.get("backtest_reserved_cores", 4),
    )
    logger.info("[%s] %d configuraciones, %d procesos", horizon_key, len(tasks), n_workers)

    if n_workers <= 1 or len(tasks) <= 1:
        # Ejecucion secuencial (sin overhead de procesos)
        for task in tasks:
            key, result = _backtest_config_task({**task})
            all_results[key] = result
            logger.info("  [%s] %s: total_return=%.4f", horizon_key, key,
                        result["metrics"].get("total_return") or 0.0)
    else:
        # Paralelizacion por configuracion (CPU-bound, cvxpy). Los DataFrames se
        # comparten una sola vez por worker via Arrow IPC en el initializer.
        import io
        # Fijar hilos de polars a 1 en el entorno ANTES de spawn: en Windows los
        # procesos hijo re-importan el modulo (y polars) al arrancar, por lo que
        # POLARS_MAX_THREADS debe estar en el entorno heredado para tener efecto.
        prev_polars_threads = os.environ.get("POLARS_MAX_THREADS")
        os.environ["POLARS_MAX_THREADS"] = "1"
        val_buf = io.BytesIO(); val_predictions.write_ipc(val_buf)
        prices_buf = io.BytesIO(); prices_df.write_ipc(prices_buf)
        try:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_backtest_pool_initializer,
                initargs=(val_buf.getvalue(), prices_buf.getvalue()),
            ) as executor:
                for key, result in executor.map(_backtest_config_task, tasks):
                    all_results[key] = result
                    logger.info("  [%s] %s: total_return=%.4f", horizon_key, key,
                                result["metrics"].get("total_return") or 0.0)
        finally:
            # Restaurar el entorno del proceso padre
            if prev_polars_threads is None:
                os.environ.pop("POLARS_MAX_THREADS", None)
            else:
                os.environ["POLARS_MAX_THREADS"] = prev_polars_threads

    # Oracle y equiponderado
    for top_k in top_k_configs:
        oracle_result = _backtest_oracle(
            val_predictions=val_predictions,
            prices_df=prices_df,
            rebalance_dates=rebalance_dates,
            top_k=top_k,
            max_risk=max_risk,
            cov_lookback=cov_lookback,
            min_history=min_history,
            holding_sessions=holding_sessions,
            max_weight=max_weight,
        )
        all_results[f"top{top_k}_oracle"] = oracle_result

        equal_result = _backtest_equal_weight(
            val_predictions=val_predictions,
            prices_df=prices_df,
            rebalance_dates=rebalance_dates,
            top_k=top_k,
            holding_sessions=holding_sessions,
        )
        all_results[f"top{top_k}_equal_weight"] = equal_result

    # ─────────────────────────────────────────────────────────────────────
    # Comparadores de mercado: S&P 500 (buy & hold) y cartera {S&P, activo
    # seguro} optimizada por Markowitz. Se restringen al periodo de validacion
    # (mismas rebalance_dates que el resto de estrategias) y NO dependen del
    # modelo recurrente: consumen solo benchmark_prices_eur.
    # ─────────────────────────────────────────────────────────────────────
    benchmark_params = markowitz_params.get("benchmark", {})
    market_ticker = benchmark_params.get("market_ticker", "^GSPC")
    market_label = benchmark_params.get("market_label", "S&P 500")
    risk_free_label = benchmark_params.get(
        "risk_free_label", "Activo seguro (inflacion fija)"
    )

    if benchmark_prices_eur is not None and not benchmark_prices_eur.is_empty():
        bench_prices = (
            benchmark_prices_eur.select(["date", "ticker", "close_eur"])
            .sort(["ticker", "date"])
        )

        # Opcion 1: S&P 500 buy & hold sobre el periodo de validacion.
        all_results["benchmark_sp500"] = _backtest_buy_and_hold(
            prices_df=bench_prices,
            ticker=market_ticker,
            rebalance_dates=rebalance_dates,
            holding_sessions=holding_sessions,
            label=market_label,
        )

        # Opcion 2: cartera Markowitz de dos activos {S&P 500, activo seguro}.
        all_results["benchmark_sp500_riskfree"] = _backtest_market_riskfree(
            prices_df=bench_prices,
            market_ticker=market_ticker,
            risk_free_ticker=RISK_FREE_TICKER,
            rebalance_dates=rebalance_dates,
            max_risk=max_risk,
            cov_lookback=cov_lookback,
            min_history=min_history,
            holding_sessions=holding_sessions,
            labels={"market": market_label, "risk_free": risk_free_label},
        )
    else:
        logger.warning(
            "benchmark_prices_eur vacio: no se generan comparadores de mercado."
        )

    return {
        "horizon": horizon_key,
        "horizon_label": horizon_label,
        "evaluation_split": evaluation_split,
        "holding_sessions": holding_sessions,
        "rebalance_dates": [str(d) for d in rebalance_dates],
        "n_rebalances": len(rebalance_dates) - 1,
        "configs": all_results,
    }


def _backtest_single_config(
    val_predictions: pl.DataFrame,
    prices_df: pl.DataFrame,
    rebalance_dates: list[date],
    top_k: int,
    variant: str,
    max_risk: float,
    max_leverage: float,
    cov_lookback: int,
    min_history: int,
    holding_sessions: int,
    max_weight: float | None = None,
    strategy: str = "max_return",
    direction: str = "long_only",
    risk_aversion: float | None = None,
    target_annual_return: float | None = None,
) -> dict[str, Any]:
    """Backtest para una configuracion (top-K + estrategia de optimizacion).

    El riesgo se trabaja EN ESCALA DE PERIODO de holding: mu = log-retorno del
    periodo (sin anualizar), covarianza escalada al periodo, y max_risk (anual)
    convertido a periodo. Se aplica un limite de concentracion max_weight.

    Args:
        strategy: Formulacion de optimizacion:
            - "max_return": max muᵀw s.a. riesgo <= max_risk (enfoque original).
            - "min_variance": min wᵀΣw (no usa mu ni max_risk en la ponderacion).
            - "mean_variance": max muᵀw - lambda·wᵀΣw (usa risk_aversion).
            - "target_return": min wᵀΣw s.a. muᵀw >= objetivo (usa
              target_annual_return; fallback a min_variance si infactible).
        direction: "long_only" o "long_short".
        risk_aversion: lambda para mean_variance.
        target_annual_return: retorno objetivo ANUAL para target_return (se
            convierte a escala de periodo internamente).

    Returns:
        Dict con equity_curve, period_returns, rebalance_details, metrics.
    """
    max_risk_period = _annual_risk_to_period(max_risk, holding_sessions)
    # Leverage efectivo segun direccion (None => long-only en los nuevos solvers)
    leverage = max_leverage if direction == "long_short" else None
    # Convertir retorno objetivo anual a escala de periodo (media lineal en tiempo)
    target_return_period = (
        None if target_annual_return is None
        else target_annual_return * holding_sessions / TRADING_SESSIONS_PER_YEAR
    )
    equity = 1.0
    equity_curve = [equity]
    period_returns = []
    rebalance_details = []

    for i in range(len(rebalance_dates) - 1):
        reb_date = rebalance_dates[i]
        next_reb_date = rebalance_dates[i + 1]

        # Seleccionar top-K tickers por prediccion en esta fecha
        day_preds = val_predictions.filter(pl.col("signal_date") == reb_date)

        if day_preds.height < top_k:
            # No hay suficientes tickers, mantener cash
            period_returns.append(0.0)
            equity_curve.append(equity)
            rebalance_details.append({
                "date": str(reb_date),
                "status": "insufficient_tickers",
                "available": day_preds.height,
            })
            continue

        top_tickers = (
            day_preds.sort("predicted_log_return", descending=True)
            .head(top_k)
            .select(["ticker", "predicted_log_return"])
        )
        ticker_list = top_tickers.get_column("ticker").to_list()
        predicted_returns = top_tickers.get_column("predicted_log_return").to_numpy()

        # Retorno esperado EN ESCALA DE PERIODO de holding: predicted_log_return
        # ya es el log-retorno del periodo, no se anualiza.
        mu = predicted_returns.astype(np.float64)

        # Covarianza EN ESCALA DE PERIODO de holding (Sigma_diaria * holding).
        cov_matrix = _period_covariance(
            prices_df=prices_df,
            tickers=ticker_list,
            as_of_date=reb_date,
            lookback_sessions=cov_lookback,
            min_history=min_history,
            holding_sessions=holding_sessions,
        )

        if cov_matrix is None:
            # No se pudo estimar covarianza, usar equiponderado
            weights = np.ones(len(ticker_list)) / len(ticker_list)
            opt_status = "fallback_equal_weight"
        else:
            # Despachar segun la estrategia de optimizacion (escala de periodo)
            if strategy == "max_return":
                if direction == "long_only":
                    weights = _solve_markowitz_long_only(
                        mu, cov_matrix, max_risk_period, max_weight
                    )
                else:
                    weights = _solve_markowitz_long_short(
                        mu, cov_matrix, max_risk_period, max_leverage, max_weight
                    )
                opt_status = "optimal"
            elif strategy == "min_variance":
                weights = _solve_min_variance(cov_matrix, max_weight, leverage)
                opt_status = "optimal"
            elif strategy == "mean_variance":
                weights = _solve_mean_variance(
                    mu, cov_matrix, float(risk_aversion), max_weight, leverage
                )
                opt_status = "optimal"
            elif strategy == "target_return":
                weights = _solve_target_return(
                    mu, cov_matrix, float(target_return_period), max_weight, leverage
                )
                if weights is None:
                    # Objetivo infactible: fallback a minima varianza
                    weights = _solve_min_variance(cov_matrix, max_weight, leverage)
                    opt_status = "target_infeasible_minvar"
                else:
                    opt_status = "optimal"
            else:
                raise ValueError(f"Estrategia desconocida: {strategy}")

            if weights is None:
                weights = np.ones(len(ticker_list)) / len(ticker_list)
                opt_status = "infeasible_fallback"

        # Calcular retorno realizado del portfolio en el periodo
        period_return = _compute_period_return(
            prices_df=prices_df,
            tickers=ticker_list,
            weights=weights,
            start_date=reb_date,
            end_date=next_reb_date,
        )

        period_returns.append(period_return)
        equity *= (1 + period_return)
        equity_curve.append(equity)

        rebalance_details.append({
            "date": str(reb_date),
            "next_date": str(next_reb_date),
            "tickers": ticker_list,
            "weights": weights.tolist(),
            "predicted_returns": mu.tolist(),
            "period_return": period_return,
            "cumulative_return": equity - 1.0,
            "optimization_status": opt_status,
        })

    metrics = _compute_portfolio_metrics(equity_curve, period_returns, holding_sessions)

    return {
        "equity_curve": equity_curve,
        "period_returns": period_returns,
        "rebalance_details": rebalance_details,
        "metrics": metrics,
    }


def _backtest_buy_and_hold(
    prices_df: pl.DataFrame,
    ticker: str,
    rebalance_dates: list[date],
    holding_sessions: int,
    label: str,
) -> dict[str, Any]:
    """Backtest buy & hold de un unico activo (comparador de mercado).

    Mantiene el activo (ej: S&P 500) durante todo el periodo de validacion,
    encadenando el retorno realizado entre fechas de rebalanceo consecutivas.
    Usa la misma malla de rebalanceo que el resto de estrategias para que las
    metricas anualizadas sean comparables.

    Args:
        prices_df: Precios en EUR (date, ticker, close_eur).
        ticker: Ticker del activo a mantener.
        rebalance_dates: Fechas de rebalanceo (periodo de validacion).
        holding_sessions: Sesiones por periodo de holding.
        label: Etiqueta legible para el informe.

    Returns:
        Dict con equity_curve, period_returns, rebalance_details, metrics, label.
    """
    equity = 1.0
    equity_curve = [equity]
    period_returns = []
    rebalance_details = []
    weights = np.array([1.0])

    for i in range(len(rebalance_dates) - 1):
        reb_date = rebalance_dates[i]
        next_reb_date = rebalance_dates[i + 1]

        period_return = _compute_period_return(
            prices_df=prices_df,
            tickers=[ticker],
            weights=weights,
            start_date=reb_date,
            end_date=next_reb_date,
        )

        period_returns.append(period_return)
        equity *= (1 + period_return)
        equity_curve.append(equity)
        rebalance_details.append({
            "date": str(reb_date),
            "next_date": str(next_reb_date),
            "tickers": [ticker],
            "weights": [1.0],
            "period_return": period_return,
            "cumulative_return": equity - 1.0,
            "optimization_status": "buy_and_hold",
        })

    metrics = _compute_portfolio_metrics(equity_curve, period_returns, holding_sessions)
    return {
        "label": label,
        "equity_curve": equity_curve,
        "period_returns": period_returns,
        "rebalance_details": rebalance_details,
        "metrics": metrics,
    }


def _backtest_market_riskfree(
    prices_df: pl.DataFrame,
    market_ticker: str,
    risk_free_ticker: str,
    rebalance_dates: list[date],
    max_risk: float,
    cov_lookback: int,
    min_history: int,
    holding_sessions: int,
    labels: dict[str, str],
) -> dict[str, Any]:
    """Backtest Markowitz de dos activos: {indice de mercado, activo seguro}.

    En cada rebalanceo estima retorno esperado (media historica de log-retornos
    del lookback) y covarianza de ambos activos, y optimiza long-only con la
    misma restriccion de riesgo (max_risk) que el resto de carteras. El activo
    seguro actua como refugio (volatilidad ~0), por lo que Markowitz reparte
    peso segun el trade-off riesgo/retorno del periodo.

    Restringido al periodo de validacion (mismas rebalance_dates).

    Returns:
        Dict con equity_curve, period_returns, rebalance_details, metrics, label.
    """
    tickers = [market_ticker, risk_free_ticker]
    max_risk_period = _annual_risk_to_period(max_risk, holding_sessions)
    equity = 1.0
    equity_curve = [equity]
    period_returns = []
    rebalance_details = []

    for i in range(len(rebalance_dates) - 1):
        reb_date = rebalance_dates[i]
        next_reb_date = rebalance_dates[i + 1]

        # Retorno esperado por activo, EN ESCALA DE PERIODO de holding. El
        # helper devuelve la media diaria anualizada (x252); se pasa a periodo
        # multiplicando por (holding/252).
        mu_annual = _estimate_expected_returns(
            prices_df=prices_df,
            tickers=tickers,
            as_of_date=reb_date,
            lookback_sessions=cov_lookback,
        )
        mu = None if mu_annual is None else mu_annual * (holding_sessions / TRADING_SESSIONS_PER_YEAR)
        cov_matrix = _period_covariance(
            prices_df=prices_df,
            tickers=tickers,
            as_of_date=reb_date,
            lookback_sessions=cov_lookback,
            min_history=min_history,
            holding_sessions=holding_sessions,
        )

        if mu is None or cov_matrix is None or cov_matrix.shape[0] != len(tickers):
            # Sin datos suficientes: refugio total en el activo seguro.
            weights = np.array([0.0, 1.0])
            opt_status = "fallback_risk_free"
        else:
            weights = _solve_markowitz_long_only(mu, cov_matrix, max_risk_period)
            if weights is None:
                weights = np.array([0.0, 1.0])
                opt_status = "infeasible_fallback_risk_free"
            else:
                opt_status = "optimal"

        period_return = _compute_period_return(
            prices_df=prices_df,
            tickers=tickers,
            weights=weights,
            start_date=reb_date,
            end_date=next_reb_date,
        )

        period_returns.append(period_return)
        equity *= (1 + period_return)
        equity_curve.append(equity)
        rebalance_details.append({
            "date": str(reb_date),
            "next_date": str(next_reb_date),
            "tickers": tickers,
            "weights": weights.tolist(),
            "period_return": period_return,
            "cumulative_return": equity - 1.0,
            "optimization_status": opt_status,
        })

    metrics = _compute_portfolio_metrics(equity_curve, period_returns, holding_sessions)
    return {
        "label": f"{labels.get('market', market_ticker)} + {labels.get('risk_free', risk_free_ticker)}",
        "equity_curve": equity_curve,
        "period_returns": period_returns,
        "rebalance_details": rebalance_details,
        "metrics": metrics,
    }


def _estimate_expected_returns(
    prices_df: pl.DataFrame,
    tickers: list[str],
    as_of_date: date,
    lookback_sessions: int,
) -> np.ndarray | None:
    """Estima el log-retorno esperado anualizado por activo (media historica).

    Calcula la media de log-retornos diarios del lookback y la anualiza
    (media_diaria * 252) para dejarla en la misma escala que la covarianza
    anualizada. Devuelve un vector alineado con `tickers` o None si falta
    algun activo.
    """
    mus = []
    for ticker in tickers:
        hist = (
            prices_df.filter(
                (pl.col("ticker") == ticker) & (pl.col("date") <= as_of_date)
            )
            .sort("date")
            .tail(lookback_sessions + 1)
        )
        prices = hist.get_column("close_eur").to_numpy().astype(np.float64)
        prices = prices[prices > 0]
        if prices.shape[0] < 2:
            return None
        log_returns = np.diff(np.log(prices))
        mus.append(float(np.mean(log_returns) * TRADING_SESSIONS_PER_YEAR))
    return np.array(mus)


def _backtest_oracle(
    val_predictions: pl.DataFrame,
    prices_df: pl.DataFrame,
    rebalance_dates: list[date],
    top_k: int,
    max_risk: float,
    cov_lookback: int,
    min_history: int,
    holding_sessions: int,
    max_weight: float | None = None,
) -> dict[str, Any]:
    """Backtest oracle (predicción perfecta): cota superior coherente.

    Representa el rendimiento alcanzable con **selección perfecta** de activos
    (se eligen, con conocimiento del futuro, los K tickers de mayor retorno
    realizado en el periodo) manteniendo la **misma metodología de construcción
    de cartera** que la estrategia a la que acota: pesos **equiponderados** sobre
    los K seleccionados.

    Motivación del cambio de diseño: la versión anterior re-optimizaba Markowitz
    usando los retornos realizados como retorno esperado (``mu``). Esto inyecta
    el conocimiento del futuro **dos veces** (en la selección y en la ponderación)
    y produce carteras degeneradas con retornos compuestos de miles de millones
    de por ciento, sin valor interpretativo como cota superior. Con pesos
    equiponderados, el oracle sigue siendo un límite superior (selección
    perfecta) pero comparable de forma homogénea con las carteras equiponderadas
    del modelo, aislando así el valor de la *selección* perfecta.

    Los parámetros de optimización (``max_risk``, ``cov_lookback``,
    ``min_history``, ``max_weight``) se mantienen por compatibilidad de interfaz
    pero no se utilizan en la ponderación.
    """
    equity = 1.0
    equity_curve = [equity]
    period_returns = []
    rebalance_details = []

    for i in range(len(rebalance_dates) - 1):
        reb_date = rebalance_dates[i]
        next_reb_date = rebalance_dates[i + 1]

        day_preds = val_predictions.filter(pl.col("signal_date") == reb_date)

        if day_preds.height < top_k:
            period_returns.append(0.0)
            equity_curve.append(equity)
            continue

        # Oracle: seleccionar por retorno realizado (selección perfecta)
        top_tickers = (
            day_preds.sort("target_log_return", descending=True)
            .head(top_k)
            .select(["ticker", "target_log_return"])
        )
        ticker_list = top_tickers.get_column("ticker").to_list()
        # Ponderación equiponderada (misma metodología que la cartera equal_weight
        # del modelo): aísla el efecto de la selección perfecta.
        weights = np.ones(len(ticker_list)) / len(ticker_list)

        period_return = _compute_period_return(
            prices_df=prices_df,
            tickers=ticker_list,
            weights=weights,
            start_date=reb_date,
            end_date=next_reb_date,
        )

        period_returns.append(period_return)
        equity *= (1 + period_return)
        equity_curve.append(equity)

        rebalance_details.append({
            "date": str(reb_date),
            "tickers": ticker_list,
            "weights": weights.tolist(),
            "period_return": period_return,
        })

    metrics = _compute_portfolio_metrics(equity_curve, period_returns, holding_sessions)
    return {
        "equity_curve": equity_curve,
        "period_returns": period_returns,
        "rebalance_details": rebalance_details,
        "metrics": metrics,
    }


def _backtest_equal_weight(
    val_predictions: pl.DataFrame,
    prices_df: pl.DataFrame,
    rebalance_dates: list[date],
    top_k: int,
    holding_sessions: int,
) -> dict[str, Any]:
    """Backtest equiponderado: mismos tickers del modelo, pesos iguales."""
    equity = 1.0
    equity_curve = [equity]
    period_returns = []

    for i in range(len(rebalance_dates) - 1):
        reb_date = rebalance_dates[i]
        next_reb_date = rebalance_dates[i + 1]

        day_preds = val_predictions.filter(pl.col("signal_date") == reb_date)

        if day_preds.height < top_k:
            period_returns.append(0.0)
            equity_curve.append(equity)
            continue

        top_tickers = (
            day_preds.sort("predicted_log_return", descending=True)
            .head(top_k)
        )
        ticker_list = top_tickers.get_column("ticker").to_list()
        weights = np.ones(len(ticker_list)) / len(ticker_list)

        period_return = _compute_period_return(
            prices_df=prices_df,
            tickers=ticker_list,
            weights=weights,
            start_date=reb_date,
            end_date=next_reb_date,
        )

        period_returns.append(period_return)
        equity *= (1 + period_return)
        equity_curve.append(equity)

    metrics = _compute_portfolio_metrics(equity_curve, period_returns, holding_sessions)
    return {
        "equity_curve": equity_curve,
        "period_returns": period_returns,
        "metrics": metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Covarianza y retornos
# ─────────────────────────────────────────────────────────────────────────────


TRADING_SESSIONS_PER_YEAR = 252


def _annual_risk_to_period(max_risk_annual: float, holding_sessions: int) -> float:
    """Convierte una volatilidad ANUAL a la escala del periodo de holding.

    Bajo el supuesto de log-retornos i.i.d., la varianza escala linealmente con
    el tiempo, luego la volatilidad escala con la raiz del tiempo:

        sigma_periodo = sigma_anual * sqrt(holding_sessions / 252)

    Esto mantiene `max_risk` como un parametro intuitivo y comparable entre
    horizontes (volatilidad anual objetivo), pero lo deja en la misma escala que
    `mu` y `Sigma` cuando estos se expresan por periodo de holding.
    """
    return float(max_risk_annual) * math.sqrt(holding_sessions / TRADING_SESSIONS_PER_YEAR)


def _period_covariance(
    prices_df: pl.DataFrame,
    tickers: list[str],
    as_of_date: date,
    lookback_sessions: int,
    min_history: int,
    holding_sessions: int,
) -> np.ndarray | None:
    """Covarianza de log-retornos EN ESCALA DE PERIODO de holding.

    Estima la covarianza DIARIA (annualize=False) y la escala al periodo de
    holding multiplicando por `holding_sessions` (varianza aditiva en el tiempo
    para log-retornos i.i.d.). Es la escala coherente con `mu = r_holding` y con
    `max_risk` convertido a periodo via `_annual_risk_to_period`.
    """
    cov_daily = _estimate_covariance(
        prices_df=prices_df,
        tickers=tickers,
        as_of_date=as_of_date,
        lookback_sessions=lookback_sessions,
        min_history=min_history,
        annualize=False,
    )
    if cov_daily is None:
        return None
    cov = np.atleast_2d(cov_daily) * holding_sessions
    cov += np.eye(cov.shape[0]) * 1e-8
    return cov


def _estimate_covariance(
    prices_df: pl.DataFrame,
    tickers: list[str],
    as_of_date: date,
    lookback_sessions: int,
    min_history: int,
    annualize: bool = True,
) -> np.ndarray | None:
    """Estima la matriz de covarianza de log-retornos para los tickers dados.

    Usa log-retornos diarios del periodo [as_of_date - lookback, as_of_date].
    Por defecto anualiza la covarianza (Sigma_anual = Sigma_diaria * 252),
    asumiendo log-retornos i.i.d., para que quede en la misma escala que los
    retornos esperados anualizados y que el umbral `max_risk` (volatilidad
    anual). La aditividad temporal de la varianza aplica sobre log-retornos.

    Args:
        annualize: Si True, escala la covarianza diaria a anual multiplicando
            por 252 (sesiones bursatiles/anho).

    Returns:
        Matriz de covarianza (K, K) o None si datos insuficientes.
    """
    # Filtrar precios hasta as_of_date para los tickers dados
    hist_prices = (
        prices_df.filter(
            (pl.col("ticker").is_in(tickers))
            & (pl.col("date") <= as_of_date)
        )
        .sort(["ticker", "date"])
    )

    # Tomar las ultimas lookback_sessions por ticker
    hist_prices = hist_prices.group_by("ticker").tail(lookback_sessions + 1)

    # Pivotar a formato wide: date x ticker
    pivot = (
        hist_prices.pivot(
            on="ticker",
            index="date",
            values="close_eur",
        )
        .sort("date")
    )

    # Verificar que todos los tickers estan presentes con suficientes datos
    available_tickers = [t for t in tickers if t in pivot.columns]
    if len(available_tickers) < len(tickers):
        # Usar solo los disponibles
        pass

    if not available_tickers:
        return None

    # Extraer matriz de precios
    price_matrix = pivot.select(available_tickers).to_numpy().astype(np.float64)

    # Verificar minimo de filas
    valid_rows = np.sum(~np.isnan(price_matrix).any(axis=1))
    if valid_rows < min_history:
        return None

    # Calcular log-retornos
    # Reemplazar 0 y negativos para evitar log(0)
    price_matrix = np.where(price_matrix > 0, price_matrix, np.nan)
    log_returns = np.diff(np.log(price_matrix), axis=0)

    # Eliminar filas con NaN
    valid_mask = ~np.isnan(log_returns).any(axis=1)
    log_returns = log_returns[valid_mask]

    if log_returns.shape[0] < min_history:
        return None

    # Covarianza muestral de log-retornos DIARIOS
    cov = np.cov(log_returns, rowvar=False)

    # Anualizar: Sigma_anual = Sigma_diaria * 252 (log-retornos i.i.d.)
    if annualize:
        cov = cov * TRADING_SESSIONS_PER_YEAR

    # Asegurar que es semidefinida positiva (regularizacion minima)
    cov = np.atleast_2d(cov)
    cov += np.eye(cov.shape[0]) * 1e-8

    return cov


def _compute_period_return(
    prices_df: pl.DataFrame,
    tickers: list[str],
    weights: np.ndarray,
    start_date: date,
    end_date: date,
) -> float:
    """Calcula el retorno simple del portfolio entre dos fechas.

    Para cada ticker, obtiene el precio en start_date y end_date (o el mas
    cercano disponible) y calcula el retorno ponderado.

    Returns:
        Retorno simple del portfolio en el periodo.
    """
    # Precios en el rango [start_date, end_date]
    period_prices = prices_df.filter(
        (pl.col("ticker").is_in(tickers))
        & (pl.col("date") >= start_date)
        & (pl.col("date") <= end_date)
    )

    individual_returns = []
    for ticker in tickers:
        ticker_prices = (
            period_prices.filter(pl.col("ticker") == ticker)
            .sort("date")
        )

        if ticker_prices.height < 2:
            individual_returns.append(0.0)
            continue

        entry_price = ticker_prices.get_column("close_eur").item(0)
        exit_price = ticker_prices.get_column("close_eur").item(-1)

        if entry_price > 0:
            individual_returns.append(exit_price / entry_price - 1.0)
        else:
            individual_returns.append(0.0)

    returns_array = np.array(individual_returns)
    portfolio_return = float(np.dot(weights, returns_array))

    return portfolio_return


# ─────────────────────────────────────────────────────────────────────────────
# Nodo: Generar informe Markdown
# ─────────────────────────────────────────────────────────────────────────────


def generate_portfolio_report(
    backtest_results: dict[str, Any],
    markowitz_params: dict[str, Any],
) -> str:
    """Genera un informe Markdown con los resultados del backtest.

    Args:
        backtest_results: Output de run_markowitz_backtest.
        markowitz_params: Parametros del pipeline.

    Returns:
        String con el informe en formato Markdown.
    """
    horizon_key = backtest_results["horizon"]
    horizon_label = backtest_results.get("horizon_label", horizon_key)
    n_rebalances = backtest_results.get("n_rebalances", 0)
    configs = backtest_results.get("configs", {})
    top_k_configs = markowitz_params["top_k_configs"]
    evaluation_split = backtest_results.get("evaluation_split", "test")
    split_label = {
        "test": "test (out-of-sample, modelo congelado)",
        "validation": "validacion (desarrollo)",
        "train": "entrenamiento",
    }.get(evaluation_split, evaluation_split)

    lines = [
        f"# Informe de Cartera Markowitz — Horizonte: {horizon_label}",
        "",
        f"**Horizonte de holding**: {horizon_label} "
        f"({backtest_results.get('holding_sessions', '?')} sesiones)",
        f"**Periodo de evaluacion**: {split_label}",
        f"**Periodos de rebalanceo**: {n_rebalances}",
        f"**Riesgo maximo (volatilidad anual)**: {markowitz_params['max_risk']:.0%}",
        f"**Top-K configuraciones**: {top_k_configs}",
        "",
        "---",
        "",
        "## Resumen de Metricas",
        "",
    ]

    # Tabla resumen
    lines.append(
        "| Configuracion | Retorno Total | Retorno Anualizado "
        "| Volatilidad Anual | Sharpe | Sortino | Max Drawdown |"
    )
    lines.append("|---|---|---|---|---|---|---|")

    benchmark_keys = {"benchmark_sp500", "benchmark_sp500_riskfree"}
    for config_key, result in sorted(configs.items()):
        if config_key in benchmark_keys:
            continue
        m = result.get("metrics", {})
        tr = m.get("total_return")
        ar = m.get("annualized_return")
        av = m.get("annualized_volatility")
        sr = m.get("sharpe_ratio")
        so = m.get("sortino_ratio")
        mdd = m.get("max_drawdown")

        lines.append(
            f"| {config_key} "
            f"| {_fmt_pct(tr)} "
            f"| {_fmt_pct(ar)} "
            f"| {_fmt_pct(av)} "
            f"| {_fmt_num(sr)} "
            f"| {_fmt_num(so)} "
            f"| {_fmt_pct(mdd)} |"
        )

    lines.append("")

    # Comparadores de mercado (solo periodo de validacion)
    present_benchmarks = [k for k in ("benchmark_sp500", "benchmark_sp500_riskfree") if k in configs]
    if present_benchmarks:
        lines.append("## Comparadores de Mercado")
        lines.append("")
        lines.append(
            "> Evaluados **solo sobre el periodo de validacion**, con la misma "
            "malla de rebalanceo que las carteras del modelo (comparacion homogenea)."
        )
        lines.append("")
        lines.append(
            "| Comparador | Retorno Total | Retorno Anualizado "
            "| Volatilidad Anual | Sharpe | Sortino | Max Drawdown |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for bk in present_benchmarks:
            result = configs[bk]
            m = result.get("metrics", {})
            label = result.get("label", bk)
            lines.append(
                f"| {label} "
                f"| {_fmt_pct(m.get('total_return'))} "
                f"| {_fmt_pct(m.get('annualized_return'))} "
                f"| {_fmt_pct(m.get('annualized_volatility'))} "
                f"| {_fmt_num(m.get('sharpe_ratio'))} "
                f"| {_fmt_num(m.get('sortino_ratio'))} "
                f"| {_fmt_pct(m.get('max_drawdown'))} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")

    # Detalle por top-K
    for top_k in top_k_configs:
        lines.append(f"## Top-{top_k}")
        lines.append("")

        # Comparativa de variantes para este top-K: todas las configuraciones
        # cuya clave empieza por "top{top_k}_" (incluye todas las estrategias:
        # max_return, min_variance, mean_variance, target_return, oracle y
        # equal_weight). El sufijo "_" evita confundir top1_ con top10_.
        prefix = f"top{top_k}_"
        existing_variants = sorted(k for k in configs if k.startswith(prefix))

        if existing_variants:
            lines.append("| Variante | Retorno Total | Sharpe | Sortino | Max DD |")
            lines.append("|---|---|---|---|---|")
            for vk in existing_variants:
                m = configs[vk].get("metrics", {})
                label = vk.replace(f"top{top_k}_", "")
                lines.append(
                    f"| {label} "
                    f"| {_fmt_pct(m.get('total_return'))} "
                    f"| {_fmt_num(m.get('sharpe_ratio'))} "
                    f"| {_fmt_num(m.get('sortino_ratio'))} "
                    f"| {_fmt_pct(m.get('max_drawdown'))} |"
                )

        lines.append("")

        # Detalle de rebalanceos para long_only
        lo_key = f"top{top_k}_long_only"
        if lo_key in configs:
            details = configs[lo_key].get("rebalance_details", [])
            if details:
                lines.append(f"### Detalle de rebalanceos (long-only, top-{top_k})")
                lines.append("")
                lines.append("| Fecha | Tickers | Retorno Periodo | Acumulado | Status |")
                lines.append("|---|---|---|---|---|")
                for d in details:
                    tickers_str = ", ".join(d.get("tickers", [])[:5])
                    if len(d.get("tickers", [])) > 5:
                        tickers_str += "..."
                    lines.append(
                        f"| {d.get('date', '?')} "
                        f"| {tickers_str} "
                        f"| {_fmt_pct(d.get('period_return'))} "
                        f"| {_fmt_pct(d.get('cumulative_return'))} "
                        f"| {d.get('optimization_status', d.get('status', '?'))} |"
                    )
                lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _fmt_pct(value: float | None) -> str:
    """Formatea un valor como porcentaje."""
    if value is None:
        return "—"
    return f"{value:.2%}"


def _fmt_num(value: float | None) -> str:
    """Formatea un valor numerico."""
    if value is None:
        return "—"
    return f"{value:.3f}"
