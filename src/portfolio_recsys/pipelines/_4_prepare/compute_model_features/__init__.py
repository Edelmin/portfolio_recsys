"""Pipeline compute_model_features.

Genera features de mercado y fundamentales a partir del dataset consolidado
(COMPANY_{sector}), aplicando las transformaciones necesarias para alimentar
modelos de series temporales:

- Retornos logaritmicos multi-horizonte.
- Log del precio y tipo de cambio.
- FX return diario.
- Signed log1p de fundamentales.
- Ratios financieros renombrados.
- Dias desde disponibilidad del fundamental.
- Trading targets (entry/exit con lag de ejecucion).
- Split temporal para backtesting.
- Escalado robusto (mediana + IQR).

Produce dos variantes:
- market: solo features de mercado (precio, FX, retornos).
- enriched: mercado + fundamentales.
"""
