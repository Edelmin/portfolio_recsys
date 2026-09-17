"""Pipeline impute_missing_values.

Rellena huecos en los datasets consolidados por sector (COMPANY_*):
- Campos de estados financieros: interpolacion lineal + forward-fill al final.
- Precios diarios (close_eur): forward-fill.
"""
