"""Pipeline clean_financial_statements.

Transforma los Parquets crudos de estados financieros (formato ancho: métricas
como filas, fechas como columnas) a formato limpio transpuesto (fechas como filas,
métricas como columnas), filtrando solo los campos seleccionados por configuración.

Produce un Parquet por empresa con las métricas de las 3 hojas unidas por fecha.
"""
