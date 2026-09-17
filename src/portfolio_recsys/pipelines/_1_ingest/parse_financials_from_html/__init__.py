"""Pipeline parse_financials_from_html.

Extrae y estructura los datos financieros de empresas a partir de ficheros HTML
descargados manualmente del portal TIKR, organizados por sector y capitalización
(large/small cap).

Entrada: ficheros HTML con tablas de empresas por sector (símbolo, empresa,
         industria, valor de mercado, sector, ubicación).
Salida:  CSVs limpios con las columnas normalizadas, uno por sector y capitalización.
"""
