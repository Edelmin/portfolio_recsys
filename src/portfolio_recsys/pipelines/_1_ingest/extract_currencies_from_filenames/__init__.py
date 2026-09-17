"""Pipeline extract_currencies_from_filenames.

Extrae la moneda y el ticker de cada empresa a partir de los nombres de fichero
de los estados financieros historicos (patron: {currency} - {ticker} - Financials...).
Produce un dataset con las divisas unicas por ticker y sector, que sirve como
input para fetch_exchange_rates.
"""
