"""Nodos del pipeline parse_financials_from_html.

Contiene la logica de parsing de ficheros HTML descargados manualmente desde
TIKR que incluyen tablas con datos de empresas cotizadas por sector. Cada tabla tiene
las columnas: Favorito, Simbolo, Empresa, Industria, Valor de Mercado,
Sector, Ubicacion.

El nodo principal (extract_data_from_html) recibe el contenido HTML como string,
parsea la tabla con BeautifulSoup, normaliza espacios unicode y devuelve un
polars DataFrame limpio sin la columna 'Favorito' ni registros duplicados.
"""

import polars as pl
from bs4 import BeautifulSoup


def normalize_spaces_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Reemplaza caracteres unicode de espacio no estandar por espacios normales."""
    weird_spaces = [
        "\u202f",  # NARROW NO-BREAK SPACE
        "\u00a0",  # NO-BREAK SPACE
        "\u2009",  # THIN SPACE
        "\u2007",  # FIGURE SPACE
        "\u200b",  # ZERO WIDTH SPACE
        "\ufeff",  # BOM
    ]
    str_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.Utf8]
    for col in str_cols:
        expr = pl.col(col)
        for sp in weird_spaces:
            expr = expr.str.replace_all(sp, " ", literal=True)
        expr = expr.str.replace_all(r"\s+", " ").str.strip_chars()
        df = df.with_columns(expr.alias(col))
    return df


def extract_data_from_html(html: str) -> pl.DataFrame:
    """Parsea un fichero HTML con una tabla de empresas y devuelve un polars DataFrame.

    Args:
        html: Contenido HTML como string (leido por TextDataset).

    Returns:
        polars DataFrame con columnas: Simbolo, Empresa, Industria,
        Valor de Mercado, Sector, Ubicacion. Sin duplicados y con espacios normalizados.
    """
    soup = BeautifulSoup(html, "html.parser")

    data = []
    rows = soup.find_all("tr")
    for row in rows:
        cols = row.find_all("td")
        cols = [ele.text.strip() for ele in cols]
        if len(cols) == 7:  # Ignorar filas de cabecera (<th>) o vacías
            data.append(cols)

    df = pl.DataFrame(
        data,
        schema=[
            "Favorito",
            "Simbolo",
            "Empresa",
            "Industria",
            "Valor de Mercado",
            "Sector",
            "Ubicacion",
        ],
        orient="row",
    )

    # Eliminar columna Favorito
    df = df.drop("Favorito")

    # Eliminar duplicados
    n_dupes = df.shape[0] - df.unique().shape[0]
    if n_dupes == 0:
        print("El archivo se proceso sin repeticiones.")
    else:
        print(f"El archivo tenia {n_dupes} entradas repetidas (eliminadas).")
        df = df.unique()

    df = normalize_spaces_polars(df)

    return df
