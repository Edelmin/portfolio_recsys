import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pandas import DataFrame

path = "data/01_raw/FicherosHtmlSectores/"
dest_path = "data/02_intermediate/DatosSectoresExtraido/"


def normalize_spaces(df: DataFrame) -> DataFrame:

    WEIRD_SPACES = [
        "\u202f",  # NARROW NO-BREAK SPACE
        "\u00a0",  # NO-BREAK SPACE
        "\u2009",  # THIN SPACE
        "\u2007",  # FIGURE SPACE
        "\u200b",  # ZERO WIDTH SPACE
        "\ufeff",  # BOM
    ]
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col].astype("string")
        for sp in WEIRD_SPACES:
            s = s.str.replace(sp, " ", regex=False)
        # opcional: colapsa múltiples espacios
        s = s.str.replace(r"\s+", " ", regex=True).str.strip()
        df[col] = s
    return df


def extract_data_from_html(file) -> DataFrame:

    # Abrir y leer el archivo como un string
    html = file

    # Parsear el HTML
    soup = BeautifulSoup(html, "html.parser")

    # Extraer los datos de las filas de la tabla
    data = []

    # Buscar todas las filas de la tabla
    rows = soup.find_all("tr")

    # Iterar sobre cada fila y extraer los datos de las celdas
    for row in rows:
        cols = row.find_all("td")
        # Extraer el texto de cada celda
        cols = [ele.text.strip() for ele in cols]
        data.append(cols)

    # Crear un DataFrame con pandas
    df = pd.DataFrame(
        data,
        columns=[
            "Favorito",
            "Símbolo",
            "Empresa",
            "Industria",
            "Valor de Mercado",
            "Sector",
            "Ubicación",
        ],
    )

    # Mostrar el DataFrame
    df[
        [
            "Símbolo",
            "Empresa",
            "Industria",
            "Valor de Mercado",
            "Sector",
            "Ubicación",
        ]
    ]

    if sum(df.duplicated()) == 0:
        print(f"El archivo se guardó sin repeticiones.")

    else:
        print("El archivo se guardó con entradas repetidas.")

    df = normalize_spaces(df)

    return df[
        [
            "Símbolo",
            "Empresa",
            "Industria",
            "Valor de Mercado",
            "Sector",
            "Ubicación",
        ]
    ]
