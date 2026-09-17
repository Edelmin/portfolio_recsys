"""Mapeo canonico de sectores.

Este modulo define la tabla de traduccion entre los nombres de carpetas/ficheros
reales (en espanol) y los nombres canonicos usados en datasets y codigo (en ingles).

CONVENCION:
- Los identificadores de datasets usan SIEMPRE el nombre canonico en ingles.
- Los paths al filesystem usan el nombre real de la carpeta (espanol, puede tener tildes).
- Este mapeo se usa en todos los pipelines para garantizar consistencia.

Uso:
    from portfolio_recsys.pipelines.sector_mapping import SECTORS, CAPS
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SectorInfo:
    """Informacion de un sector."""

    canonical: str
    """Nombre canonico en ingles (usado en datasets)."""

    folder_html: str
    """Nombre del fichero HTML (sin extension, sin cap prefix)."""

    folder_fs: str | None
    """Nombre de la carpeta de estados financieros (puede tener tildes). None si no existe."""


# Tabla maestra de sectores
# El key es el nombre canonico (ingles), que es lo que se usa en los datasets.
SECTORS: dict[str, SectorInfo] = {
    "RealEstate": SectorInfo(
        canonical="RealEstate",
        folder_html="BienesRaices",
        folder_fs="BienesRa\u00edces",
    ),
    "ConsumerDiscretionary": SectorInfo(
        canonical="ConsumerDiscretionary",
        folder_html="ConsumoDiscrecional",
        folder_fs="ConsumoDiscrecional",
    ),
    "Energy": SectorInfo(
        canonical="Energy",
        folder_html="Energia",
        folder_fs="Energ\u00eda",
    ),
    "Financials": SectorInfo(
        canonical="Financials",
        folder_html="Financiero",
        folder_fs="Financiero",
    ),
    "Industrials": SectorInfo(
        canonical="Industrials",
        folder_html="Industria",
        folder_fs="Industria",
    ),
    "Materials": SectorInfo(
        canonical="Materials",
        folder_html="MateriasPrimas",
        folder_fs="MateriasPrimas",
    ),
    "ConsumerStaples": SectorInfo(
        canonical="ConsumerStaples",
        folder_html="ProductosDePrimeraNecesidad",
        folder_fs="ProductosDePrimeraNecesidad",
    ),
    "Healthcare": SectorInfo(
        canonical="Healthcare",
        folder_html="Salud",
        folder_fs="Salud",
    ),
    "Utilities": SectorInfo(
        canonical="Utilities",
        folder_html="Servicios",
        folder_fs="Servicios",
    ),
    "CommunicationServices": SectorInfo(
        canonical="CommunicationServices",
        folder_html="ServiciosDeComunicacion",
        folder_fs="ServiciosDeComunicaci\u00f3n",
    ),
    "InformationTechnology": SectorInfo(
        canonical="InformationTechnology",
        folder_html="TecnologiasDeLaInformacion",
        folder_fs="Tecnolog\u00edasDeLaInformaci\u00f3n",
    ),
}

# Capitalizaciones
CAPS = ["LargeCaps", "SmallCaps"]

# Helpers
SECTORS_WITH_FS = [k for k, v in SECTORS.items() if v.folder_fs is not None]
"""Sectores que tienen carpeta de estados financieros disponible."""

# Mapeo inverso: nombre carpeta FS (sin tildes, key del catalogo) -> canonical
FS_KEY_TO_CANONICAL = {
    v.folder_html: k for k, v in SECTORS.items() if v.folder_fs is not None
}
"""Mapeo del nombre usado en FS_{key} del catalogo al nombre canonico."""
