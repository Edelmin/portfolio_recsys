# 📦 Portfolio Recomendation System Comparison

Proyecto de comparación de sistemas de recomendación de carteras de inversión, desarrollado con **Kedro**.

---

# 🚀 Setup completo del entorno (con uv)

Este proyecto utiliza **uv** como gestor de entorno y dependencias.

Se asume que **Python ya está instalado** en el sistema.

---

## 1️⃣ Verificar Python instalado

```bash
python --version
```

Se recomienda **Python 3.12**.

Si necesitas instalar Python 3.12 usando uv:

```bash
uv python install 3.12
```

---

## 2️⃣ Instalar uv

### Windows (PowerShell)

```powershell
pip install --upgrade pip
pip install uv
```

### Linux / macOS

```bash
pip install --upgrade pip
pip install uv
```

Verificar instalación:

```bash
uv --version
```

---

## 3️⃣ Clonar el repositorio

```bash
git clone git@github.com:Edelmin/portfolio_recsys_comparison.git
cd portfolio-recsys-comparison
```

---

## 4️⃣ Crear entorno virtual del proyecto

```bash
uv venv .venv
```

Esto crea la carpeta:

```
.venv/
```

(No se versiona; está incluida en `.gitignore`)

---

## 5️⃣ Sincronizar dependencias del proyecto

Instalar dependencias principales y de desarrollo:

```bash
uv sync --extra dev
```

Este comando:

- Lee `pyproject.toml`
- Usa `uv.lock` (si existe)
- Instala versiones exactas y reproducibles
- Deja el entorno completamente sincronizado

---

## 6️⃣ Ejecutar Kedro

Sin activar el entorno manualmente:

```bash
uv run kedro run
```

También puedes ejecutar:

```bash
uv run kedro viz
```

---

## 7️⃣ (Opcional) Activar el entorno manualmente

### Windows

```powershell
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

---

## 8️⃣ Crear kernel de Jupyter

Para usar el entorno en notebooks:

```bash
uv run prc-kernel
```

Luego selecciona en Jupyter o VS Code:

```
Portfolio Recsys (dev)
```

---

# 🔄 Reinstalación limpia del entorno

Si necesitas reconstruir todo desde cero:

```bash
rm -rf .venv
uv sync --extra dev
```

---

# 📁 Estructura relevante

```
.venv/              # Entorno virtual (no versionado)
pyproject.toml      # Dependencias del proyecto
uv.lock             # Lockfile reproducible
conf/               # Configuración Kedro
src/                # Código del proyecto
```

---

# ✅ Comandos habituales

```bash
uv run kedro run
uv run kedro viz
uv run pytest
uv run python script.py
```

---

# 🏁 Notas importantes

- `.venv/` está en `.gitignore`
- `uv.lock` garantiza reproducibilidad entre equipos
- No se usa conda
- El entorno es aislado por proyecto
- Python recomendado: 3.11