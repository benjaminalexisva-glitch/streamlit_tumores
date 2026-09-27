import streamlit as st
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import pandas as pd


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Clasificación de tumores cerebrales",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# RUTAS
# ============================================================

# Carpeta donde está este mismo archivo app_inferencia.py
BASE_DIR = Path(__file__).resolve().parent

# Modelo
MODELO = BASE_DIR / "best.pt"

# Imágenes que se mostrarán automáticamente
CARPETA_IMAGENES = BASE_DIR / "imagenes_inferencias"

EXTENSIONES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
}


# ============================================================
# PARÁMETROS
# ============================================================

UMBRAL = 0.75


# ============================================================
# COMPROBACIONES
# ============================================================

if not MODELO.exists():
    st.error(
        f"No se encontró el modelo:\n\n{MODELO}"
    )
    st.stop()

if not CARPETA_IMAGENES.exists():
    st.error(
        f"No se encontró la carpeta de imágenes:\n\n"
        f"{CARPETA_IMAGENES}"
    )
    st.stop()


# ============================================================
# CARGAR MODELO
# ============================================================

@st.cache_resource
def cargar_modelo():
    return YOLO(str(MODELO))


modelo = cargar_modelo()


# ============================================================
# CABECERA
# ============================================================

st.title("🧠 Clasificación automática de imágenes MRI")

st.markdown(
    """
    Esta aplicación utiliza un modelo de clasificación basado en
    **YOLO11m-cls** para distinguir imágenes de tumores cerebrales
    clasificados como **benignos** o **malignos**.
    """
)

st.info(
    f"Predicciones con confianza inferior a "
    f"**{UMBRAL * 100:.0f}%** se consideran "
    f"**inciertas / fuera del dominio**."
)

st.warning(
    "⚠️ Modelo experimental utilizado con fines académicos. "
    "Los resultados no constituyen un diagnóstico médico."
)

st.divider()


# ============================================================
# BUSCAR IMÁGENES
# ============================================================

imagenes = sorted(
    [
        archivo
        for archivo in CARPETA_IMAGENES.iterdir()
        if archivo.suffix.lower() in EXTENSIONES
    ]
)


st.header("📂 Imágenes evaluadas")

st.write(
    f"Se encontraron **{len(imagenes)} imágenes** "
    f"para realizar la inferencia."
)


if len(imagenes) == 0:

    st.error(
        "No existen imágenes compatibles en "
        "`imagenes_inferencias`."
    )

    st.stop()


# ============================================================
# MOSTRAR IMÁGENES ORIGINALES
# ============================================================

st.subheader("Imágenes de entrada")

columnas = st.columns(5)

for i, archivo in enumerate(imagenes):

    with columnas[i % 5]:

        imagen = Image.open(
            archivo
        ).convert("RGB")

        st.image(
            imagen,
            caption=f"Imagen {i + 1}",
            use_container_width=True
        )

        st.caption(
            archivo.name
        )


st.divider()


# ============================================================
# REALIZAR INFERENCIAS AUTOMÁTICAMENTE
# ============================================================

st.header("🔬 Resultados de inferencia")

registros = []


with st.spinner(
    "Analizando imágenes con el modelo..."
):

    for numero, archivo in enumerate(
        imagenes,
        start=1
    ):

        # ----------------------------------------------------
        # Predicción
        # ----------------------------------------------------

        resultado = modelo.predict(
            source=str(archivo),
            imgsz=320,
            verbose=False
        )[0]


        probabilidades = (
            resultado.probs
            .data
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # Encontrar las clases
        # ----------------------------------------------------

        try:

            idx_benigno = next(
                k
                for k, v in resultado.names.items()
                if v.lower().strip() == "benigno"
            )

            idx_maligno = next(
                k
                for k, v in resultado.names.items()
                if v.lower().strip() == "maligno"
            )

        except StopIteration:

            st.error(
                "Las clases del modelo no se llaman "
                "'benigno' y 'maligno'."
            )

            st.write(
                "Clases detectadas:",
                resultado.names
            )

            st.stop()


        # ----------------------------------------------------
        # Probabilidades
        # ----------------------------------------------------

        prob_benigno = float(
            probabilidades[idx_benigno]
        )

        prob_maligno = float(
            probabilidades[idx_maligno]
        )


        confianza = max(
            prob_benigno,
            prob_maligno
        )


        # ----------------------------------------------------
        # Decisión
        # ----------------------------------------------------

        if confianza < UMBRAL:

            prediccion = (
                "NO CORRESPONDE / INCIERTO"
            )

        elif prob_maligno > prob_benigno:

            prediccion = "MALIGNO"

        else:

            prediccion = "BENIGNO"


        # ----------------------------------------------------
        # Guardar resultado
        # ----------------------------------------------------

        registros.append(
            {
                "Imagen": numero,
                "Archivo": archivo.name,
                "Predicción": prediccion,
                "Benigno (%)":
                    prob_benigno * 100,
                "Maligno (%)":
                    prob_maligno * 100,
                "Confianza (%)":
                    confianza * 100
            }
        )


# ============================================================
# RESULTADOS INDIVIDUALES
# ============================================================

st.success(
    "✅ Inferencia finalizada correctamente"
)


for i, registro in enumerate(registros):

    archivo = imagenes[i]

    col_imagen, col_resultado = st.columns(
        [1, 1.4]
    )


    # ========================================================
    # IMAGEN
    # ========================================================

    with col_imagen:

        imagen = Image.open(
            archivo
        ).convert("RGB")

        st.image(
            imagen,
            caption=registro["Archivo"],
            use_container_width=True
        )


    # ========================================================
    # RESULTADO
    # ========================================================

    with col_resultado:

        st.subheader(
            f"Imagen {registro['Imagen']}"
        )


        pred = registro[
            "Predicción"
        ]


        # ----------------------------------------------------
        # Resultado principal
        # ----------------------------------------------------

        if pred == "MALIGNO":

            st.error(
                "## 🔴 MALIGNO"
            )

        elif pred == "BENIGNO":

            st.success(
                "## 🟢 BENIGNO"
            )

        else:

            st.warning(
                "## 🟡 INCIERTO / "
                "FUERA DEL DOMINIO"
            )


        # ----------------------------------------------------
        # Métricas
        # ----------------------------------------------------

        c1, c2, c3 = st.columns(3)


        c1.metric(
            "Benigno",
            f"{registro['Benigno (%)']:.2f}%"
        )


        c2.metric(
            "Maligno",
            f"{registro['Maligno (%)']:.2f}%"
        )


        c3.metric(
            "Confianza",
            f"{registro['Confianza (%)']:.2f}%"
        )


        # ----------------------------------------------------
        # Barras de probabilidad
        # ----------------------------------------------------

        st.write(
            "**Probabilidad de tumor benigno**"
        )

        st.progress(
            registro["Benigno (%)"] / 100
        )


        st.write(
            "**Probabilidad de tumor maligno**"
        )

        st.progress(
            registro["Maligno (%)"] / 100
        )


    st.divider()


# ============================================================
# TABLA RESUMEN
# ============================================================

st.header(
    "📊 Resumen de las inferencias"
)


df = pd.DataFrame(
    registros
)


for columna in [
    "Benigno (%)",
    "Maligno (%)",
    "Confianza (%)"
]:

    df[columna] = (
        df[columna]
        .round(2)
    )


st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# ESTADÍSTICAS SIMPLES
# ============================================================

st.subheader(
    "Resumen de clasificaciones"
)


cantidad_benignos = sum(
    df["Predicción"] == "BENIGNO"
)

cantidad_malignos = sum(
    df["Predicción"] == "MALIGNO"
)

cantidad_inciertos = sum(
    df["Predicción"]
    == "NO CORRESPONDE / INCIERTO"
)


c1, c2, c3 = st.columns(3)


c1.metric(
    "🟢 Benignos",
    cantidad_benignos
)


c2.metric(
    "🔴 Malignos",
    cantidad_malignos
)


c3.metric(
    "🟡 Inciertos",
    cantidad_inciertos
)


# ============================================================
# DESCARGA CSV
# ============================================================

csv = df.to_csv(
    index=False
).encode(
    "utf-8-sig"
)


st.download_button(
    label="⬇️ Descargar resultados CSV",
    data=csv,
    file_name="resultados_inferencias.csv",
    mime="text/csv"
)


# ============================================================
# PIE
# ============================================================

st.divider()

st.caption(
    "Modelo experimental de clasificación de imágenes MRI "
    "basado en YOLO11m-cls."
)