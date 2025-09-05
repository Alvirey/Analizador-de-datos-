import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import chardet

# ===============================
# Función para cargar y procesar datos
# ===============================
def load_data(uploaded_file):
    def detect_and_try_encodings(file):
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        for encoding in encodings:
            try:
                file.seek(0)
                chunks = []
                for chunk in pd.read_csv(
                    file, 
                    delimiter='|', 
                    encoding=encoding,
                    chunksize=10000,
                    on_bad_lines='warn',
                    dtype=str
                ):
                    chunks.append(chunk)
                return pd.concat(chunks, ignore_index=True)
            except UnicodeDecodeError:
                continue
            except Exception:
                continue
        return None

    def convert_specific_columns(df):
        datetime_cols = [
            'FECHA DE REGISTRO',
            'FECHA LIMITE RESPUESTA',
            'FECHAHORADICTADO',
            'FECHAHORALLEGADA',
            'FECHAHORAATENCION'
        ]

        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format='%d/%m/%Y %H:%M:%S', errors='coerce')

        if 'TIEMPO DESDE DICTADO [HRS]' in df.columns:
            df['TIEMPO DESDE DICTADO [HRS]'] = pd.to_numeric(
                df['TIEMPO DESDE DICTADO [HRS]'], errors='coerce'
            ).apply(lambda x: pd.Timedelta(hours=x) if pd.notna(x) else pd.NA)

        numeric_cols = {
            'CÓDIGO': 'int64',
            'TELEFONO': 'Int64',
            'X': 'Int64',
            'Y': 'Int64',
            'CODIGOACTUAL': 'Int64',
            'CODIGOAPOYO': 'Int64',
            'PINTADOAPOYO': 'Int64',
            'LATITUD': 'float64',
            'LONGITUD': 'float64'
        }

        for col, dtype in numeric_cols.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)

        text_cols = df.select_dtypes(include=['object']).columns
        df[text_cols] = df[text_cols].astype('string')

        return df

    try:
        file_content = io.BytesIO(uploaded_file.read())
        df = detect_and_try_encodings(file_content)

        if df is None:
            file_content.seek(0)
            sample = file_content.read(50000)
            file_content.seek(0)
            detected_encoding = chardet.detect(sample)['encoding']
            df = pd.read_csv(file_content, delimiter='|', encoding=detected_encoding, dtype=str, on_bad_lines='warn')

        if df is None:
            st.error("No se pudo leer el archivo con ninguna codificación probada")
            return None

        df = df.replace(['', 'nsm', 'NA', 'N/A', 'nan', 'None'], pd.NA)
        df.dropna(axis=1, how='all', inplace=True)
        df = convert_specific_columns(df)

        return df

    except Exception as e:
        st.error(f"Error crítico al procesar el archivo: {str(e)}")
        return None

# ===============================
# Interfaz con Streamlit
# ===============================
st.set_page_config(page_title="Sistema de Análisis de Reportes", layout="wide")
st.title("📊 Sistema de Análisis de Reportes By AVR")

uploaded_file = st.file_uploader("Sube un archivo TXT", type=["txt"])

if uploaded_file:
    df = load_data(uploaded_file)

    if df is not None and not df.empty:
        st.success(f"Archivo cargado correctamente con {len(df)} registros")

        # Mostrar filtros dinámicos
        st.sidebar.header("Filtros")
        filter_columns = [
            'CÓDIGO', 'ESTADO REPORTE', 'COMUNA', 'BARRIO',
            'GRUPO DE TRABAJO', 'TIPO DAÑO', 'FECHA DE REGISTRO',
            'FECHAHORAATENCION', 'NOMBREADMINISTRATIVO', 'TIPO ASIGNACIÓN'
        ]

        filtered_df = df.copy()

        for col in filter_columns:
            if col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    start, end = st.sidebar.date_input(
                        f"Rango de {col}", 
                        value=[df[col].min().date(), df[col].max().date()]
                    )
                    filtered_df = filtered_df[
                        (filtered_df[col].dt.date >= start) & 
                        (filtered_df[col].dt.date <= end)
                    ]
                else:
                    options = ["Todos"] + df[col].dropna().unique().astype(str).tolist()
                    choice = st.sidebar.selectbox(f"Filtrar por {col}", options)
                    if choice != "Todos":
                        filtered_df = filtered_df[filtered_df[col].astype(str) == choice]

        # Mostrar tabla
        st.subheader("Datos filtrados")
        st.dataframe(filtered_df.head(100))

        # Resumen
        st.subheader("Resumen de Categorías")
        main_cat_cols = ['ESTADO REPORTE', 'NOMBREADMINISTRATIVO', 'COMUNA', 'TIPO DAÑO']
        for col in main_cat_cols:
            if col in filtered_df.columns:
                st.write(f"### {col}")
                st.write(filtered_df[col].value_counts().head(5))

        # Gráficos
        st.subheader("Visualización")
        cat_cols = filtered_df.select_dtypes(include=['string']).columns
        if len(cat_cols) > 0:
            selected_col = st.selectbox("Seleccionar columna categórica", cat_cols)
            top_values = filtered_df[selected_col].value_counts().iloc[:15]

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                y=top_values.index.astype(str),
                x=top_values.values,
                ax=ax,
                palette="Blues_d"
            )
            ax.set_title(f"Distribución de {selected_col} (Top 15)")
            st.pyplot(fig)