# app.py
import io
import chardet
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Sistema de Análisis de Reportes By AVR",
    layout="wide",
    page_icon="📊"
)

# -------------------------
# Carga y preparación de datos (web-friendly)
# -------------------------
REPLACE_NA = ['', 'nsm', 'NA', 'N/A', 'nan', 'None']

@st.cache_data(show_spinner=True)
def load_data_from_bytes(file_bytes: bytes) -> pd.DataFrame | None:
    """Lee TXT '|' separado probando varias codificaciones y aplica conversiones."""
    def try_encodings(b: bytes) -> pd.DataFrame | None:
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        for enc in encodings:
            try:
                f = io.BytesIO(b)
                chunks = []
                for chunk in pd.read_csv(
                    f,
                    delimiter='|',
                    encoding=enc,
                    chunksize=10000,
                    on_bad_lines='warn',
                    dtype=str
                ):
                    chunks.append(chunk)
                return pd.concat(chunks, ignore_index=True)
            except UnicodeDecodeError:
                continue
            except Exception as e:
                # Log suave, no romper
                print(f"Intento fallido con {enc}: {e}")
                continue
        # Fallback con chardet
        det = chardet.detect(b[:50000]).get('encoding') or 'utf-8'
        try:
            f = io.BytesIO(b)
            return pd.read_csv(
                f,
                delimiter='|',
                encoding=det,
                dtype=str,
                on_bad_lines='warn'
            )
        except Exception as e:
            print(f"Fallo incluso con chardet ({det}): {e}")
            return None

    def convert_specific_columns(df: pd.DataFrame) -> pd.DataFrame:
        # Fechas con formato exacto dd/mm/yyyy HH:MM:SS
        def convert_datetime(col):
            return pd.to_datetime(col, format='%d/%m/%Y %H:%M:%S', errors='coerce')

        datetime_cols = [
            'FECHA DE REGISTRO',
            'FECHA LIMITE RESPUESTA',
            'FECHAHORADICTADO',
            'FECHAHORALLEGADA',
            'FECHAHORAATENCION'
        ]
        for c in datetime_cols:
            if c in df.columns:
                df[c] = convert_datetime(df[c])

        # Horas a Timedelta
        if 'TIEMPO DESDE DICTADO [HRS]' in df.columns:
            df['TIEMPO DESDE DICTADO [HRS]'] = pd.to_numeric(
                df['TIEMPO DESDE DICTADO [HRS]'], errors='coerce'
            ).apply(lambda x: pd.Timedelta(hours=x) if pd.notna(x) else pd.NaT)

        # Numéricos (si existen)
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
        for c, dtype in numeric_cols.items():
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype(dtype)

        # Resto a string (para filtros categóricos)
        text_cols = df.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            df[text_cols] = df[text_cols].astype('string')

        # Manejo de otras columnas con "fecha" en el nombre
        generic_date_cols = [
            c for c in df.columns
            if 'fecha' in c.lower() and c not in datetime_cols
        ]
        for c in generic_date_cols:
            try:
                # Si dice "fechahora" => datetime; si no => fecha
                if 'fechahora' in c.lower():
                    df[c] = pd.to_datetime(df[c], dayfirst=True, errors='coerce')
                else:
                    df[c] = pd.to_datetime(df[c], dayfirst=True, errors='coerce')
            except Exception as e:
                print(f"No se pudo convertir {c} a fecha: {e}")
                continue

        return df

    df = try_encodings(file_bytes)
    if df is None:
        return None

    # Limpieza
    df = df.replace(REPLACE_NA, pd.NA)
    df.dropna(axis=1, how='all', inplace=True)

    # Conversiones específicas
    df = convert_specific_columns(df)
    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("### 🔎 Filtros")
    selected_cols = st.sidebar.multiselect(
        "Selecciona columnas para filtrar",
        options=list(df.columns),
        default=[]
    )
    filtered = df.copy()

    for col in selected_cols:
        if pd.api.types.is_datetime64_any_dtype(filtered[col]):
            # Rango de fechas
            min_dt = pd.to_datetime(filtered[col], errors='coerce').min()
            max_dt = pd.to_datetime(filtered[col], errors='coerce').max()
            if pd.isna(min_dt) or pd.isna(max_dt):
                continue
            start, end = st.sidebar.date_input(
                f"Rango {col}",
                value=(min_dt.date(), max_dt.date()),
                min_value=min_dt.date(),
                max_value=max_dt.date()
            )
            if start and end:
                start_ts = pd.to_datetime(start)
                end_ts = pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                filtered = filtered[(filtered[col] >= start_ts) & (filtered[col] <= end_ts)]
        else:
            # Multiselección categórica
            vals = (
                filtered[col]
                .astype("string")
                .dropna()
                .unique()
                .tolist()
            )
            chosen = st.sidebar.multiselect(f"Valores de {col}", options=sorted(map(str, vals)))
            if chosen:
                filtered = filtered[filtered[col].astype(str).isin(chosen)]

    st.sidebar.info(f"Registros después de filtrar: **{len(filtered):,}**")
    return filtered


def draw_summary(df: pd.DataFrame):
    st.subheader("📌 Resumen rápido")
    main_cols = ['ESTADO REPORTE', 'PRIORIDAD', 'COMUNA', 'TIPO DAÑO']
    cols = st.columns(len(main_cols))
    for i, c in enumerate(main_cols):
        with cols[i]:
            if c in df.columns:
                counts = df[c].astype("string").value_counts(dropna=True).head(6)
                st.markdown(f"**{c}**")
                for idx, val in counts.items():
                    st.write(f"- {idx}: {val}")
            else:
                st.caption(f"*No está la columna **{c}***")


def draw_chart(df: pd.DataFrame):
    st.subheader("📈 Distribución categórica (Top 15)")
    cat_cols = df.select_dtypes(include=['string', 'object', 'category']).columns.tolist()
    if not cat_cols:
        st.info("No hay columnas categóricas para graficar.")
        return

    col = st.selectbox("Selecciona la columna categórica", options=cat_cols, index=0)
    top_values = (
        df[col].astype("string").fillna("Sin dato").value_counts().head(15)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(y=top_values.index.astype(str), x=top_values.values, ax=ax)
    ax.set_title(f"Distribución de {col} (Top 15)")
    ax.set_xlabel("Cantidad")
    ax.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


def main():
    st.title("📊 Sistema de Análisis de Reportes By AVR (Web)")
    st.caption("Sube un archivo TXT delimitado por **|**. La app intentará detectar la codificación y convertir fechas/numéricos automáticamente.")

    up = st.file_uploader("Subir archivo TXT", type=["txt"])
    if up is None:
        st.info("👉 Esperando archivo. Ejemplo: `mis_reportes.txt` delimitado por `|`.")
        return

    # Cargar datos
    with st.spinner("Cargando y procesando datos..."):
        df = load_data_from_bytes(up.read())

    if df is None or df.empty:
        st.error("No se pudo leer el archivo o está vacío.")
        return

    st.success(f"Archivo cargado correctamente ({len(df):,} registros, {len(df.columns)} columnas)")

    # Filtros
    filtered = apply_filters(df)

    # Datos (primeros 100)
    st.subheader("🧾 Vista de datos (Top 100)")
    st.dataframe(filtered.head(100), use_container_width=True)

    # Descarga CSV filtrado
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar CSV filtrado",
        data=csv_bytes,
        file_name="reportes_filtrados.csv",
        mime="text/csv"
    )

    # Resumen + gráfico
    draw_summary(filtered)
    draw_chart(filtered)


if __name__ == "__main__":
    main()
