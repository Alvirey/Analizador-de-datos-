import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import chardet
import calendar
from datetime import datetime, timedelta
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Análisis de Reportes By AVR",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Función para cargar y procesar los datos
@st.cache_data
def load_data(uploaded_file):
    def detect_and_try_encodings(file):
        """Prueba múltiples codificaciones"""
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
            except Exception as e:
                print(f"Intento fallido con {encoding}: {str(e)}")
                continue
        
        return None

    def convert_specific_columns(df):
        """Conversiones específicas de tipos"""
        # Función para convertir fechas con formato exacto
        def convert_datetime(col):
            return pd.to_datetime(
                col, 
                format='%d/%m/%Y %H:%M:%S', 
                errors='coerce'
            )

        # Lista de columnas de fecha/hora
        datetime_cols = [
            'FECHA DE REGISTRO',
            'FECHA LIMITE RESPUESTA',
            'FECHAHORADICTADO',
            'FECHAHORALLEGADA',
            'FECHAHORAATENCION'
        ]

        # Convertir columnas específicas
        for col in datetime_cols:
            if col in df.columns:
                df[col] = convert_datetime(df[col])

        # Manejo especial para la columna de horas
        if 'TIEMPO DESDE DICTADO [HRS]' in df.columns:
            df['TIEMPO DESDE DICTADO [HRS]'] = pd.to_numeric(
                df['TIEMPO DESDE DICTADO [HRS]'], 
                errors='coerce'
            ).apply(lambda x: pd.Timedelta(hours=x) if pd.notna(x) else pd.NA)

        # Conversión de otros tipos
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

        # Convertir el resto a string
        text_cols = df.select_dtypes(include=['object']).columns
        df[text_cols] = df[text_cols].astype('string')

        return df

    try:
        # Convertir el contenido a un objeto similar a archivo
        file_content = io.BytesIO(uploaded_file.read())
        
        # Paso 1: Intentar con múltiples codificaciones
        df = detect_and_try_encodings(file_content)
        
        # Paso 2: Si falla, probar con chardet como último recurso
        if df is None:
            print("Probando con detección automática de codificación...")
            file_content.seek(0)
            sample = file_content.read(50000)
            file_content.seek(0)
            detected_encoding = chardet.detect(sample)['encoding']
            df = pd.read_csv(file_content, delimiter='|', encoding=detected_encoding, 
                            dtype=str, on_bad_lines='warn')
        
        if df is None:
            st.error("No se pudo leer el archivo con ninguna codificación probada")
            return None

        # Limpieza inicial
        df = df.replace(['', 'nsm', 'NA', 'N/A', 'nan', 'None'], pd.NA)
        df.dropna(axis=1, how='all', inplace=True)
        
        # Convertir columnas específicas
        df = convert_specific_columns(df)

        # Manejo adicional de fechas genéricas
        date_columns = [col for col in df.columns 
                       if 'fecha' in col.lower() 
                       and col not in ['FECHA DE REGISTRO', 'FECHA LIMITE RESPUESTA',
                                      'FECHAHORADICTADO', 'FECHAHORALLEGADA', 'FECHAHORAATENCION']]
        
        for col in date_columns:
            try:
                if 'fechahora' in col.lower():
                    df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
                else:
                    df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce').dt.date
            except Exception as e:
                print(f"No se pudo convertir {col} a fecha: {str(e)}")
                continue

        st.success(f"Archivo cargado correctamente ({len(df)} registros)")
        return df

    except Exception as e:
        st.error(f"Error crítico al procesar el archivo: {str(e)}")
        return None

def apply_filters(df, filters):
    """Aplica filtros al DataFrame"""
    filtered_df = df.copy()
    
    for column, value in filters.items():
        if isinstance(value, tuple):
            start_date, end_date = value
            filtered_df = filtered_df[
                (filtered_df[column].dt.date >= pd.to_datetime(start_date).date()) & 
                (filtered_df[column].dt.date <= pd.to_datetime(end_date).date())
            ]
        elif value != "Todos":
            filtered_df = filtered_df[filtered_df[column].astype(str) == str(value)]
    
    return filtered_df

def create_enhanced_bar_plot(df, column):
    """Crea un gráfico de barras mejorado con Plotly"""
    if df is None or df.empty or column not in df.columns:
        return None
    
    top_values = df[column].value_counts().head(15)
    
    fig = px.bar(
        x=top_values.values,
        y=top_values.index.astype(str),
        orientation='h',
        title=f'Distribución de {column} (Top 15)',
        labels={'x': 'Cantidad de Registros', 'y': column.replace('_', ' ').title()},
        color=top_values.values,
        color_continuous_scale='viridis'
    )
    
    # Agregar números en las barras
    for i, (index, value) in enumerate(top_values.items()):
        fig.add_annotation(
            x=value,
            y=i,
            text=f'{value:,}',
            showarrow=False,
            xshift=15,
            font=dict(color='#2c3e50', size=11),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#bdc3c7',
            borderwidth=1
        )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        font=dict(size=12),
        title_font_size=16
    )
    
    return fig

def calculate_zone_metrics(df, zone_name, selected_year, selected_month):
    """Calcula métricas para una zona específica"""
    if df is None or df.empty:
        return None, None
    
    # Filtrar por zona
    zone_df = df[df['NOMBREADMINISTRATIVO'].str.contains(zone_name, case=False, na=False)]
    
    if zone_df.empty:
        return None, None
    
    # Crear fechas del mes seleccionado
    start_date = pd.Timestamp(year=selected_year, month=selected_month, day=1)
    end_date = pd.Timestamp(year=selected_year, month=selected_month, 
                           day=calendar.monthrange(selected_year, selected_month)[1])
    
    # Obtener todos los días del mes
    days_in_month = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Inicializar métricas por día
    daily_metrics = []
    
    for day in days_in_month:
        day_str = day.strftime('%d/%m/%Y')
        
        # 1. Generación (reportes creados ese día)
        generacion = len(zone_df[
            zone_df['FECHA DE REGISTRO'].dt.date == day.date()
        ])
        
        # 2. Ejecución Corrientes (reportes creados y reparados el mismo mes)
        corrientes = len(zone_df[
            (zone_df['FECHAHORAATENCION'].dt.date == day.date()) &
            (zone_df['ESTADO REPORTE'].str.upper() == 'REPARADO') &
            (zone_df['FECHA DE REGISTRO'].dt.month == selected_month) &
            (zone_df['FECHA DE REGISTRO'].dt.year == selected_year)
        ])
        
        # 3. Ejecución Antiguas (reportes de meses anteriores reparados ese día)
        antiguas = len(zone_df[
            (zone_df['FECHAHORAATENCION'].dt.date == day.date()) &
            (zone_df['ESTADO REPORTE'].str.upper() == 'REPARADO') &
            ((zone_df['FECHA DE REGISTRO'].dt.month != selected_month) |
             (zone_df['FECHA DE REGISTRO'].dt.year != selected_year)) &
            (zone_df['FECHA DE REGISTRO'] < start_date)
        ])
        
        daily_metrics.append({
            'fecha': day_str,
            'day': day.day,
            'generacion': generacion,
            'ejecucion_corrientes': corrientes,
            'ejecucion_antiguas': antiguas
        })
    
    # Crear DataFrame con métricas
    metrics_df = pd.DataFrame(daily_metrics)
    
    # Calcular totales para la tabla resumen
    total_generacion = metrics_df['generacion'].sum()
    total_corrientes = metrics_df['ejecucion_corrientes'].sum()
    total_antiguas = metrics_df['ejecucion_antiguas'].sum()
    
    summary_data = {
        'Generacion': total_generacion,
        'Ejecucion Corrientes': total_corrientes,
        'Ejecucion Antiguas': total_antiguas,
        'Total general': total_generacion + total_corrientes + total_antiguas
    }
    
    return metrics_df, summary_data

def create_zone_line_chart(metrics_df, zone_name):
    """Crea gráfico de líneas para una zona"""
    if metrics_df is None or metrics_df.empty:
        return None
    
    fig = go.Figure()
    
    # Línea de Generación
    fig.add_trace(go.Scatter(
        x=metrics_df['day'],
        y=metrics_df['generacion'],
        mode='lines+markers',
        name='Generación',
        line=dict(color='#3498db', width=3),
        marker=dict(size=8)
    ))
    
    # Línea de Ejecución Corrientes
    fig.add_trace(go.Scatter(
        x=metrics_df['day'],
        y=metrics_df['ejecucion_corrientes'],
        mode='lines+markers',
        name='Ejecución Corrientes',
        line=dict(color='#e67e22', width=3),
        marker=dict(size=8)
    ))
    
    # Línea de Ejecución Antiguas
    fig.add_trace(go.Scatter(
        x=metrics_df['day'],
        y=metrics_df['ejecucion_antiguas'],
        mode='lines+markers',
        name='Ejecución Antiguas',
        line=dict(color='#27ae60', width=3),
        marker=dict(size=8)
    ))
    
    # Agregar anotaciones con los valores
    for _, row in metrics_df.iterrows():
        if row['generacion'] > 0:
            fig.add_annotation(
                x=row['day'], y=row['generacion'],
                text=str(row['generacion']),
                showarrow=False,
                yshift=10,
                font=dict(color='#3498db', size=10)
            )
        if row['ejecucion_corrientes'] > 0:
            fig.add_annotation(
                x=row['day'], y=row['ejecucion_corrientes'],
                text=str(row['ejecucion_corrientes']),
                showarrow=False,
                yshift=10,
                font=dict(color='#e67e22', size=10)
            )
        if row['ejecucion_antiguas'] > 0:
            fig.add_annotation(
                x=row['day'], y=row['ejecucion_antiguas'],
                text=str(row['ejecucion_antiguas']),
                showarrow=False,
                yshift=10,
                font=dict(color='#27ae60', size=10)
            )
    
    fig.update_layout(
        title=f'Generación vs Ejecución - {zone_name}',
        xaxis_title='Día del Mes',
        yaxis_title='Cantidad',
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# Interfaz principal de Streamlit
def main():
    st.title("Sistema de Análisis de Reportes By AVR")
    
    # Sidebar para carga de archivos
    with st.sidebar:
        st.header("Configuración")
        uploaded_file = st.file_uploader("Subir archivo TXT", type=['txt'])
        
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            st.session_state.df = df
        else:
            df = st.session_state.get('df', None)
    
    if df is None:
        st.info("Por favor, sube un archivo TXT para comenzar el análisis.")
        return
    
    # Pestañas principales
    tabs = st.tabs(["Datos", "Gráficos", "Reporte Mensual", "Reporte por Zona"])
    
    # === PESTAÑA DATOS ===
    with tabs[0]:
        st.header("Vista de Datos")
        
        # Filtros en el sidebar
        with st.sidebar:
            st.header("Filtros")
            filters = {}
            
            # Lista de columnas para filtros
            filter_columns = [
                'CÓDIGO', 'COMUNA', 'BARRIO', 'GRUPO DE TRABAJO',
                'TIPO DAÑO', 'FECHA DE REGISTRO', 'FECHAHORAATENCION',
                'ADMINISTRATIVO', 'NOMBREADMINISTRATIVO', 'ESTADO REPORTE',
                'TIPO ASIGNACIÓN'
            ]
            
            available_columns = [col for col in filter_columns if col in df.columns]
            
            for col in available_columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    st.subheader(f"Filtro de {col}")
                    min_date = df[col].min().date() if not df[col].isna().all() else None
                    max_date = df[col].max().date() if not df[col].isna().all() else None
                    
                    if min_date and max_date:
                        date_range = st.date_input(
                            f"Rango de {col}",
                            value=(min_date, max_date),
                            min_value=min_date,
                            max_value=max_date,
                            key=f"date_{col}"
                        )
                        
                        if len(date_range) == 2:
                            filters[col] = date_range
                else:
                    unique_values = ['Todos'] + sorted(df[col].dropna().unique().astype(str).tolist())
                    selected_value = st.selectbox(f"Filtrar por {col}", unique_values, key=f"select_{col}")
                    if selected_value != "Todos":
                        filters[col] = selected_value
        
        # Aplicar filtros
        filtered_df = apply_filters(df, filters)
        
        st.info(f"Mostrando {len(filtered_df)} de {len(df)} registros")
        
        # Mostrar resumen
        if not filtered_df.empty:
            col1, col2, col3 = st.columns(3)
            
            main_cat_cols = ['ESTADO REPORTE', 'NOMBREADMINISTRATIVO', 'COMUNA']
            
            for i, col in enumerate(main_cat_cols):
                if col in filtered_df.columns:
                    with [col1, col2, col3][i]:
                        st.subheader(f"{col}")
                        counts = filtered_df[col].value_counts().head(5)
                        for value, count in counts.items():
                            if pd.notna(value):
                                st.write(f"**{value}:** {count}")
        
        # Mostrar datos
        st.dataframe(filtered_df.head(100), use_container_width=True)
    
    # === PESTAÑA GRÁFICOS ===
    with tabs[1]:
        st.header("Análisis Gráfico")
        
        if not df.empty:
            # Filtrar el DataFrame actual
            current_df = apply_filters(df, st.session_state.get('current_filters', {}))
            
            # Columnas excluidas para gráficos
            excluded_columns = [
                'CÓDIGO', 'PRIORIDAD', 'PUNTO DE REFERENCIA', 'LUMINARIAS', 'TELEFONO',
                'NOMBRE LABOR', 'TIEMPO DE ATENCIÓN [HRS]', 'FECHA LIMITE RESPUESTA',
                'DIRECCIÓN', 'CODIGOADMINISTRATIVO', 'X', 'Y', 'CODIGOACTUAL',
                'CODIGOAPOYO', 'PINTADOAPOYO', 'DESCRIPCION', 'OBSERVACION',
                'FECHAHORADICTADO', 'FECHAHORALLEGADA', 'TIEMPO DESDE DICTADO [HRS]',
                'LATITUD', 'LONGITUD'
            ]
            
            cat_cols = [
                col for col in current_df.columns 
                if col not in excluded_columns 
                and pd.api.types.is_string_dtype(current_df[col])
            ]
            
            if cat_cols:
                selected_column = st.selectbox("Seleccionar columna para graficar", cat_cols)
                
                fig = create_enhanced_bar_plot(current_df, selected_column)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay columnas categóricas disponibles para graficar.")
    
    # === PESTAÑA REPORTE MENSUAL ===
    with tabs[2]:
        st.header("Reporte Mensual")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            # Obtener años disponibles
            if 'FECHA DE REGISTRO' in df.columns:
                years = sorted(df['FECHA DE REGISTRO'].dt.year.dropna().unique().astype(int).tolist())
                selected_year = st.selectbox("Año", years)
            else:
                st.error("No se encontró la columna 'FECHA DE REGISTRO'")
                return
        
        with col2:
            months = {i: calendar.month_name[i] for i in range(1, 13)}
            selected_month = st.selectbox("Mes", list(months.keys()), format_func=lambda x: months[x])
        
        with col3:
            if st.button("Generar Reporte Mensual"):
                # Generar reporte
                days_in_month = calendar.monthrange(selected_year, selected_month)[1]
                
                report_data = []
                for day in range(1, days_in_month + 1):
                    current_date = pd.Timestamp(year=selected_year, month=selected_month, day=day)
                    
                    # Reportes registrados
                    registered = len(df[df['FECHA DE REGISTRO'].dt.date == current_date.date()])
                    
                    # Reportes reparados
                    repaired = 0
                    if 'FECHAHORAATENCION' in df.columns:
                        repaired = len(df[df['FECHAHORAATENCION'].dt.date == current_date.date()])
                    
                    report_data.append({
                        'Día': day,
                        'Reportes Registrados': registered,
                        'Reportes Reparados': repaired
                    })
                
                report_df = pd.DataFrame(report_data)
                
                # Agregar fila de totales
                totals = {
                    'Día': 'TOTAL',
                    'Reportes Registrados': report_df['Reportes Registrados'].sum(),
                    'Reportes Reparados': report_df['Reportes Reparados'].sum()
                }
                report_df = pd.concat([report_df, pd.DataFrame([totals])], ignore_index=True)
                
                st.dataframe(report_df, use_container_width=True)
    
    # === PESTAÑA REPORTE POR ZONA ===
    with tabs[3]:
        st.header("Reporte por Zona")
        
        # Controles de selección
        col1, col2 = st.columns(2)
        
        with col1:
            if 'FECHA DE REGISTRO' in df.columns:
                years = sorted(df['FECHA DE REGISTRO'].dt.year.dropna().unique().astype(int).tolist())
                zone_year = st.selectbox("Año", years, key="zone_year")
            else:
                st.error("No se encontró la columna 'FECHA DE REGISTRO'")
                return
        
        with col2:
            months = {i: calendar.month_name[i] for i in range(1, 13)}
            zone_month = st.selectbox("Mes", list(months.keys()), 
                                    format_func=lambda x: months[x], key="zone_month")
        
        if st.button("Generar Reporte por Zona"):
            # Verificar que existan las zonas
            if 'NOMBREADMINISTRATIVO' not in df.columns:
                st.error("No se encontró la columna 'NOMBREADMINISTRATIVO'")
                return
            
            # Crear dos columnas para las zonas
            col1, col2 = st.columns(2)
            
            # ZONA SUR
            with col1:
                st.subheader("ZONA SUR")
                
                sur_metrics, sur_summary = calculate_zone_metrics(df, "SUR", zone_year, zone_month)
                
                if sur_summary:
                    # Tabla resumen
                    sur_df = pd.DataFrame(list(sur_summary.items()), columns=['Métrica', 'Valor'])
                    st.dataframe(sur_df, use_container_width=True, hide_index=True)
                    
                    # Gráfico de líneas
                    sur_chart = create_zone_line_chart(sur_metrics, "ZONA SUR")
                    if sur_chart:
                        st.plotly_chart(sur_chart, use_container_width=True)
                else:
                    st.warning("No se encontraron datos para la Zona Sur")
            
            # ZONA CENTRO
            with col2:
                st.subheader("ZONA CENTRO")
                
                centro_metrics, centro_summary = calculate_zone_metrics(df, "CENTRO", zone_year, zone_month)
                
                if centro_summary:
                    # Tabla resumen
                    centro_df = pd.DataFrame(list(centro_summary.items()), columns=['Métrica', 'Valor'])
                    st.dataframe(centro_df, use_container_width=True, hide_index=True)
                    
                    # Gráfico de líneas
                    centro_chart = create_zone_line_chart(centro_metrics, "ZONA CENTRO")
                    if centro_chart:
                        st.plotly_chart(centro_chart, use_container_width=True)
                else:
                    st.warning("No se encontraron datos para la Zona Centro")

if __name__ == "__main__":
    # Inicializar session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    
    main()