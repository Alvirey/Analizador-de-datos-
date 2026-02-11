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

def classify_column_type(df, column):
    """Clasifica el tipo de columna para determinar el tipo de filtro"""
    if column not in df.columns:
        return None
    
    col_data = df[column].dropna()
    
    if len(col_data) == 0:
        return 'empty'
    
    # Verificar si es fecha/datetime
    if pd.api.types.is_datetime64_any_dtype(col_data):
        return 'datetime'
    
    # Verificar si es numérico
    if pd.api.types.is_numeric_dtype(col_data):
        unique_values = col_data.nunique()
        # Si tiene muchos valores únicos, usar slider; si pocos, usar selectbox
        if unique_values > 20:
            return 'numeric_slider'
        else:
            return 'numeric_select'
    
    # Para columnas de texto/string
    if pd.api.types.is_string_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
        unique_values = col_data.nunique()
        # Si tiene muchos valores únicos, usar multiselect; si pocos, usar selectbox
        if unique_values > 50:
            return 'text_multiselect'
        elif unique_values > 10:
            return 'text_select_multi'
        else:
            return 'text_select_single'
    
    return 'other'

def create_dynamic_filters(df):
    """Crea filtros dinámicos según el tipo de datos de cada columna"""
    with st.sidebar:
        st.header("🔍 Filtros Dinámicos")
        
        filters = {}
        
        # Permitir al usuario seleccionar qué columnas filtrar
        all_columns = [col for col in df.columns if not col.startswith('Unnamed')]
        
        # Excluir columnas que no son útiles para filtrar
        excluded_for_filters = [
            'DESCRIPCION', 'OBSERVACION', 'PUNTO DE REFERENCIA', 
            'DIRECCIÓN', 'NOMBRE LABOR'
        ]
        
        filterable_columns = [col for col in all_columns if col not in excluded_for_filters]
        
        # Selectbox para elegir columnas a filtrar
        st.subheader("Seleccionar columnas para filtrar:")
        selected_columns = st.multiselect(
            "Columnas disponibles:",
            filterable_columns,
            default=[col for col in [] if col in filterable_columns][:4],
            key="column_selector"
        )
        
        st.divider()
        
        # Crear filtros dinámicos para cada columna seleccionada
        for column in selected_columns:
            column_type = classify_column_type(df, column)
            
            st.subheader(f"📋 {column}")
            
            if column_type == 'datetime':
                # Filtro de fecha con date_input
                col_data = df[column].dropna()
                if len(col_data) > 0:
                    min_date = col_data.min().date()
                    max_date = col_data.max().date()
                    
                    date_range = st.date_input(
                        f"Rango de fechas:",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                        key=f"date_{column}"
                    )
                    
                    if len(date_range) == 2:
                        filters[column] = {
                            'type': 'date_range',
                            'value': date_range
                        }
            
            elif column_type == 'numeric_slider':
                # Slider para columnas numéricas con muchos valores
                col_data = df[column].dropna()
                if len(col_data) > 0:
                    min_val = float(col_data.min())
                    max_val = float(col_data.max())
                    
                    if min_val != max_val:
                        slider_range = st.slider(
                            f"Rango de valores:",
                            min_value=min_val,
                            max_value=max_val,
                            value=(min_val, max_val),
                            key=f"slider_{column}"
                        )
                        filters[column] = {
                            'type': 'numeric_range',
                            'value': slider_range
                        }
            
            elif column_type in ['numeric_select', 'text_select_single']:
                # Selectbox para columnas con pocos valores únicos
                unique_values = sorted(df[column].dropna().unique().astype(str).tolist())
                
                selected_value = st.selectbox(
                    f"Seleccionar valor:",
                    ['Todos'] + unique_values,
                    key=f"select_{column}"
                )
                
                if selected_value != 'Todos':
                    filters[column] = {
                        'type': 'single_select',
                        'value': selected_value
                    }
            
            elif column_type in ['text_select_multi', 'text_multiselect']:
                # Multiselect para columnas de texto con varios valores
                unique_values = sorted(df[column].dropna().unique().astype(str).tolist())
                
                # Opción para seleccionar todos
                select_all = st.checkbox(f"Seleccionar todos", key=f"all_{column}")
                
                if select_all:
                    selected_values = unique_values
                else:
                    selected_values = st.multiselect(
                        f"Seleccionar valores:",
                        unique_values,
                        key=f"multi_{column}"
                    )
                
                if selected_values:
                    filters[column] = {
                        'type': 'multi_select',
                        'value': selected_values
                    }
            
            st.divider()
        
        # Botón para limpiar todos los filtros
        if st.button("🗑️ Limpiar todos los filtros"):
            st.rerun()
        
        # Mostrar resumen de filtros aplicados
        if filters:
            st.subheader("📊 Filtros activos:")
            for col, filter_info in filters.items():
                if filter_info['type'] == 'date_range':
                    st.write(f"• **{col}**: {filter_info['value'][0]} - {filter_info['value'][1]}")
                elif filter_info['type'] == 'numeric_range':
                    st.write(f"• **{col}**: {filter_info['value'][0]:.2f} - {filter_info['value'][1]:.2f}")
                elif filter_info['type'] == 'single_select':
                    st.write(f"• **{col}**: {filter_info['value']}")
                elif filter_info['type'] == 'multi_select':
                    if len(filter_info['value']) <= 3:
                        st.write(f"• **{col}**: {', '.join(filter_info['value'])}")
                    else:
                        st.write(f"• **{col}**: {len(filter_info['value'])} valores seleccionados")
    
    return filters

def apply_dynamic_filters(df, filters):
    """Aplica los filtros dinámicos al DataFrame"""
    filtered_df = df.copy()
    
    for column, filter_info in filters.items():
        if column not in filtered_df.columns:
            continue
        
        filter_type = filter_info['type']
        filter_value = filter_info['value']
        
        if filter_type == 'date_range' and len(filter_value) == 2:
            start_date, end_date = filter_value
            filtered_df = filtered_df[
                (filtered_df[column].dt.date >= start_date) & 
                (filtered_df[column].dt.date <= end_date)
            ]
        
        elif filter_type == 'numeric_range':
            min_val, max_val = filter_value
            filtered_df = filtered_df[
                (filtered_df[column] >= min_val) & 
                (filtered_df[column] <= max_val)
            ]
        
        elif filter_type == 'single_select':
            filtered_df = filtered_df[filtered_df[column].astype(str) == str(filter_value)]
        
        elif filter_type == 'multi_select':
            filtered_df = filtered_df[filtered_df[column].astype(str).isin([str(v) for v in filter_value])]
    
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
    st.title("🏢 Sistema de Análisis de Reportes By AVR")
    st.markdown("---")
    
    # Sidebar para carga de archivos
    with st.sidebar:
        st.header("📁 Configuración")
        uploaded_file = st.file_uploader("Subir archivo TXT", type=['txt'])
        
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            st.session_state.df = df
        else:
            df = st.session_state.get('df', None)
    
    if df is None:
        st.info("📤 Por favor, sube un archivo TXT para comenzar el análisis.")
        return
    
    # CREAR FILTROS DINÁMICOS
    filters = create_dynamic_filters(df)
    
    # Aplicar filtros dinámicos al DataFrame
    filtered_df = apply_dynamic_filters(df, filters)
    
    # Mostrar estadísticas de filtrado
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📊 Total de registros",
            value=f"{len(df):,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="🔍 Registros filtrados",
            value=f"{len(filtered_df):,}",
            delta=f"{len(filtered_df) - len(df):,}" if len(filtered_df) != len(df) else None
        )
    
    with col3:
        percentage = (len(filtered_df) / len(df)) * 100 if len(df) > 0 else 0
        st.metric(
            label="📈 Porcentaje mostrado",
            value=f"{percentage:.1f}%"
        )
    
    st.markdown("---")
    
    # Pestañas principales
    tabs = st.tabs(["📋 Tabla Filtrada", "📊 Gráficos", "📅 Reporte Mensual", "🗺️ Reporte por Zona"])
    
    # === PESTAÑA TABLA FILTRADA ===
    with tabs[0]:
        st.header("📋 Tabla de Datos Filtrados")
        
        if not filtered_df.empty:
            # Mostrar resumen por categorías principales
            st.subheader("📈 Resumen de datos filtrados")
            
            col1, col2, col3 = st.columns(3)
            
            main_cat_cols = ['ESTADO REPORTE', 'NOMBREADMINISTRATIVO', 'COMUNA']
            
            for i, col in enumerate(main_cat_cols):
                if col in filtered_df.columns:
                    with [col1, col2, col3][i]:
                        st.write(f"**{col}**")
                        counts = filtered_df[col].value_counts().head(5)
                        for value, count in counts.items():
                            if pd.notna(value):
                                st.write(f"• {value}: **{count:,}**")
            
            st.markdown("---")
            
            # Controles para la tabla
            col1, col2 = st.columns(2)
            
            with col1:
                # Selectbox para elegir cuántas filas mostrar
                rows_to_show = st.selectbox(
                    "Filas a mostrar:",
                    [50, 100, 200, 500, 1000, "Todas"],
                    index=1
                )
            
            with col2:
                # Checkbox para mostrar solo columnas con datos
                hide_empty_cols = st.checkbox("Ocultar columnas vacías", value=True)
            
            # Preparar DataFrame para mostrar
            display_df = filtered_df.copy()
            
            if hide_empty_cols:
                # Remover columnas que están completamente vacías
                display_df = display_df.dropna(axis=1, how='all')
            
            # Aplicar límite de filas
            if rows_to_show != "Todas":
                display_df = display_df.head(int(rows_to_show))
            
            # Mostrar tabla con opciones de descarga
            st.subheader(f"📊 Mostrando {len(display_df)} de {len(filtered_df)} registros")
            
            # Opción para descargar datos filtrados
            if not filtered_df.empty:
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar datos filtrados (CSV)",
                    data=csv,
                    file_name=f"datos_filtrados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # Mostrar la tabla
            st.dataframe(
                display_df, 
                use_container_width=True,
                height=600
            )
        
        else:
            st.warning("⚠️ No hay datos que coincidan con los filtros aplicados.")
            st.info("💡 Intenta ajustar o limpiar algunos filtros para ver más resultados.")
    
    # === PESTAÑA GRÁFICOS ===
    with tabs[1]:
        st.header("📊 Análisis Gráfico")
        
        if not filtered_df.empty:
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
                col for col in filtered_df.columns 
                if col not in excluded_columns 
                and pd.api.types.is_string_dtype(filtered_df[col])
                and filtered_df[col].nunique() > 1
                and filtered_df[col].nunique() <= 50
            ]
            
            if cat_cols:
                selected_column = st.selectbox(
                    "📊 Seleccionar columna para graficar:", 
                    cat_cols
                )
                
                fig = create_enhanced_bar_plot(filtered_df, selected_column)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar estadísticas adicionales
                    st.subheader("📈 Estadísticas de la columna seleccionada")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Valores únicos", filtered_df[selected_column].nunique())
                    with col2:
                        st.metric("Valores no nulos", filtered_df[selected_column].notna().sum())
                    with col3:
                        most_common = filtered_df[selected_column].mode().iloc[0] if not filtered_df[selected_column].empty else "N/A"
                        st.metric("Valor más común", str(most_common))
            else:
                st.warning("⚠️ No hay columnas categóricas adecuadas para graficar con los filtros actuales.")
                st.info("💡 Intenta ajustar los filtros para incluir más variedad de datos.")
        else:
            st.warning("⚠️ No hay datos para graficar con los filtros aplicados.")
    
    # === PESTAÑA REPORTE MENSUAL ===
    with tabs[2]:
        st.header("📅 Reporte Mensual")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            # Obtener años disponibles de los datos filtrados
            if 'FECHA DE REGISTRO' in filtered_df.columns and not filtered_df.empty:
                years = sorted(filtered_df['FECHA DE REGISTRO'].dt.year.dropna().unique().astype(int).tolist())
                if years:
                    selected_year = st.selectbox("📅 Año", years)
                else:
                    st.warning("No hay datos de fecha disponibles")
                    return
            else:
                st.error("No se encontró la columna 'FECHA DE REGISTRO' o no hay datos filtrados")
                return
        
        with col2:
            months = {i: calendar.month_name[i] for i in range(1, 13)}
            selected_month = st.selectbox("📅 Mes", list(months.keys()), format_func=lambda x: months[x])
        
        with col3:
            if st.button("📊 Generar Reporte Mensual"):
                # Generar reporte usando datos filtrados
                days_in_month = calendar.monthrange(selected_year, selected_month)[1]
                
                report_data = []
                for day in range(1, days_in_month + 1):
                    current_date = pd.Timestamp(year=selected_year, month=selected_month, day=day)
                    
                    # Reportes registrados
                    registered = len(filtered_df[filtered_df['FECHA DE REGISTRO'].dt.date == current_date.date()])
                    
                    # Reportes reparados
                    repaired = 0
                    if 'FECHAHORAATENCION' in filtered_df.columns:
                        repaired = len(filtered_df[
                            (filtered_df['FECHAHORAATENCION'].dt.date == current_date.date()) &
                            (filtered_df['ESTADO REPORTE'].str.upper() == 'REPARADO')
                        ])
                    
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
                
                # Mostrar el reporte
                st.subheader(f"📊 Reporte de {calendar.month_name[selected_month]} {selected_year}")
                st.dataframe(report_df, use_container_width=True, height=600)
                
                # Crear gráfico del reporte mensual
                if len(report_df) > 1:  # Excluir la fila de totales para el gráfico
                    chart_df = report_df[:-1]  # Sin la fila TOTAL
                    
                    fig = go.Figure()
                    
                    # Línea de registros
                    fig.add_trace(go.Scatter(
                        x=chart_df['Día'],
                        y=chart_df['Reportes Registrados'],
                        mode='lines+markers',
                        name='Registrados',
                        line=dict(color='#3498db', width=3),
                        marker=dict(size=8)
                    ))
                    
                    # Línea de reparados
                    fig.add_trace(go.Scatter(
                        x=chart_df['Día'],
                        y=chart_df['Reportes Reparados'],
                        mode='lines+markers',
                        name='Reparados',
                        line=dict(color='#e67e22', width=3),
                        marker=dict(size=8)
                    ))
                    
                    fig.update_layout(
                        title=f'📈 Reportes por Día - {calendar.month_name[selected_month]} {selected_year}',
                        xaxis_title='Día del Mes',
                        yaxis_title='Cantidad de Reportes',
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Opción para descargar el reporte
                csv = report_df.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Reporte Mensual (CSV)",
                    data=csv,
                    file_name=f"reporte_mensual_{selected_year}_{selected_month:02d}.csv",
                    mime="text/csv"
                )
    
    # === PESTAÑA REPORTE POR ZONA ===
    with tabs[3]:
        st.header("🗺️ Reporte por Zona")
        
        # Controles de selección
        col1, col2 = st.columns(2)
        
        with col1:
            if 'FECHA DE REGISTRO' in filtered_df.columns and not filtered_df.empty:
                years = sorted(filtered_df['FECHA DE REGISTRO'].dt.year.dropna().unique().astype(int).tolist())
                if years:
                    zone_year = st.selectbox("📅 Año", years, key="zone_year")
                else:
                    st.warning("No hay datos de fecha disponibles")
                    return
            else:
                st.error("No se encontró la columna 'FECHA DE REGISTRO' o no hay datos filtrados")
                return
        
        with col2:
            months = {i: calendar.month_name[i] for i in range(1, 13)}
            zone_month = st.selectbox("📅 Mes", list(months.keys()), 
                                    format_func=lambda x: months[x], key="zone_month")
        
        if st.button("📊 Generar Reporte por Zona"):
            # Verificar que existan las zonas
            if 'NOMBREADMINISTRATIVO' not in filtered_df.columns:
                st.error("No se encontró la columna 'NOMBREADMINISTRATIVO'")
                return
            
            # Mostrar las zonas disponibles en los datos filtrados
            available_zones = filtered_df['NOMBREADMINISTRATIVO'].dropna().unique()
            sur_zones = [zone for zone in available_zones if 'SUR' in str(zone).upper()]
            centro_zones = [zone for zone in available_zones if 'CENTRO' in str(zone).upper()]
            
            st.info(f"🔍 Zonas encontradas: SUR ({len(sur_zones)} zonas), CENTRO ({len(centro_zones)} zonas)")
            
            # Crear dos columnas para las zonas
            col1, col2 = st.columns(2)
            
            # ZONA SUR - usando datos filtrados
            with col1:
                st.subheader("🌅 ZONA SUR")
                
                sur_metrics, sur_summary = calculate_zone_metrics(filtered_df, "SUR", zone_year, zone_month)
                
                if sur_summary:
                    # Tabla resumen
                    sur_df = pd.DataFrame(list(sur_summary.items()), columns=['Métrica', 'Valor'])
                    st.dataframe(sur_df, use_container_width=True, hide_index=True)
                    
                    # Gráfico de líneas
                    sur_chart = create_zone_line_chart(sur_metrics, "ZONA SUR")
                    if sur_chart:
                        st.plotly_chart(sur_chart, use_container_width=True)
                    
                    # Opción para descargar datos de la zona sur
                    if sur_metrics is not None and not sur_metrics.empty:
                        csv_sur = sur_metrics.to_csv(index=False)
                        st.download_button(
                            label="📥 Descargar datos Zona Sur",
                            data=csv_sur,
                            file_name=f"zona_sur_{zone_year}_{zone_month:02d}.csv",
                            mime="text/csv",
                            key="download_sur"
                        )
                else:
                    st.warning("⚠️ No se encontraron datos para la Zona Sur con los filtros aplicados")
            
            # ZONA CENTRO - usando datos filtrados
            with col2:
                st.subheader("🏙️ ZONA CENTRO")
                
                centro_metrics, centro_summary = calculate_zone_metrics(filtered_df, "CENTRO", zone_year, zone_month)
                
                if centro_summary:
                    # Tabla resumen
                    centro_df = pd.DataFrame(list(centro_summary.items()), columns=['Métrica', 'Valor'])
                    st.dataframe(centro_df, use_container_width=True, hide_index=True)
                    
                    # Gráfico de líneas
                    centro_chart = create_zone_line_chart(centro_metrics, "ZONA CENTRO")
                    if centro_chart:
                        st.plotly_chart(centro_chart, use_container_width=True)
                    
                    # Opción para descargar datos de la zona centro
                    if centro_metrics is not None and not centro_metrics.empty:
                        csv_centro = centro_metrics.to_csv(index=False)
                        st.download_button(
                            label="📥 Descargar datos Zona Centro",
                            data=csv_centro,
                            file_name=f"zona_centro_{zone_year}_{zone_month:02d}.csv",
                            mime="text/csv",
                            key="download_centro"
                        )
                else:
                    st.warning("⚠️ No se encontraron datos para la Zona Centro con los filtros aplicados")
            
            # Comparativa entre zonas
            if sur_summary and centro_summary:
                st.markdown("---")
                st.subheader("📊 Comparativa entre Zonas")
                
                # Crear DataFrame comparativo
                comparison_data = {
                    'Métrica': list(sur_summary.keys()),
                    'Zona Sur': list(sur_summary.values()),
                    'Zona Centro': list(centro_summary.values())
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df['Diferencia'] = comparison_df['Zona Sur'] - comparison_df['Zona Centro']
                comparison_df['% Sur vs Centro'] = (
                    (comparison_df['Zona Sur'] / comparison_df['Zona Centro'] * 100).round(1)
                    .fillna(0).astype(str) + '%'
                )
                
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                # Gráfico comparativo
                metrics_for_chart = ['Generacion', 'Ejecucion Corrientes', 'Ejecucion Antiguas']
                
                fig_comp = go.Figure(data=[
                    go.Bar(name='Zona Sur', x=metrics_for_chart, 
                           y=[sur_summary[m] for m in metrics_for_chart], 
                           marker_color='#3498db'),
                    go.Bar(name='Zona Centro', x=metrics_for_chart, 
                           y=[centro_summary[m] for m in metrics_for_chart],
                           marker_color='#e67e22')
                ])
                
                fig_comp.update_layout(
                    title='📊 Comparación por Métricas - Sur vs Centro',
                    xaxis_title='Métricas',
                    yaxis_title='Cantidad',
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig_comp, use_container_width=True)

if __name__ == "__main__":
    # Inicializar session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'current_filters' not in st.session_state:
        st.session_state.current_filters = {}
    
    main()