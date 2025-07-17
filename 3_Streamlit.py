import streamlit as st
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


# Título de  la página
st.set_page_config(layout="centered",
    page_title="Energía Zonas no Interconectadas",
    page_icon="💡"
)

# Columnas

t1, t2 = st.columns([0.3,0.7]) 

t1.image('zonas_no_interconectadas.webp', width = 300)
t2.title("Estado de Prestación de Servicios en Zonas no Interconectadas de Colombia")
t2.markdown(" Daniel | César | Juan Diego ")

# # Using object notation
# add_selectbox = st.sidebar.selectbox(
#     "How would you like to be contacted?",
#     ("Email", "Home phone", "Mobile phone")
# )

# # Using "with" notation
# with st.sidebar:
#     add_radio = st.radio(
#         "Choose a shipping method",
#         ("Standard (5-15 days)", "Express (2-5 days)")
#     )

# Datos

db_url = "postgresql://postgres:Entropia18*@localhost:5432/EnergiasZonasNoInterconectadasCol"
engine = create_engine(db_url)  

# Función para obtener ID's de los nombres del sitio

def consulta_ids (tab_id, column_id, column_name, name):
    with engine.connect() as connection: 
        consulta_id = connection.execute(text(f'SELECT "{column_id}" FROM {tab_id} WHERE "{column_name}" = :name'), {"name": name})
    row_id = consulta_id.fetchone()
    id = row_id[0] 
    return(id)

# Secciones

steps=st.tabs(["Pestaña 1"])

with steps[0]:
    energias_df = pd.read_sql('SELECT * FROM energias.servicios_detalle;' , engine)
    energias_df = energias_df.sort_values(by='Fecha Demanda Máxima', ascending=True)

    energias_df['Factor de Potencia'] = energias_df['Energía Activa [kWh]'] / np.sqrt((energias_df['Energía Activa [kWh]']**2) + (energias_df['Energía Reactiva [kVArh]']**2))
    
    energias_df1 = energias_df[['Departamento', 'Municipio', 'Centro Poblado', 'Energía Activa [kWh]', 'Energía Reactiva [kVArh]', 
                              'Factor de Potencia', 'Potencia Máxima [kW]', 'Fecha Demanda Máxima', 'Promedio Diario [h]']]
    st.dataframe(energias_df1)

    departamento = st.selectbox('Escoge el departamento de interés', energias_df['Departamento'].sort_values(ascending=True).drop_duplicates())
    id_departamento =  consulta_ids ('energias.servicios_detalle', 'Código Departamento', 'Departamento', departamento)
    municipio = st.selectbox('Escoge el municipio de interés', energias_df[energias_df['Código Departamento'] == id_departamento]['Municipio'].sort_values(ascending=True).drop_duplicates())
    id_municipio = consulta_ids('energias.servicios_detalle', 'Código Municipio', 'Municipio', municipio)
    centro_poblado = st.selectbox('Escoge el centro poblado de interés', energias_df[energias_df['Código Municipio'] == id_municipio]['Centro Poblado'].sort_values(ascending=True).drop_duplicates())
    id_centro_poblado = consulta_ids('energias.servicios_detalle', 'Código Centro Poblado', 'Centro Poblado', centro_poblado)


    df_centro_poblado = energias_df[energias_df['Código Centro Poblado'] == id_centro_poblado]

    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x = df_centro_poblado['Fecha Demanda Máxima'], y = df_centro_poblado['Potencia Máxima [kW]'], mode='lines', name = 'Potencia Máxima [kW]', yaxis='y1'))
    # fig.add_trace(go.Scatter(x = df_centro_poblado['Fecha Demanda Máxima'], y = df_centro_poblado['Promedio Diario [h]'], mode='lines', name = 'Promedio Diario [h]', yaxis='y2'))
    # fig.update_layout(title='Potencia Máxima y Promedio Diario por Hora vs Fecha Demanda Máxima', xaxis_title='Fecha', 
    #                   yaxis=dict(title='Potencia Máxima [kW]', side='left'), 
    #                   yaxis2=dict(title='Promedio Diario [h]', overlaying='y', side='right'), 
    #                   hovermode='x unified')
    # st.plotly_chart(fig, use_container_with=True)


    fig_pot_fech = px.line(df_centro_poblado, x = 'Fecha Demanda Máxima', y = 'Potencia Máxima [kW]', title='Potencia Máxima Vs Fecha Demanda Máxima')
    st.plotly_chart(fig_pot_fech, use_container_width=True) 

    fig_prom_fech = px.line(df_centro_poblado, x = 'Fecha Demanda Máxima', y = 'Promedio Diario [h]', title='Promedio Diario Vs Fecha Demanda Máxima')
    st.plotly_chart(fig_prom_fech, use_container_width=True) 

