import streamlit as st
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from scipy.stats import gaussian_kde
import pydeck as pdk
st.set_page_config(layout="centered",
    page_title="Energía Zonas no Interconectadas",
    page_icon="💡"
)
t1,t2=st.columns(
    [0.3,0.7]
    ) 
# t1.image(
#     'zonas_no_interconectadas.webp', width = 300
#     )
t2.title(
    "Estado de Prestación de Servicios en Zonas no Interconectadas de Colombia"
    )
engine=create_engine(
    "postgresql+psycopg2://postgres:123456@localhost:5432/zniBasedatos?client_encoding=WIN1252"
    )
energias_df=pd.read_sql(
    'SELECT * FROM energias.servicios_detalle;' , engine
    )
def consulta_ids (tab_id,column_id,column_name,name):
    with engine.connect() as connection: 
        consulta_id=connection.execute(
            text(
                f'''SELECT "{column_id}" FROM {tab_id} WHERE "{column_name}"=:name'''
                ),{"name":name}
        )
    row_id=consulta_id.fetchone()
    id=row_id[0] 
    return(id)
st.sidebar.title("Opciones")
opcion = st.sidebar.selectbox(
    "Selecciona una opción:",
    ["Análisis general", "Análisis por centro poblado"]
)
if opcion=="Análisis general":
    steps1=st.tabs(
        ['General', 'Análisis descriptivo', 'Modelo']
    )
    with steps1[0]:
        st.markdown('## Visión general de los datos')
        st.dataframe(energias_df)

        # Selección de variable
        var_map = st.selectbox(
            'Seleccione la variable a visualizar',
            [
                'Promedio Diario [h]', 'Energía Activa [kWh]', 'Energía Reactiva [kVArh]',
                'Potencia Máxima [kW]', 'Total Personas en Hogares Particulares',
                'Personas en NBI [%]', 'Componente Servicios [%]'
            ]
        )

        # Elegir método de agregación
        metodo_agregacion = st.radio(
            "Método de agregación por centro poblado:",
            ['Media', 'Mediana'],
            horizontal=True
        )

        # Agregación por centro poblado con lat/lon únicos por centro poblado + municipio + departamento
        agrupadores = ['Centro Poblado', 'Municipio', 'Departamento']

        if metodo_agregacion == 'Media':
            energias_map = energias_df.groupby(agrupadores).agg({
                var_map: 'mean',
                'Latitud': 'mean',
                'Longitud': 'mean'
            }).reset_index()
        else:
            energias_map = energias_df.groupby(agrupadores).agg({
                var_map: 'median',
                'Latitud': 'mean',
                'Longitud': 'mean'
            }).reset_index()

        # Limpieza de datos
        energias_map = energias_map.dropna(subset=['Latitud', 'Longitud', var_map])
        energias_map['elevation_norm'] = energias_map[var_map] / energias_map[var_map].max() * 10000

        # Escala de color dinámica (verde = bajo, rojo = alto)
        energias_map['color_r'] = (255 * energias_map[var_map] / energias_map[var_map].max()).astype(int)
        energias_map['color_g'] = (255 * (1 - energias_map[var_map] / energias_map[var_map].max())).astype(int)
        energias_map['color_b'] = 120
        energias_map['color'] = energias_map[['color_r', 'color_g', 'color_b']].values.tolist()

        # Capa del mapa
        column_layer = pdk.Layer(
            'ColumnLayer',
            data=energias_map,
            get_position='[Longitud, Latitud]',
            get_elevation='elevation_norm',
            elevation_scale=100,
            radius=2000,
            get_fill_color='color',
            pickable=True,
            auto_highlight=True,
        )

        view_state = pdk.ViewState(
            latitude=energias_map['Latitud'].mean(),
            longitude=energias_map['Longitud'].mean(),
            zoom=6,
            pitch=45,
            bearing=0    
        )

        # Mapa final
        st.pydeck_chart(pdk.Deck(
            map_style="light",
            initial_view_state=view_state,
            layers=[column_layer],
            tooltip = {
                "html": """
                    <b>Centro Poblado:</b> {Centro Poblado}<br>
                    <b>Municipio:</b> {Municipio}<br>
                    <b>%s:</b> {%s}
                """ % (var_map, var_map),
                "style": {"backgroundColor": "steelblue", "color": "white"}
            }

        ))

    with steps1[1]:
        analisis_g=st.selectbox('Medidas individuales',[
            'Descripción zonas no interconectadas',
            'Descripción CNPV',
            'Descripción NBI'
        ])
        if analisis_g=='Descripción zonas no interconectadas':
            zonas_no_int=st.selectbox('Zonas no interconectadas',[
                'Departamentos con más zonas no interconectadas',
                'Municipios con más zonas no interconectadas'
                ])
            if zonas_no_int=='Departamentos con más zonas no interconectadas':
                loc_dep=energias_df.groupby('Departamento')['Centro Poblado']\
                    .nunique().sort_values(ascending=False).reset_index()
                loc_dep=loc_dep.rename(
                    columns={'Centro Poblado': 'Cantidad Centros Poblados'}
                    )
                st.dataframe(loc_dep)
                pie_dep=px.pie(
                    loc_dep,names='Departamento',values='Cantidad Centros Poblados'
                    )
                st.plotly_chart(
                    pie_dep, use_container_width=True
                    )
            elif zonas_no_int=='Municipios con más zonas no interconectadas':
                loc_mun=energias_df.groupby(['Departamento','Municipio'])['Centro Poblado']\
                    .nunique().sort_values(ascending=False).reset_index()
                loc_mun=loc_mun.rename(
                    columns={'Centro Poblado': 'Cantidad Centros Poblados'}
                )
                st.dataframe(loc_mun)
                pie_mun=px.pie(
                    loc_mun,names='Municipio',values='Cantidad Centros Poblados'
                    )
                st.plotly_chart(
                    pie_mun, use_container_width=True
                    )
        elif analisis_g=='Descripción CNPV':
            des_censo=st.selectbox('Descripción CNPV',[
                'Departamentos con mayor población en zonas no interconectadas',
                'Municipios con mayor población en zonas no interconectadas',
                'Centros poblados no interconectados con mayor población'
            ])
            if des_censo=='Departamentos con mayor población en zonas no interconectadas':
                pobl_dep=energias_df.groupby(['Departamento'])[
                    'Total Personas en Hogares Particulares'
                    ].sum().reset_index().sort_values(
                        by='Total Personas en Hogares Particulares',ascending=False
                        )
                st.dataframe(pobl_dep)
                pie_pobl_dep=px.pie(
                    pobl_dep,names='Departamento',\
                        values='Total Personas en Hogares Particulares'
                )
                st.plotly_chart(
                    pie_pobl_dep, use_container_width=True
                )
            elif des_censo=='Municipios con mayor población en zonas no interconectadas':
                pobl_mun=energias_df.groupby(['Departamento','Municipio'])[
                    'Total Personas en Hogares Particulares'
                    ].sum().reset_index().sort_values(
                        by='Total Personas en Hogares Particulares',ascending=False
                        )
                st.dataframe(pobl_mun)
                pie_pobl_mundep=px.pie(
                    pobl_mun,names='Municipio',\
                        values='Total Personas en Hogares Particulares'
                )
                st.plotly_chart(
                    pie_pobl_mundep, use_container_width=True
                )
            elif des_censo=='Centros poblados no interconectados con mayor población':
                pobl_loc=energias_df.groupby(['Departamento','Municipio','Centro Poblado'])[
                    'Total Personas en Hogares Particulares'
                    ].sum().reset_index().sort_values(
                        by='Total Personas en Hogares Particulares',ascending=False
                        )
                st.dataframe(pobl_loc)
                pie_pobl_loc=px.pie(
                    pobl_loc,names='Centro Poblado',\
                        values='Total Personas en Hogares Particulares'
                )
                st.plotly_chart(
                    pie_pobl_loc, use_container_width=True
                )
        elif analisis_g=='Descripción NBI':
            des_nbi=st.selectbox('Descripción NBI',[
                'Departamentos con mayor cantidad de población en NBI',
                'Municipios con mayor cantidad de población en NBI',
                'Centros poblados con mayor cantidad de población en NBI'
            ])
            if des_nbi=='Departamentos con mayor cantidad de población en NBI':
                nbi_dep=energias_df.groupby(['Departamento'])[
                    'Personas en NBI [%]'
                    ].sum().reset_index().sort_values(
                        by='Personas en NBI [%]',ascending=False
                        )
                nbi_dep=nbi_dep.rename(columns={'Personas en NBI [%]':'Personas en NBI'})
                st.dataframe(nbi_dep)
                pie_pobl_dep=px.pie(
                    nbi_dep,names='Departamento',\
                        values='Personas en NBI'
                )
                st.plotly_chart(
                    pie_pobl_dep, use_container_width=True
                )
            elif des_nbi=='Municipios con mayor cantidad de población en NBI':
                nbi_mun=energias_df.groupby(['Departamento','Municipio'])[
                    'Personas en NBI [%]'
                    ].sum().reset_index().sort_values(
                        by='Personas en NBI [%]',ascending=False
                        )
                nbi_mun=nbi_mun.rename(columns={'Personas en NBI [%]':'Personas en NBI'})
                st.dataframe(nbi_mun)
                pie_nbi_mun=px.pie(
                    nbi_mun,names='Municipio',\
                        values='Personas en NBI'
                )
                st.plotly_chart(
                    pie_nbi_mun, use_container_width=True
                )
            elif des_nbi=='Centros poblados con mayor cantidad de población en NBI':
                nbi_loc=energias_df.groupby(['Departamento','Municipio','Centro Poblado'])[
                    'Personas en NBI [%]'
                    ].sum().reset_index().sort_values(
                        by='Personas en NBI [%]',ascending=False
                        )
                nbi_loc=nbi_loc.rename(columns={'Personas en NBI [%]':'Personas en NBI'})
                st.dataframe(nbi_loc)
                pie_nbi_loc=px.pie(
                    nbi_loc,names='Centro Poblado',\
                        values='Personas en NBI'
                )
                st.plotly_chart(
                    pie_nbi_loc, use_container_width=True
                )

if opcion=="Análisis por centro poblado":
    steps=st.tabs(
        ['Ubicación','Censo','Estado de Servicio','Estadística Descriptiva']
        )
    with steps[0]:
        departamento=st.selectbox(
            'Escoge el departamento de interés',
            energias_df['Departamento']
            .sort_values(ascending=True)
            .drop_duplicates()
            )
        id_departamento=consulta_ids(
            'energias.servicios_detalle','Código Departamento',
            'Departamento',departamento
            )
        st.text(f'''El código del departamento seleccionado es: {id_departamento}''')
        municipio=st.selectbox(
            'Escoge el municipio de interés',
            energias_df[energias_df['Código Departamento']==id_departamento]['Municipio']
            .sort_values(ascending=True).drop_duplicates()
            )
        id_municipio=consulta_ids(
            'energias.servicios_detalle','Código Municipio','Municipio',municipio
            )
        st.text(f'''El código del municipio seleccionado es: {id_municipio}''')
        centro_poblado=st.selectbox(
            'Escoge el centro poblado de interés',
            energias_df[energias_df['Código Municipio']==id_municipio]['Centro Poblado']
            .sort_values(ascending=True).drop_duplicates()
            )
        id_centro_poblado=consulta_ids(
            'energias.servicios_detalle','Código Centro Poblado','Centro Poblado',
            centro_poblado
            )
        st.text(f'''El código del centro poblado seleccionado es: {id_centro_poblado}''')
        df_centro_poblado=(
            energias_df[energias_df['Código Centro Poblado']==id_centro_poblado]
            .sort_values(by='Fecha Demanda Máxima',ascending=True)
            )
        ubicacion_centro_poblado=df_centro_poblado[[
            'Latitud','Longitud'
            ]].drop_duplicates()
        ubicacion_centro_poblado=ubicacion_centro_poblado.rename(
            columns={'Latitud':'lat','Longitud':'lon'}
            )
        if ubicacion_centro_poblado.empty\
            or ubicacion_centro_poblado['lat'].isna().all()\
                or ubicacion_centro_poblado['lon'].isna().all():
            st.write(
                'No hay una ubicación registrada'
                )
        else:
            st.map(
                ubicacion_centro_poblado
                )
    with steps[1]:
        df_cnpv_nbi=df_centro_poblado
        df_cnpv_nbi['Personas Sin NBI [%]']=100-df_cnpv_nbi['Personas en NBI [%]']
        df_cnpv_nbi['Personas Con Servicios [%]']=100-df_cnpv_nbi[
            'Componente Servicios [%]']
        st.dataframe(
            df_cnpv_nbi[
                ['Total Personas en Hogares Particulares','Personas en NBI [%]',
                    'Componente Servicios [%]']
            ].drop_duplicates()
            )
        df_nbi=df_cnpv_nbi[
            ['Personas en NBI [%]','Personas Sin NBI [%]']
            ]
        df_nbi_melt=df_nbi.melt(
            var_name='Condición',value_name='Cantidad'
            )
        df_nbi_servicios=df_cnpv_nbi[
            ['Componente Servicios [%]','Personas Con Servicios [%]']
        ]
        df_nbi_servicios_melt=df_nbi_servicios.melt(
            var_name='Condición',value_name='Cantidad'
        )
        fig_nbi_nbi=px.pie(
            df_nbi_melt,names='Condición',values='Cantidad'
            )
        fig_nbi_servicios=px.pie(
            df_nbi_servicios_melt,names='Condición',values='Cantidad'
        )
        fig_nbi=make_subplots(
            rows=1,cols=2,
            subplot_titles=('Distribución NBI','Distribución Componente Servicios'),
            specs=[[{'type':'pie'},{'type':'pie'}]]
        )
        fig_nbi.add_trace(fig_nbi_nbi.data[0],row=1,col=1)
        fig_nbi.add_trace(fig_nbi_servicios.data[0],row=1,col=2)
        st.plotly_chart(
            fig_nbi, use_container_width=True
            )
    with steps[2]:
        df_energias_centro_poblado=df_centro_poblado[
            ['Fecha Demanda Máxima','Promedio Diario [h]','Energía Activa [kWh]',
            'Energía Reactiva [kVArh]','Factor de Potencia','Potencia Máxima [kW]']
            ]
        st.dataframe(
            df_energias_centro_poblado
            )
        x_var='Fecha Demanda Máxima'
        y_var=st.selectbox(
            'Escoge la variable que quieres ver respecto al tiempo',
            ['Promedio Diario [h]','Energía Activa [kWh]','Energía Reactiva [kVArh]',
            'Potencia Máxima [kW]','Factor de Potencia']
                )
        fig_energias=px.line(
            df_energias_centro_poblado,x=x_var,y=y_var,
            title='Variable Energética Vs Fecha Demanda Máxima'
            )
        st.plotly_chart(
            fig_energias,use_container_width=True
            )
    with steps[3]:
        analisis = st.selectbox(
            'Seleccione el tipo de análsis descriptivo',
            ['Análisis descriptivos para variables cuantitativas',
            'Análisis descriptivos multivariados']
        )
        if analisis=='Análisis descriptivos para variables cuantitativas':
            estadistica = st.selectbox(
                f'Medidas individales de la variable seleccionada ({y_var})',
                ['Medidas de tendencia central','Medidas de variabilidad',
                'Medidas de forma','Medidas de posición']
                )
            variable=df_energias_centro_poblado[y_var]
            mean=variable.mean()
            median=variable.median()
            mode=variable.mode()
            min=variable.min()
            max=variable.max()
            var=variable.var()
            std=round(variable.std(),2)
            range=max-min
            cv=round((std/mean)*100,2)
            asimetria=variable.skew()
            kurtosis=variable.kurt()
            Q1=variable.quantile(0.25)
            Q2=variable.quantile(0.50)
            Q3=variable.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            if estadistica=='Medidas de tendencia central':
                st.write("""
                    Las **medidas de tendencia central** incluyen la **media**, 
                    **mediana** y **moda**. 
                    Estas medidas nos indican el valor alrededor del cual se agrupan los datos. 
                    La **media** es el promedio, 
                    la **mediana** es el valor en el medio de los datos ordenados, 
                    y la **moda** es el valor más frecuente.
                """)
                tendencia_cental=pd.DataFrame(
                    {'Media':mean,
                    'Mediana':median,
                    'Moda':mode}
                )
                st.dataframe(tendencia_cental)
            elif estadistica=='Medidas de variabilidad':
                st.write("""
                    Las **medidas de variabilidad** incluyen el **máximo**, **mínimo**, **rango**,
                    **varianza**, **desviación estándar** y **coeficiente de variación**. 
                    Estas medidas nos indican cuán dispersos o concentrados están 
                    los datos alrededor de la tendencia central.           
                """)
                variabilidad=pd.DataFrame(
                    {'Mínimo':[min],
                    'Máximo':[max],
                    'Rango':[range],
                    'Varianza':[var],
                    'Desviación Estándar':[std],
                    'Coeficiente de Variación':[cv]}
                )
                st.dataframe(variabilidad)
            elif estadistica=='Medidas de forma':
                st.write("""
                Las **medidas de forma** incluyen la **asimetría (skewness)** y la **curtosis**. 
                La **asimetría** nos indica si la distribución está sesgada hacia la derecha o 
                hacia la izquierda. 
                La **curtosis** nos indica la "altitud" de las colas de la distribución 
                (si son más gruesas o más delgadas que una distribución normal).         
                """)      
                forma=pd.DataFrame(
                {'Coeficiente de asimetría':[asimetria],
                'Coeficiente de Kurtosis':[kurtosis]}
                )
                st.dataframe(forma)
            elif estadistica=='Medidas de posición':
                st.write("""
                Las **medidas de posición** incluyen los **cuartiles**. 
                Estas medidas nos indican la posición relativa de un valor en el conjunto de datos, 
                dividiendo los datos en diferentes intervalos para mejor comprensión de su 
                distribución.        
                """)
                posicion=pd.DataFrame(
                    {'Primer cuartil':[Q1],
                    'Segundo cuartil':[Q2],
                    'Tercer cuartil':[Q3],
                    'Rango intercuartílico':[IQR],
                    'Límite inferior':[lower_bound],
                    'Límite superior':[upper_bound]}
                )
                st.dataframe(posicion)
            graficas=st.selectbox(
                '**Gráficas descriptivas**',
                ['Boxplot','Histograma de frecuencias','Gráfico de densidad']
                )
            if graficas=='Boxplot':
                boxplot=px.box(
                    df_centro_poblado,x='Centro Poblado',y=y_var 
                    )
                st.plotly_chart(
                    boxplot, use_container_width=True
                    )
            elif graficas=='Histograma de frecuencias':
                numero_datos=len(variable)
                bins=int(math.log2(numero_datos+1))
                clases=pd.cut(variable, bins=bins)
                fa=clases.value_counts().sort_index()
                fa_acum =fa.cumsum()
                fr=(fa/fa.sum())*100
                fr_acum=fr.cumsum()
                t_frecuencia=pd.DataFrame({
                    'Intervalo de clase':fa.index.astype(str),
                    'Frecuencia absoluta (f)':fa.values,
                    'Frecuencia absoluta acumulada (F)':fa_acum.values,
                    'Frecuencia relativa (fr) [%]':fr.values,
                    'Frecuencia relativa acumulada (Fr) [%]':fr_acum.values
                })
                st.dataframe(t_frecuencia)
                histograma=px.histogram(
                    df_energias_centro_poblado,x=y_var,nbins=bins
                )
                st.plotly_chart(
                    histograma, use_container_width=True
                    )
            elif graficas=='Gráfico de densidad':
                kde = gaussian_kde(variable, bw_method=0.1)
                x_densidad=np.linspace(min,max,1000)
                y_densidad=kde(x_densidad)
                densidad=go.Figure()
                densidad.add_trace(go.Scatter(
                    x=x_densidad,y=y_densidad,mode='lines',name='Densidad KDE'
                    ))
                densidad.update_layout(
                    xaxis_title=y_var,
                    yaxis_title='Densidad'
                )
                st.plotly_chart(
                    densidad, use_container_width=True
                    )
#Este es mi PR