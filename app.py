import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from scipy.optimize import curve_fit

st.set_page_config(layout="wide")
st.title('Análisis de Ventas de Vehículos Híbridos y Eléctricos en México')

@st.cache_data
def load_data():
    urls = [
        "https://raw.githubusercontent.com/Omar2127/Proyecto-final-BI-/refs/heads/main/HibridoMensual2020.csv",
        "https://raw.githubusercontent.com/Omar2127/Proyecto-final-BI-/refs/heads/main/HibridoMensual2021.csv",
        "https://raw.githubusercontent.com/Omar2127/Proyecto-final-BI-/refs/heads/main/HibridoMensual2022.csv",
        "https://raw.githubusercontent.com/Omar2127/Proyecto-final-BI-/refs/heads/main/HibridoMensual2023.csv",
        "https://raw.githubusercontent.com/Omar2127/Proyecto-final-BI-/refs/heads/main/HibridoMensual2024.csv",
        "https://raw.githubusercontent.com/Omar2127/Proyecto-final-BI-/refs/heads/main/HibridoMensual2025.csv",
        "https://raw.githubusercontent.com/Omar2127/Proyecto-final-BI-/refs/heads/main/HibridoMensual2026.csv"
    ]
    lista_df = []
    for url in urls:
        try:
            df = pd.read_csv(url)
            df["archivo_origen"] = url.split("/")[-1]
            lista_df.append(df)
        except Exception as e:
            st.error(f"ERROR al cargar {url}: {e}")
    data = pd.concat(lista_df, ignore_index=True)

    url_mes = "https://raw.githubusercontent.com/Omar2127/Proyecto-final-BI-/refs/heads/main/Meses.csv"
    url_estado = "https://raw.githubusercontent.com/Omar2127/Proyecto-final-BI-/refs/heads/main/Entidades.csv"
    url_marcas = "https://raw.githubusercontent.com/Omar2127/Proyecto-final-BI-/refs/heads/main/marcas_origen_ene_ago_2025_2026.csv"

    meses = pd.read_csv(url_mes)
    estados = pd.read_csv(url_estado)
    marcas = pd.read_csv(url_marcas)

    # Cleaning and processing steps as in the notebook
    cols2remove = ['PROD_EST', 'COBERTURA', 'ESTATUS', 'archivo_origen']
    data.drop(columns=cols2remove, inplace=True)

    diccionario_entidades = dict(zip(estados["ID_ENTIDAD"], estados["DESC_ENTIDAD"]))
    data["ID_ENTIDAD"] = data["ID_ENTIDAD"].map(diccionario_entidades)

    diccionario_meses = dict(zip(meses["ID_MES"], meses["DESCRIPCION_MES"]))
    data["ID_MES"] = data["ID_MES"].map(diccionario_meses)

    data["HIBRIDOS"] = data["VEH_HIBRIDAS_PLUGIN"] + data["VEH_HIBRIDAS"]
    data = data.drop(columns=["VEH_HIBRIDAS_PLUGIN", "VEH_HIBRIDAS"])

    data = data.rename(columns={ "ANIO":"AÑO", "ID_MES":"MES", "ID_ENTIDAD":"ESTADO", "VEH_ELECTR":"ELECTRICOS"})

    marcas = marcas.rename (columns={
        "marca":"Marca",
        "anio":"Año",
        "origen":"Origen",
        "unidades":"Unidades"
        })

    data["ESTADO"] = data["ESTADO"].replace({ "Coahuila de Zaragoza": "Coahuila", "Michoacán de Ocampo": "Michoacán",
                                             "Veracruz de Ignacio de la Llave": "Veracruz"})

    return data, marcas

data, marcas = load_data()

# --- Sidebar for Navigation --- 
st.sidebar.title("Navegación")
option = st.sidebar.radio("Selecciona una sección:", 
                        ["Resumen de Datos", 
                         "Visualizaciones", 
                         "Modelos Predictivos",
                         "Viabilidad de Inversión"])


# --- Data Summary Section --- 
if option == "Resumen de Datos":
    st.header("Resumen de Datos")
    st.subheader("Primeras filas de los datos principales")
    st.dataframe(data.head())
    st.subheader("Primeras filas de los datos de marcas")
    st.dataframe(marcas.head())
    st.subheader("Estadísticas Descriptivas (Híbridos y Eléctricos)")
    st.dataframe(data.describe().round(2))
    st.subheader("Estadísticas Descriptivas (Marcas)")
    st.dataframe(marcas.describe().round(2))

# --- Visualizations Section --- 
elif option == "Visualizaciones":
    st.header("Visualizaciones")

    st.subheader("Híbridos vs Eléctricos (Total Nacional)")
    comparacion = data[["HIBRIDOS", "ELECTRICOS"]].sum()
    fig1, ax1 = plt.subplots(figsize=(6,6))
    ax1.pie(comparacion, labels=comparacion.index, autopct="%1.1f%%", startangle=90)
    ax1.set_title("Proporción de Híbridos vs Eléctricos (Total Nacional)")
    st.pyplot(fig1)

    st.subheader("Ventas de Híbridos vs Eléctricos por Estado (Total)")
    resumen_estado = data.groupby("ESTADO")[["HIBRIDOS", "ELECTRICOS"]].sum().sort_values(by="HIBRIDOS", ascending=True)
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    resumen_estado.plot(kind="bar", stacked=True, ax=ax2, color=["#1f3a5f", "#e8c468"])
    ax2.set_title("Ventas de Híbridos vs Eléctricos por Estado")
    ax2.set_xlabel("Estado")
    ax2.set_ylabel("Unidades")
    plt.xticks(rotation=90)
    ax2.legend(title="Tipo")
    st.pyplot(fig2)

    st.subheader("Ventas por Año y Tipo de Vehículo (Mapa Interactivo)")
    url_mexico = "https://raw.githubusercontent.com/angelnmara/geojson/master/mexicoHigh.json"
    mexico = gpd.read_file(url_mexico)

    selected_year = st.slider("Selecciona un Año para ver el mapa", 
                              min_value=int(data["AÑO"].min()), 
                              max_value=int(data["AÑO"].max()), 
                              value=int(data["AÑO"].min()))

    resumen_año_mapa = data[data["AÑO"] == selected_year].groupby("ESTADO")[["HIBRIDOS", "ELECTRICOS"]].sum().reset_index()
    mexico_datos = mexico.merge(resumen_año_mapa, left_on="name", right_on="ESTADO", how="left")

    fig3, ax3 = plt.subplots(1, 2, figsize=(15, 7))
    mexico_datos.plot(column="HIBRIDOS", cmap="Blues", legend=True, ax=ax3[0], edgecolor="black",
                       vmin=data["HIBRIDOS"].min(), vmax=data.groupby(["ESTADO","AÑO"])["HIBRIDOS"].sum().max())
    ax3[0].set_title(f"Híbridos — {selected_year}")
    ax3[0].axis("off")

    mexico_datos.plot(column="ELECTRICOS", cmap="Blues", legend=True, ax=ax3[1], edgecolor="black",
                       vmin=data["ELECTRICOS"].min(), vmax=data.groupby(["ESTADO","AÑO"])["ELECTRICOS"].sum().max())
    ax3[1].set_title(f"Eléctricos — {selected_year}")
    ax3[1].axis("off")
    st.pyplot(fig3)

    st.subheader("Participación de Marcas en el Mercado (2025-2026)")
    ventas_marca = marcas.groupby("Marca")["Unidades"].sum().sort_values(ascending=False)
    top_n = st.slider("Número de Top Marcas a mostrar", min_value=5, max_value=len(ventas_marca), value=10)
    top_marcas = ventas_marca.nlargest(top_n)
    otras = ventas_marca.iloc[top_n:].sum()
    ventas_marca_agrupado = pd.concat([top_marcas, pd.Series({"Otras": otras})])

    fig4, ax4 = plt.subplots(figsize=(8,8))
    ventas_marca_agrupado.plot(kind="pie", autopct="%1.1f%%", ax=ax4)
    ax4.set_ylabel("")
    ax4.set_title(f"Participación de Marcas en el Mercado (2025-2026) Top {top_n}")
    st.pyplot(fig4)

    st.subheader("Crecimiento Interanual (YoY)")
    type_selected = st.selectbox("Selecciona tipo de vehículo para ver crecimiento YoY", ("HIBRIDOS", "ELECTRICOS"))
    total_anual = data.groupby("AÑO")[type_selected].sum().sort_index()
    crecimiento_yoy = total_anual.pct_change() * 100

    fig5, ax5 = plt.subplots(figsize=(10,5))
    colores = ["green" if x >= 0 else "red" for x in crecimiento_yoy]
    ax5.bar(crecimiento_yoy.index.astype(str), crecimiento_yoy, color=colores)
    ax5.axhline(0, color="black", linewidth=0.8)
    ax5.set_title(f"Crecimiento interanual (YoY) de ventas de {type_selected}")
    ax5.set_xlabel("Año")
    ax5.set_ylabel("% de crecimiento vs año anterior")
    for i, v in enumerate(crecimiento_yoy):
        if pd.notna(v):
            ax5.text(i, v + (1 if v >= 0 else -3), f"{v:.1f}%", ha="center")
    st.pyplot(fig5)

# --- Predictive Models Section --- 
elif option == "Modelos Predictivos":
    st.header("Modelos Predictivos: Regresión Logística")

    def func_logistica(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    st.subheader("Modelo de Regresión Logística para Híbridos")
    resumen_anual_hibridos = data.groupby("AÑO")["HIBRIDOS"].sum().reset_index()
    resumen_anual_hibridos["AÑO_REL"] = resumen_anual_hibridos["AÑO"] - resumen_anual_hibridos["AÑO"].min()

    X_h = resumen_anual_hibridos["AÑO_REL"].values
    y_h = resumen_anual_hibridos["HIBRIDOS"].values
    params_h, _ = curve_fit(func_logistica, X_h, y_h, p0=(y_h.max()*2, 1, X_h.mean()), maxfev=10000)
    L_h, k_h, x0_h = params_h

    X_suave_h = np.linspace(X_h.min(), X_h.max(), 100)
    y_pred_h = func_logistica(X_suave_h, L_h, k_h, x0_h)

    fig6, ax6 = plt.subplots(figsize=(10,5))
    ax6.scatter(resumen_anual_hibridos["AÑO"], y_h, color="#1f3a5f", s=100, label="Datos reales", zorder=3)
    ax6.plot(X_suave_h + resumen_anual_hibridos["AÑO"].min(), y_pred_h, color="red", linewidth=2, label="Modelo logístico")
    ax6.set_xlabel("Año")
    ax6.set_ylabel("Unidades híbridas vendidas")
    ax6.set_title("Ajuste logístico: ventas de híbridos en México")
    ax6.legend()
    st.pyplot(fig6)
    st.write(f"Techo estimado (L): {L_h:.0f}")
    st.write(f"Velocidad de crecimiento (k): {k_h:.3f}")
    st.write(f"Punto de inflexión (x0): año {x0_h + resumen_anual_hibridos['AÑO'].min():.1f}")

    st.subheader("Modelo de Regresión Logística para Eléctricos")
    resumen_anual_electricos = data.groupby("AÑO")["ELECTRICOS"].sum().reset_index()
    resumen_anual_electricos["AÑO_REL"] = resumen_anual_electricos["AÑO"] - resumen_anual_electricos["AÑO"].min()

    X_e = resumen_anual_electricos["AÑO_REL"].values
    y_e = resumen_anual_electricos["ELECTRICOS"].values
    params_e, _ = curve_fit(func_logistica, X_e, y_e, p0=(y_e.max()*2, 1, X_e.mean()), maxfev=10000)
    L_e, k_e, x0_e = params_e

    X_suave_e = np.linspace(X_e.min(), X_e.max(), 100)
    y_pred_e = func_logistica(X_suave_e, L_e, k_e, x0_e)

    fig7, ax7 = plt.subplots(figsize=(10,5))
    ax7.scatter(resumen_anual_electricos["AÑO"], y_e, color="#1f3a5f", s=100, label="Datos reales", zorder=3)
    ax7.plot(X_suave_e + resumen_anual_electricos["AÑO"].min(), y_pred_e, color="red", linewidth=2, label="Modelo logístico")
    ax7.set_xlabel("Año")
    ax7.set_ylabel("Unidades eléctricas vendidas")
    ax7.set_title("Ajuste logístico: ventas de eléctricos en México")
    ax7.legend()
    st.pyplot(fig7)
    st.write(f"Techo estimado (L): {L_e:.0f}")
    st.write(f"Velocidad de crecimiento (k): {k_e:.3f}")
    st.write(f"Punto de inflexión (x0): año {x0_e + resumen_anual_electricos['AÑO'].min():.1f}")

# --- Investment Viability Section --- 
elif option == "Viabilidad de Inversión":
    st.header("Viabilidad de Crecimiento para Inversión")
    
    st.subheader("Proyección a 5 años: Híbridos")
    resumen_anual_hibridos = data.groupby("AÑO")["HIBRIDOS"].sum().reset_index()
    resumen_anual_hibridos["AÑO_REL"] = resumen_anual_hibridos["AÑO"] - resumen_anual_hibridos["AÑO"].min()

    X_h = resumen_anual_hibridos["AÑO_REL"].values
    y_h = resumen_anual_hibridos["HIBRIDOS"].values
    params_h, _ = curve_fit(func_logistica, X_h, y_h, p0=(y_h.max()*2, 1, X_h.mean()), maxfev=10000)
    L_h, k_h, x0_h = params_h

    X_suave_h_proj = np.linspace(X_h.min(), X_h.max() + 5, 100) # Project 5 years further
    y_pred_h_proj = func_logistica(X_suave_h_proj, L_h, k_h, x0_h)

    fig8, ax8 = plt.subplots(figsize=(10,5))
    ax8.scatter(resumen_anual_hibridos["AÑO"], y_h, color="#1f3a5f", s=100, label="Datos reales", zorder=3)
    ax8.plot(X_suave_h_proj + resumen_anual_hibridos["AÑO"].min(), y_pred_h_proj, color="red", linewidth=2, label="Modelo logístico (Proyección)")
    ax8.set_xlabel("Año")
    ax8.set_ylabel("Unidades híbridas vendidas")
    ax8.set_title("Proyección de Ventas de Híbridos en México (hasta 2031)")
    ax8.legend()
    st.pyplot(fig8)
    st.write("El modelo logístico proyecta que las ventas de híbridos seguirán creciendo, indicando un potencial de inversión.")

    st.subheader("Proyección a 5 años: Eléctricos")
    resumen_anual_electricos = data.groupby("AÑO")["ELECTRICOS"].sum().reset_index()
    resumen_anual_electricos["AÑO_REL"] = resumen_anual_electricos["AÑO"] - resumen_anual_electricos["AÑO"].min()

    X_e = resumen_anual_electricos["AÑO_REL"].values
    y_e = resumen_anual_electricos["ELECTRICOS"].values
    params_e, _ = curve_fit(func_logistica, X_e, y_e, p0=(y_e.max()*2, 1, X_e.mean()), maxfev=10000)
    L_e, k_e, x0_e = params_e

    X_suave_e_proj = np.linspace(X_e.min(), X_e.max() + 5, 100) # Project 5 years further
    y_pred_e_proj = func_logistica(X_suave_e_proj, L_e, k_e, x0_e)

    fig9, ax9 = plt.subplots(figsize=(10,5))
    ax9.scatter(resumen_anual_electricos["AÑO"], y_e, color="#1f3a5f", s=100, label="Datos reales", zorder=3)
    ax9.plot(X_suave_e_proj + resumen_anual_electricos["AÑO"].min(), y_pred_e_proj, color="red", linewidth=2, label="Modelo logístico (Proyección)")
    ax9.set_xlabel("Año")
    ax9.set_ylabel("Unidades eléctricas vendidas")
    ax9.set_title("Proyección de Ventas de Eléctricos en México (hasta 2031)")
    ax9.legend()
    st.pyplot(fig9)
    st.write("El modelo logístico muestra que las ventas de eléctricos parecen estar saturándose o incluso disminuyendo en la proyección, lo que podría indicar una menor viabilidad de inversión a largo plazo bajo este modelo.")

    st.markdown("**Conclusión para Inversión:** Conforme al modelo logístico, la inversión en vehículos híbridos parece ser más prometedora que en vehículos eléctricos en México, ya que los híbridos aún muestran una fase de crecimiento mientras que los eléctricos muestran signos de estancamiento o declive en el futuro cercano según este modelo.")
