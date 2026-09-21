import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import requests
from scipy.optimize import curve_fit

st.set_page_config(layout="wide")

st.title('Análisis de Ventas de Vehículos Híbridos y Eléctricos en México')


# ============================================================
# FUNCIÓN PARA CARGAR LOS DATOS
# ============================================================

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


    # Archivos complementarios
    url_mes = "https://raw.githubusercontent.com/Omar2127/Proyecto-final-BI-/refs/heads/main/Meses.csv"
    url_estado = "https://raw.githubusercontent.com/Omar2127/Proyecto-final-BI-/refs/heads/main/Entidades.csv"
    url_marcas = "https://raw.githubusercontent.com/Omar2127/Proyecto-final-BI-/refs/heads/main/marcas_origen_ene_ago_2025_2026.csv"

    meses = pd.read_csv(url_mes)
    estados = pd.read_csv(url_estado)
    marcas = pd.read_csv(url_marcas)


    # ========================================================
    # LIMPIEZA Y PROCESAMIENTO
    # ========================================================

    cols2remove = [
        'PROD_EST',
        'COBERTURA',
        'ESTATUS',
        'archivo_origen'
    ]

    data.drop(columns=cols2remove, inplace=True)


    # Diccionario de entidades
    diccionario_entidades = dict(
        zip(
            estados["ID_ENTIDAD"],
            estados["DESC_ENTIDAD"]
        )
    )

    data["ID_ENTIDAD"] = data["ID_ENTIDAD"].map(
        diccionario_entidades
    )


    # Diccionario de meses
    diccionario_meses = dict(
        zip(
            meses["ID_MES"],
            meses["DESCRIPCION_MES"]
        )
    )

    data["ID_MES"] = data["ID_MES"].map(
        diccionario_meses
    )


    # Crear columna de híbridos
    data["HIBRIDOS"] = (
        data["VEH_HIBRIDAS_PLUGIN"] +
        data["VEH_HIBRIDAS"]
    )

    data = data.drop(
        columns=[
            "VEH_HIBRIDAS_PLUGIN",
            "VEH_HIBRIDAS"
        ]
    )


    # Renombrar columnas
    data = data.rename(
        columns={
            "ANIO": "AÑO",
            "ID_MES": "MES",
            "ID_ENTIDAD": "ESTADO",
            "VEH_ELECTR": "ELECTRICOS"
        }
    )


    # Renombrar columnas de marcas
    marcas = marcas.rename(
        columns={
            "marca": "Marca",
            "anio": "Año",
            "origen": "Origen",
            "unidades": "Unidades"
        }
    )


    # Homologar nombres de estados
    data["ESTADO"] = data["ESTADO"].replace({
        "Coahuila de Zaragoza": "Coahuila",
        "Michoacán de Ocampo": "Michoacán",
        "Veracruz de Ignacio de la Llave": "Veracruz"
    })


    return data, marcas


# ============================================================
# FUNCIÓN DEL MODELO LOGÍSTICO
# ============================================================

def func_logistica(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))


# Cargar datos
data, marcas = load_data()


# ============================================================
# SIDEBAR / NAVEGACIÓN
# ============================================================

st.sidebar.title("Navegación")

option = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "Resumen de Datos",
        "Visualizaciones",
        "Modelos Predictivos",
        "Viabilidad de Inversión",
        "Análisis complementario"
    ]
)


# ============================================================
# RESUMEN DE DATOS
# ============================================================

if option == "Resumen de Datos":

    st.header("Resumen de Datos")

    st.subheader("Primeras filas de los datos principales")
    st.dataframe(data.head())

    st.subheader("Primeras filas de los datos de marcas")
    st.dataframe(marcas.head())

    st.subheader("Estadísticas Descriptivas (Híbridos y Eléctricos)")
    st.dataframe(
        data.describe().round(2)
    )

    st.subheader("Estadísticas Descriptivas (Marcas)")
    st.dataframe(
        marcas.describe().round(2)
    )


# ============================================================
# VISUALIZACIONES
# ============================================================

elif option == "Visualizaciones":

    st.header("Visualizaciones")


    # --------------------------------------------------------
    # HÍBRIDOS VS ELÉCTRICOS
    # --------------------------------------------------------

    st.subheader("Híbridos vs Eléctricos (Total Nacional)")

    comparacion = data[
        ["HIBRIDOS", "ELECTRICOS"]
    ].sum()

    fig1, ax1 = plt.subplots(
        figsize=(6, 6)
    )

    ax1.pie(
        comparacion,
        labels=comparacion.index,
        autopct="%1.1f%%",
        startangle=90
    )

    ax1.set_title(
        "Proporción de Híbridos vs Eléctricos (Total Nacional)"
    )

    st.pyplot(fig1)


    # --------------------------------------------------------
    # VENTAS POR ESTADO
    # --------------------------------------------------------

    st.subheader(
        "Ventas de Híbridos vs Eléctricos por Estado (Total)"
    )

    resumen_estado = (
        data
        .groupby("ESTADO")[["HIBRIDOS", "ELECTRICOS"]]
        .sum()
        .sort_values(
            by="HIBRIDOS",
            ascending=True
        )
    )

    fig2, ax2 = plt.subplots(
        figsize=(14, 6)
    )

    resumen_estado.plot(
        kind="bar",
        stacked=True,
        ax=ax2,
        color=["#1f3a5f", "#e8c468"]
    )

    ax2.set_title(
        "Ventas de Híbridos vs Eléctricos por Estado"
    )

    ax2.set_xlabel("Estado")
    ax2.set_ylabel("Unidades")

    plt.xticks(rotation=90)

    ax2.legend(title="Tipo")

    st.pyplot(fig2)


    # --------------------------------------------------------
    # MAPA INTERACTIVO
    # --------------------------------------------------------

    st.subheader(
        "Ventas por Año y Tipo de Vehículo (Mapa Interactivo)"
    )

    # URL del GeoJSON de México
    url_mexico = (
        "https://raw.githubusercontent.com/"
        "angelnmara/geojson/master/mexicoHigh.json"
    )

    try:

        # Descargar el GeoJSON directamente
        response = requests.get(
            url_mexico,
            timeout=30
        )

        response.raise_for_status()

        # Convertir respuesta a JSON
        geojson = response.json()

        # Crear GeoDataFrame a partir del GeoJSON
        mexico = gpd.GeoDataFrame.from_features(
            geojson["features"]
        )

        # Establecer sistema de coordenadas
        mexico = mexico.set_crs(
            epsg=4326
        )

    except Exception as e:

        st.error(
            f"No fue posible cargar el mapa de México: {e}"
        )

        st.stop()


    # Selector de año
    selected_year = st.slider(
        "Selecciona un Año para ver el mapa",
        min_value=int(data["AÑO"].min()),
        max_value=int(data["AÑO"].max()),
        value=int(data["AÑO"].min())
    )


    # Datos del año seleccionado
    resumen_año_mapa = (
        data[
            data["AÑO"] == selected_year
        ]
        .groupby("ESTADO")[["HIBRIDOS", "ELECTRICOS"]]
        .sum()
        .reset_index()
    )


    # Unir información geográfica con ventas
    mexico_datos = mexico.merge(
        resumen_año_mapa,
        left_on="name",
        right_on="ESTADO",
        how="left"
    )


    # --------------------------------------------------------
    # MAPAS
    # --------------------------------------------------------

    fig3, ax3 = plt.subplots(
        1,
        2,
        figsize=(15, 7)
    )


    # Mapa de híbridos
    mexico_datos.plot(
        column="HIBRIDOS",
        cmap="Blues",
        legend=True,
        ax=ax3[0],
        edgecolor="black",
        vmin=data["HIBRIDOS"].min(),
        vmax=data.groupby(
            ["ESTADO", "AÑO"]
        )["HIBRIDOS"].sum().max()
    )

    ax3[0].set_title(
        f"Híbridos — {selected_year}"
    )

    ax3[0].axis("off")


    # Mapa de eléctricos
    mexico_datos.plot(
        column="ELECTRICOS",
        cmap="Blues",
        legend=True,
        ax=ax3[1],
        edgecolor="black",
        vmin=data["ELECTRICOS"].min(),
        vmax=data.groupby(
            ["ESTADO", "AÑO"]
        )["ELECTRICOS"].sum().max()
    )

    ax3[1].set_title(
        f"Eléctricos — {selected_year}"
    )

    ax3[1].axis("off")

    st.pyplot(fig3)


    # --------------------------------------------------------
    # PARTICIPACIÓN DE MARCAS
    # --------------------------------------------------------

    st.subheader(
        "Participación de Marcas en el Mercado (2025-2026)"
    )

    ventas_marca = (
        marcas
        .groupby("Marca")["Unidades"]
        .sum()
        .sort_values(
            ascending=False
        )
    )

    top_n = st.slider(
        "Número de Top Marcas a mostrar",
        min_value=5,
        max_value=len(ventas_marca),
        value=10
    )

    top_marcas = ventas_marca.nlargest(
        top_n
    )

    otras = ventas_marca.iloc[
        top_n:
    ].sum()

    ventas_marca_agrupado = pd.concat(
        [
            top_marcas,
            pd.Series(
                {"Otras": otras}
            )
        ]
    )


    fig4, ax4 = plt.subplots(
        figsize=(8, 8)
    )

    ventas_marca_agrupado.plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax4
    )

    ax4.set_ylabel("")

    ax4.set_title(
        f"Participación de Marcas en el Mercado (2025-2026) Top {top_n}"
    )

    st.pyplot(fig4)


    # --------------------------------------------------------
    # CRECIMIENTO YOY
    # --------------------------------------------------------

    st.subheader(
        "Crecimiento Interanual (YoY)"
    )

    type_selected = st.selectbox(
        "Selecciona tipo de vehículo para ver crecimiento YoY",
        (
            "HIBRIDOS",
            "ELECTRICOS"
        )
    )

    total_anual = (
        data
        .groupby("AÑO")[type_selected]
        .sum()
        .sort_index()
    )

    crecimiento_yoy = (
        total_anual.pct_change() * 100
    )


    fig5, ax5 = plt.subplots(
        figsize=(10, 5)
    )

    colores = [
        "green" if x >= 0 else "red"
        for x in crecimiento_yoy
    ]

    ax5.bar(
        crecimiento_yoy.index.astype(str),
        crecimiento_yoy,
        color=colores
    )

    ax5.axhline(
        0,
        color="black",
        linewidth=0.8
    )

    ax5.set_title(
        f"Crecimiento interanual (YoY) de ventas de {type_selected}"
    )

    ax5.set_xlabel("Año")
    ax5.set_ylabel(
        "% de crecimiento vs año anterior"
    )


    for i, v in enumerate(
        crecimiento_yoy
    ):

        if pd.notna(v):

            ax5.text(
                i,
                v + (1 if v >= 0 else -3),
                f"{v:.1f}%",
                ha="center"
            )


    st.pyplot(fig5)


# ============================================================
# MODELOS PREDICTIVOS
# ============================================================

elif option == "Modelos Predictivos":

    st.header(
        "Modelos Predictivos"
    )

    # --------------------------------------------------------
    # MODELO DE REGRESIÓN LINEAL
    # --------------------------------------------------------

    st.subheader(
        "Modelo de Regresión Lineal"
    )

    st.write(
        "Se comienza con una regresión lineal para entender de forma simple "
        "la tendencia general de la venta de híbridos y eléctricos."
    )

    jm_anual = (
        data
        .groupby("AÑO")[["HIBRIDOS", "ELECTRICOS"]]
        .sum()
        .reset_index()
    )

    jm_x = jm_anual["AÑO"].values.astype(float)

    for jm_tipo, jm_etiqueta in [("HIBRIDOS", "híbridos"),
                                 ("ELECTRICOS", "eléctricos")]:

        jm_y = jm_anual[jm_tipo].values.astype(float)

        # Ajuste de la recta y su R²
        jm_m, jm_b = np.polyfit(jm_x, jm_y, 1)
        jm_recta = jm_m * jm_x + jm_b
        jm_r2 = 1 - ((jm_y - jm_recta) ** 2).sum() / ((jm_y - jm_y.mean()) ** 2).sum()

        jm_fig, jm_ax = plt.subplots(
            figsize=(10, 5)
        )

        jm_ax.scatter(
            jm_x,
            jm_y,
            color="#1f3a5f",
            s=100,
            label="Datos reales",
            zorder=3
        )

        jm_ax.plot(
            jm_x,
            jm_recta,
            color="red",
            linewidth=2,
            label="Modelo lineal"
        )

        jm_ax.set_xlabel("Año")
        jm_ax.set_ylabel(f"Unidades {jm_etiqueta} vendidas")

        jm_ax.set_title(
            f"Venta de {jm_etiqueta} de 2020 a 2026"
        )

        jm_ax.legend()

        st.pyplot(jm_fig)

        st.write(f"R2 score del modelo lineal ({jm_etiqueta}): {jm_r2:.4f}")

    st.write(
        "El modelo lineal puede ser viable, pero hay que tomarlo con cautela: "
        "son tecnologías todavía en adopción y pueden desacelerarse. Por eso "
        "abajo se prueba la curva logística, con su forma de S: adopción "
        "lenta, aceleración y saturación."
    )



    # --------------------------------------------------------
    # MODELO HÍBRIDOS
    # --------------------------------------------------------

    st.subheader(
        "Modelo de Regresión Logística para Híbridos"
    )

    resumen_anual_hibridos = (
        data
        .groupby("AÑO")["HIBRIDOS"]
        .sum()
        .reset_index()
    )

    resumen_anual_hibridos["AÑO_REL"] = (
        resumen_anual_hibridos["AÑO"]
        - resumen_anual_hibridos["AÑO"].min()
    )


    X_h = resumen_anual_hibridos[
        "AÑO_REL"
    ].values

    y_h = resumen_anual_hibridos[
        "HIBRIDOS"
    ].values


    params_h, _ = curve_fit(
        func_logistica,
        X_h,
        y_h,
        p0=(
            y_h.max() * 2,
            1,
            X_h.mean()
        ),
        maxfev=10000
    )

    L_h, k_h, x0_h = params_h


    X_suave_h = np.linspace(
        X_h.min(),
        X_h.max(),
        100
    )

    y_pred_h = func_logistica(
        X_suave_h,
        L_h,
        k_h,
        x0_h
    )


    fig6, ax6 = plt.subplots(
        figsize=(10, 5)
    )

    ax6.scatter(
        resumen_anual_hibridos["AÑO"],
        y_h,
        color="#1f3a5f",
        s=100,
        label="Datos reales",
        zorder=3
    )

    ax6.plot(
        X_suave_h
        + resumen_anual_hibridos["AÑO"].min(),
        y_pred_h,
        color="red",
        linewidth=2,
        label="Modelo logístico"
    )

    ax6.set_xlabel("Año")

    ax6.set_ylabel(
        "Unidades híbridas vendidas"
    )

    ax6.set_title(
        "Ajuste logístico: ventas de híbridos en México"
    )

    ax6.legend()

    st.pyplot(fig6)


    st.write(
        f"Techo estimado (L): {L_h:.0f}"
    )

    st.write(
        f"Velocidad de crecimiento (k): {k_h:.3f}"
    )

    st.write(
        f"Punto de inflexión (x0): año "
        f"{x0_h + resumen_anual_hibridos['AÑO'].min():.1f}"
    )


    # --------------------------------------------------------
    # MODELO ELÉCTRICOS
    # --------------------------------------------------------

    st.subheader(
        "Modelo de Regresión Logística para Eléctricos"
    )

    resumen_anual_electricos = (
        data
        .groupby("AÑO")["ELECTRICOS"]
        .sum()
        .reset_index()
    )

    resumen_anual_electricos["AÑO_REL"] = (
        resumen_anual_electricos["AÑO"]
        - resumen_anual_electricos["AÑO"].min()
    )


    X_e = resumen_anual_electricos[
        "AÑO_REL"
    ].values

    y_e = resumen_anual_electricos[
        "ELECTRICOS"
    ].values


    params_e, _ = curve_fit(
        func_logistica,
        X_e,
        y_e,
        p0=(
            y_e.max() * 2,
            1,
            X_e.mean()
        ),
        maxfev=10000
    )

    L_e, k_e, x0_e = params_e


    X_suave_e = np.linspace(
        X_e.min(),
        X_e.max(),
        100
    )

    y_pred_e = func_logistica(
        X_suave_e,
        L_e,
        k_e,
        x0_e
    )


    fig7, ax7 = plt.subplots(
        figsize=(10, 5)
    )

    ax7.scatter(
        resumen_anual_electricos["AÑO"],
        y_e,
        color="#1f3a5f",
        s=100,
        label="Datos reales",
        zorder=3
    )

    ax7.plot(
        X_suave_e
        + resumen_anual_electricos["AÑO"].min(),
        y_pred_e,
        color="red",
        linewidth=2,
        label="Modelo logístico"
    )

    ax7.set_xlabel("Año")

    ax7.set_ylabel(
        "Unidades eléctricas vendidas"
    )

    ax7.set_title(
        "Ajuste logístico: ventas de eléctricos en México"
    )

    ax7.legend()

    st.pyplot(fig7)


    st.write(
        f"Techo estimado (L): {L_e:.0f}"
    )

    st.write(
        f"Velocidad de crecimiento (k): {k_e:.3f}"
    )

    st.write(
        f"Punto de inflexión (x0): año "
        f"{x0_e + resumen_anual_electricos['AÑO'].min():.1f}"
    )



    # --------------------------------------------------------
    # PRUEBA: MODELO CON LA SERIE MENSUAL
    # --------------------------------------------------------

    st.subheader(
        "Prueba: regresión lineal con la serie mensual"
    )

    st.write(
        "Los modelos de arriba se ajustan sobre siete puntos anuales. Esta "
        "prueba usa la serie mensual —80 puntos— con el mes y el año como "
        "variables, que es el camino para poder pronosticar."
    )

    jm_meses_num = {"Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4,
                    "Mayo": 5, "Junio": 6, "Julio": 7, "Agosto": 8,
                    "Septiembre": 9, "Octubre": 10, "Noviembre": 11,
                    "Diciembre": 12}

    jm_mensual = (
        data
        .groupby(["MES", "AÑO"])[["HIBRIDOS", "ELECTRICOS"]]
        .sum()
        .reset_index()
    )

    jm_mensual["MES_NUM"] = jm_mensual["MES"].map(jm_meses_num)
    jm_mensual = jm_mensual.sort_values(["AÑO", "MES_NUM"]).reset_index(drop=True)

    # Mínimos cuadrados con mes, año y ordenada al origen
    jm_matriz = np.column_stack([
        jm_mensual["MES_NUM"].values.astype(float),
        jm_mensual["AÑO"].values.astype(float),
        np.ones(len(jm_mensual))
    ])

    jm_obj = jm_mensual["HIBRIDOS"].values.astype(float)
    jm_coef = np.linalg.lstsq(jm_matriz, jm_obj, rcond=None)[0]
    jm_estimado = jm_matriz @ jm_coef
    jm_r2_mes = 1 - ((jm_obj - jm_estimado) ** 2).sum() / ((jm_obj - jm_obj.mean()) ** 2).sum()

    jm_mensual["ESTIMADO"] = jm_estimado
    jm_mensual["MES_ANIO"] = (jm_mensual["MES"] + " " +
                              jm_mensual["AÑO"].astype(str))

    jm_fig_mes, jm_ax_mes = plt.subplots(
        figsize=(15, 6)
    )

    jm_ax_mes.plot(
        range(len(jm_mensual)),
        jm_mensual["HIBRIDOS"],
        color="#1f3a5f",
        marker="o",
        markersize=3,
        linewidth=1,
        label="Datos reales"
    )

    jm_ax_mes.plot(
        range(len(jm_mensual)),
        jm_mensual["ESTIMADO"],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Modelo lineal mensual"
    )

    jm_paso = range(0, len(jm_mensual), 6)
    jm_ax_mes.set_xticks(list(jm_paso))
    jm_ax_mes.set_xticklabels(jm_mensual["MES_ANIO"].iloc[list(jm_paso)],
                              rotation=90)

    jm_ax_mes.set_title(
        "Predicción de ventas de híbridos por mes y año (modelo lineal)"
    )

    jm_ax_mes.set_ylabel("Unidades híbridas vendidas")
    jm_ax_mes.legend()
    jm_ax_mes.grid(True, linestyle="--", alpha=0.7)

    st.pyplot(jm_fig_mes)

    st.write(
        f"R2 score del modelo mensual: {jm_r2_mes:.4f}, sobre "
        f"{len(jm_mensual)} puntos. El mes y el año, por sí solos, no "
        "capturan la estacionalidad de fin de año: los picos de diciembre se "
        "quedan cortos."
    )


    # --------------------------------------------------------
    # CONCLUSIONES DE LOS MODELOS
    # --------------------------------------------------------

    st.subheader(
        "Conclusiones"
    )

    st.write(
        "En los modelos presentados, tanto regresión lineal como logística, "
        "se ve un crecimiento claro del mercado de los automóviles que no "
        "dependen de combustibles fósiles. Por lo mismo es valioso invertir "
        "en la venta de autos híbridos en México."
    )

    st.write(
        "Sin embargo, el modelo logístico de los eléctricos pone el techo del "
        "mercado en 21,528 unidades, por debajo de las 24,290 que ya se "
        "vendieron en 2024: con siete puntos anuales y tres parámetros, ese "
        "techo queda indeterminado y no se puede concluir si es una inversión "
        "segura con los datos que se tienen."
    )

    st.write(
        "Los estados donde la venta de híbridos va al alza son los más "
        "relevantes para invertir: Ciudad de México, Estado de México, Nuevo "
        "León y Jalisco. El detalle, estado por estado, está en la sección "
        "Análisis complementario."
    )


# ============================================================
# VIABILIDAD DE INVERSIÓN
# ============================================================

elif option == "Viabilidad de Inversión":

    st.header(
        "Viabilidad de Crecimiento para Inversión"
    )


    # --------------------------------------------------------
    # PROYECCIÓN HÍBRIDOS
    # --------------------------------------------------------

    st.subheader(
        "Proyección a 5 años: Híbridos"
    )

    resumen_anual_hibridos = (
        data
        .groupby("AÑO")["HIBRIDOS"]
        .sum()
        .reset_index()
    )

    resumen_anual_hibridos["AÑO_REL"] = (
        resumen_anual_hibridos["AÑO"]
        - resumen_anual_hibridos["AÑO"].min()
    )


    X_h = resumen_anual_hibridos[
        "AÑO_REL"
    ].values

    y_h = resumen_anual_hibridos[
        "HIBRIDOS"
    ].values


    params_h, _ = curve_fit(
        func_logistica,
        X_h,
        y_h,
        p0=(
            y_h.max() * 2,
            1,
            X_h.mean()
        ),
        maxfev=10000
    )

    L_h, k_h, x0_h = params_h


    X_suave_h_proj = np.linspace(
        X_h.min(),
        X_h.max() + 5,
        100
    )

    y_pred_h_proj = func_logistica(
        X_suave_h_proj,
        L_h,
        k_h,
        x0_h
    )


    fig8, ax8 = plt.subplots(
        figsize=(10, 5)
    )

    ax8.scatter(
        resumen_anual_hibridos["AÑO"],
        y_h,
        color="#1f3a5f",
        s=100,
        label="Datos reales",
        zorder=3
    )

    ax8.plot(
        X_suave_h_proj
        + resumen_anual_hibridos["AÑO"].min(),
        y_pred_h_proj,
        color="red",
        linewidth=2,
        label="Modelo logístico (Proyección)"
    )

    ax8.set_xlabel("Año")

    ax8.set_ylabel(
        "Unidades híbridas vendidas"
    )

    ax8.set_title(
        "Proyección de Ventas de Híbridos en México (hasta 2031)"
    )

    ax8.legend()

    st.pyplot(fig8)

    st.write(
        "El modelo logístico proyecta que las ventas de "
        "híbridos seguirán creciendo, indicando un potencial "
        "de inversión."
    )


    # --------------------------------------------------------
    # PROYECCIÓN ELÉCTRICOS
    # --------------------------------------------------------

    st.subheader(
        "Proyección a 5 años: Eléctricos"
    )

    resumen_anual_electricos = (
        data
        .groupby("AÑO")["ELECTRICOS"]
        .sum()
        .reset_index()
    )

    resumen_anual_electricos["AÑO_REL"] = (
        resumen_anual_electricos["AÑO"]
        - resumen_anual_electricos["AÑO"].min()
    )


    X_e = resumen_anual_electricos[
        "AÑO_REL"
    ].values

    y_e = resumen_anual_electricos[
        "ELECTRICOS"
    ].values


    params_e, _ = curve_fit(
        func_logistica,
        X_e,
        y_e,
        p0=(
            y_e.max() * 2,
            1,
            X_e.mean()
        ),
        maxfev=10000
    )

    L_e, k_e, x0_e = params_e


    X_suave_e_proj = np.linspace(
        X_e.min(),
        X_e.max() + 5,
        100
    )

    y_pred_e_proj = func_logistica(
        X_suave_e_proj,
        L_e,
        k_e,
        x0_e
    )


    fig9, ax9 = plt.subplots(
        figsize=(10, 5)
    )

    ax9.scatter(
        resumen_anual_electricos["AÑO"],
        y_e,
        color="#1f3a5f",
        s=100,
        label="Datos reales",
        zorder=3
    )

    ax9.plot(
        X_suave_e_proj
        + resumen_anual_electricos["AÑO"].min(),
        y_pred_e_proj,
        color="red",
        linewidth=2,
        label="Modelo logístico (Proyección)"
    )

    ax9.set_xlabel("Año")

    ax9.set_ylabel(
        "Unidades eléctricas vendidas"
    )

    ax9.set_title(
        "Proyección de Ventas de Eléctricos en México (hasta 2031)"
    )

    ax9.legend()

    st.pyplot(fig9)

    st.write(
        "El modelo logístico muestra que las ventas de "
        "eléctricos parecen estar saturándose o incluso "
        "disminuyendo en la proyección, lo que podría indicar "
        "una menor viabilidad de inversión a largo plazo "
        "bajo este modelo."
    )


    # --------------------------------------------------------
    # CONCLUSIÓN
    # --------------------------------------------------------

    st.markdown(
        "**Conclusión para Inversión:** Conforme al modelo "
        "logístico, la inversión en vehículos híbridos parece "
        "ser más prometedora que en vehículos eléctricos en "
        "México, ya que los híbridos aún muestran una fase de "
        "crecimiento mientras que los eléctricos muestran "
        "signos de estancamiento o declive en el futuro cercano "
        "según este modelo."
    )


# ============================================================
# ANÁLISIS COMPLEMENTARIO (aportación de José)
# ============================================================
# Usa sus propias variables (todas empiezan con jm_) para no
# cambiar ninguna de las de arriba, y sólo pandas, numpy y
# matplotlib. Lee sus datos de la carpeta datos/ del equipo.

elif option == "Análisis complementario":

    st.header(
        "Análisis complementario"
    )

    st.write(
        "Qué traen mal los datos, qué hay dentro de la columna de "
        "híbridos, cómo se ve el mercado estado por estado y en qué "
        "estados conviene vender cada tecnología."
    )


    # --------------------------------------------------------
    # CARGA DE LOS DATOS COMPLEMENTARIOS
    # --------------------------------------------------------

    @st.cache_data
    def jm_cargar_datos():

        # Archivos de la carpeta datos/ del equipo en Google Drive
        jm_drive = "https://drive.google.com/uc?export=download&id="

        jm_autos = pd.read_csv(jm_drive + "1A2CMmpvCVc9qkIfzG77VgeNpPtyZIuMr")     # INEGI: estado x año, con los 3 tipos separados
        jm_mensual = pd.read_csv(jm_drive + "10DB8BtuTplvy5JOgBYPUL_2F7s40wvnO")   # INEGI: nacional por mes y tipo
        jm_ingreso = pd.read_csv(jm_drive + "1y7nkzGvZdoT7-lKpCeON3e6EiKKRQAte")   # INEGI: ingreso por hogar (ENIGH 2024)
        jm_totales = pd.read_csv(jm_drive + "1qy6EV9LlV1jpMsaHAtS-7BFK4seFBSvA")   # AMDA: autos vendidos por estado
        jm_carga = pd.read_csv(jm_drive + "1JyP86Ivm_w2zfRA1yOmBN_av3TgKbVb3")     # CFE: electrolineras públicas
        jm_agencias = pd.read_csv(jm_drive + "15MgN38Agmm3XQTPWy7OzE9NPptxvfWrG")  # INEGI DENUE: agencias de autos nuevos

        # Tabla de marcas del repositorio del equipo
        jm_marcas = pd.read_csv(
            "https://raw.githubusercontent.com/Omar2127/Proyecto-final-BI-/refs/heads/main/marcas_origen_ene_ago_2025_2026.csv"
        )

        # Mismo periodo que el resto del dashboard
        jm_autos = jm_autos[jm_autos["anio"] >= 2020].copy()
        jm_mensual = jm_mensual[jm_mensual["anio"] >= 2020].copy()

        return jm_autos, jm_mensual, jm_ingreso, jm_totales, jm_carga, jm_agencias, jm_marcas


    jm_autos, jm_mensual, jm_ingreso, jm_totales, jm_carga, jm_agencias, jm_marcas = jm_cargar_datos()

    # Nombres cortos de estados, igual que arriba
    jm_cortos = {
        "Coahuila de Zaragoza": "Coahuila",
        "Michoacán de Ocampo": "Michoacán",
        "Veracruz de Ignacio de la Llave": "Veracruz"
    }


    # --------------------------------------------------------
    # CALIDAD DE LOS DATOS
    # --------------------------------------------------------

    st.subheader(
        "Calidad de los datos"
    )

    # Meses que trae cada año
    jm_meses_anio = jm_mensual.groupby("anio")["mes_num"].nunique()

    # Tabla de marcas contra el total oficial de INEGI (enero-agosto)
    jm_inegi_ene_ago = pd.Series({2025: 969385, 2026: 1014715})
    jm_de_mas = jm_marcas.groupby("anio")["unidades"].sum() - jm_inegi_ene_ago

    jm_cuatro = (
        jm_marcas[jm_marcas["marca"].isin(["BYD", "Chirey", "Omoda", "GAC"])]
        .groupby(["marca", "anio"])["unidades"]
        .sum()
        .unstack()
    )

    st.write(
        f"2026 trae {jm_meses_anio[2026]} meses: llega a agosto. Por eso "
        "las comparaciones con 2026 se hacen enero-agosto contra enero-agosto."
    )

    st.write(
        f"La tabla de marcas trae {jm_de_mas[2026]:,} unidades más que INEGI "
        "en 2026. Son cuatro marcas que vienen de otra fuente:"
    )

    st.dataframe(jm_cuatro)

    st.write(
        "BYD nunca le ha reportado a INEGI, y Chirey y Omoda dejaron de "
        "hacerlo en abril de 2025. Todo lo que se mide por estado describe "
        "a las marcas que sí reportan."
    )


    # --------------------------------------------------------
    # ELECTRIFICADOS CONTRA EL MERCADO TOTAL
    # --------------------------------------------------------

    st.subheader(
        "Electrificados contra el mercado total"
    )

    # Enero-agosto de cada año, por tipo
    jm_ea_tipo = (
        jm_mensual[jm_mensual["mes_num"] <= 8]
        .groupby(["anio", "tipo"])["unidades"]
        .sum()
        .unstack()
    )

    jm_elec_ea = jm_ea_tipo.loc[[2025, 2026]].sum(axis=1)   # electrificados
    jm_resto_ea = jm_inegi_ene_ago - jm_elec_ea              # todo lo demás

    jm_crec = pd.Series({
        "Mercado total": jm_inegi_ene_ago[2026] / jm_inegi_ene_ago[2025] - 1,
        "Electrificados": jm_elec_ea[2026] / jm_elec_ea[2025] - 1,
        "Todo lo demás": jm_resto_ea[2026] / jm_resto_ea[2025] - 1
    }) * 100

    jm_fig1, jm_ax1 = plt.subplots(
        figsize=(10, 5)
    )

    jm_ax1.bar(
        jm_crec.index,
        jm_crec,
        color=["#1f3a5f", "#e8c468", "#1f3a5f"]
    )

    for jm_i, jm_v in enumerate(jm_crec):
        jm_ax1.text(jm_i, jm_v + 1, f"{jm_v:.1f}%", ha="center")

    jm_ax1.set_title(
        "Crecimiento enero-agosto 2025 a 2026"
    )

    jm_ax1.set_ylabel("% de crecimiento")
    jm_ax1.margins(y=0.1)      # espacio arriba para las etiquetas

    st.pyplot(jm_fig1)

    jm_pen_ea = jm_elec_ea / jm_inegi_ene_ago * 100

    st.write(
        f"De los {jm_inegi_ene_ago[2026] - jm_inegi_ene_ago[2025]:,} autos que "
        f"México vendió de más, {jm_elec_ea[2026] - jm_elec_ea[2025]:,} fueron "
        "híbridos o eléctricos: nueve de cada diez. La penetración pasó de "
        f"{jm_pen_ea[2025]:.1f} a {jm_pen_ea[2026]:.1f} de cada 100 autos nuevos."
    )


    # --------------------------------------------------------
    # HÍBRIDOS ENCHUFABLES Y CONVENCIONALES
    # --------------------------------------------------------

    st.subheader(
        "Híbridos enchufables y convencionales"
    )

    # Separamos la columna HIBRIDOS en sus dos tipos
    jm_anual = jm_mensual.groupby(["anio", "tipo"])["unidades"].sum().unstack()
    jm_anual["pct_enchufable"] = (
        jm_anual["hibrido_enchufable"]
        / (jm_anual["hibrido_enchufable"] + jm_anual["hibrido"]) * 100
    )

    # Mismo periodo: enero-agosto 2025 contra 2026
    jm_dos = jm_ea_tipo.loc[[2025, 2026]].T
    jm_dos["cambio"] = (jm_dos[2026] / jm_dos[2025] - 1) * 100
    jm_cambio_tipo = jm_dos.loc[["hibrido_enchufable", "electrico", "hibrido"], "cambio"]

    jm_fig2, jm_ax2 = plt.subplots(
        figsize=(10, 5)
    )

    jm_ax2.bar(
        ["Enchufable", "Eléctrico", "Convencional"],
        jm_cambio_tipo,
        color=["#e8c468", "#1f3a5f", "#1f3a5f"]
    )

    for jm_i, jm_v in enumerate(jm_cambio_tipo):
        jm_ax2.text(jm_i, jm_v + (3 if jm_v >= 0 else -12), f"{jm_v:.1f}%", ha="center")

    jm_ax2.axhline(0, color="black", linewidth=0.8)

    jm_ax2.set_title(
        "Cambio enero-agosto 2025 a 2026, por tipo"
    )

    jm_ax2.set_ylabel("% de cambio")
    jm_ax2.margins(y=0.1)

    st.pyplot(jm_fig2)

    st.write(
        "Dentro de lo que arriba se cuenta como HIBRIDOS, el enchufable pasó "
        f"de {jm_anual.loc[2020, 'pct_enchufable']:.1f} % en 2020 a "
        f"{jm_anual.loc[2026, 'pct_enchufable']:.1f} % en 2026. De enero a agosto "
        f"creció {jm_cambio_tipo['hibrido_enchufable']:.1f} %, contra "
        f"{jm_cambio_tipo['hibrido']:.1f} % del convencional: al sumarlos, el "
        "crecimiento del enchufable no se ve."
    )

    st.write(
        f"En 2026 el enchufable ({jm_ea_tipo.loc[2026, 'hibrido_enchufable']:,}) "
        f"ya vende más que el eléctrico puro ({jm_ea_tipo.loc[2026, 'electrico']:,})."
    )


    # --------------------------------------------------------
    # PARTICIPACIÓN ELÉCTRICA POR ESTADO (HEATMAP)
    # --------------------------------------------------------

    st.subheader(
        "Participación eléctrica por estado (heatmap)"
    )

    # Eléctricos entre el total electrificado, por estado y año
    jm_autos["share_ele"] = jm_autos["electrico"] / jm_autos["total_hye"] * 100
    jm_tabla = jm_autos.pivot(index="entidad", columns="anio", values="share_ele").round(1)
    jm_orden = jm_tabla.sort_values(2026, ascending=False).rename(index=jm_cortos)

    jm_fig3, jm_ax3 = plt.subplots(
        figsize=(14, 6)
    )

    jm_calor = jm_ax3.imshow(
        jm_orden.T,
        cmap="Blues",
        aspect="auto"
    )

    jm_fig3.colorbar(jm_calor, label="Participación eléctrica (%)")

    jm_ax3.set_xticks(range(len(jm_orden.index)))
    jm_ax3.set_xticklabels(jm_orden.index, rotation=90)
    jm_ax3.set_yticks(range(len(jm_orden.columns)))
    jm_ax3.set_yticklabels(jm_orden.columns)

    jm_ax3.set_title(
        "Participación eléctrica por estado (2026: enero-agosto)"
    )

    st.pyplot(jm_fig3)

    jm_caida_ele = (jm_anual.loc[2025, "electrico"] / jm_anual.loc[2024, "electrico"] - 1) * 100
    jm_max_23_24 = jm_tabla.idxmax(axis=1).isin([2023, 2024]).sum()
    jm_bajaron = (jm_tabla[2025] < jm_tabla[2024]).sum()

    st.write(
        f"Los eléctricos cayeron {abs(jm_caida_ele):.1f} % de 2024 a 2025. No fue "
        f"de un par de estados: {jm_max_23_24} de 32 tocaron su máximo en 2023 o "
        f"2024, y en 2025 la participación bajó en {jm_bajaron} de 32."
    )


    # --------------------------------------------------------
    # ESTACIONALIDAD Y CIERRE ESTIMADO DE 2026
    # --------------------------------------------------------

    st.subheader(
        "Estacionalidad y cierre estimado de 2026"
    )

    # Peso de cada mes dentro de su año, promedio 2021-2025 (2020 fue pandemia)
    jm_mes = jm_mensual.groupby(["anio", "mes_num"])["unidades"].sum().unstack()
    jm_peso = jm_mes.loc[2020:2025].div(jm_mes.loc[2020:2025].sum(axis=1), axis=0) * 100
    jm_perfil = jm_peso.loc[2021:2025].mean().round(2)

    jm_peso_ene_ago = jm_perfil.loc[1:8].sum()      # cuánto pesa enero-agosto en el año
    jm_cierre = jm_ea_tipo.loc[2026].sum() / (jm_peso_ene_ago / 100)

    jm_fig4, jm_ax4 = plt.subplots(
        figsize=(10, 5)
    )

    jm_ax4.bar(
        jm_perfil.index,
        jm_perfil,
        color="#1f3a5f"
    )

    jm_ax4.axhline(100 / 12, color="red", linewidth=0.8, label="Si todos los meses pesaran igual")

    jm_ax4.set_xticks(range(1, 13))
    jm_ax4.set_xticklabels(["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"])

    jm_ax4.set_title(
        "Peso de cada mes en el año (promedio 2021-2025)"
    )

    jm_ax4.set_ylabel("% del año")
    jm_ax4.legend()

    st.pyplot(jm_fig4)

    # Probamos el método con años que ya pasaron: estimar cada año sólo con los anteriores
    jm_errores = []
    for jm_anio in [2022, 2023, 2024, 2025]:
        jm_p = jm_peso.loc[2021:jm_anio - 1].mean().loc[1:8].sum()
        jm_error = (jm_ea_tipo.loc[jm_anio].sum() / (jm_p / 100) / jm_mes.loc[jm_anio].sum() - 1) * 100
        jm_errores.append(f"{jm_anio}: {jm_error:.1f} %")

    st.write(
        f"Diciembre es el mes más fuerte ({jm_perfil.loc[12]:.2f} %). Enero-agosto "
        f"es el {jm_peso_ene_ago:.1f} % del año, no el 66.7 % que supone "
        "multiplicar por 12/8, así que el cierre estimado de 2026 es de "
        f"{jm_cierre:,.0f} unidades."
    )

    st.write(
        "Probado con años que ya pasaron, el método se quedó corto las "
        f"cuatro veces ({', '.join(jm_errores)}), así que {jm_cierre:,.0f} es "
        "un piso. Es el dato que le falta al modelo logístico para meter 2026 "
        "como año completo."
    )


    # --------------------------------------------------------
    # PENETRACIÓN POR ESTADO
    # --------------------------------------------------------

    st.subheader(
        "Penetración por estado"
    )

    st.write(
        "Contar unidades por estado mide el tamaño del estado. La penetración "
        "mide el comportamiento: de cada 100 autos nuevos que se venden en el "
        "estado, cuántos son electrificados. El total de autos por estado viene "
        "de AMDA."
    )

    # Unimos los electrificados de 2026 con el total de autos vendidos por estado
    jm_2026 = jm_autos[jm_autos["anio"] == 2026].merge(
        jm_totales[jm_totales["anio"] == 2026][["entidad", "ventas_totales"]],
        on="entidad"
    )

    jm_2026["penetracion"] = jm_2026["total_hye"] / jm_2026["ventas_totales"] * 100
    jm_pen_nal = jm_2026["total_hye"].sum() / jm_2026["ventas_totales"].sum() * 100

    jm_2026["lugar_unidades"] = jm_2026["total_hye"].rank(ascending=False).astype(int)
    jm_2026["lugar_penetracion"] = jm_2026["penetracion"].rank(ascending=False).astype(int)

    jm_barras = (
        jm_2026
        .set_index("entidad")["penetracion"]
        .rename(index=jm_cortos)
        .sort_values(ascending=True)
    )

    jm_fig5, jm_ax5 = plt.subplots(
        figsize=(14, 6)
    )

    jm_ax5.bar(
        jm_barras.index,
        jm_barras,
        color="#1f3a5f"
    )

    jm_ax5.axhline(jm_pen_nal, color="red", linewidth=0.8, label=f"Nacional {jm_pen_nal:.1f}%")

    jm_ax5.set_title(
        "Penetración por estado, enero-agosto 2026"
    )

    jm_ax5.set_xlabel("Estado")
    jm_ax5.set_ylabel("Electrificados por cada 100 autos nuevos")
    jm_ax5.tick_params(axis="x", rotation=90)
    jm_ax5.legend()

    st.pyplot(jm_fig5)

    st.dataframe(
        jm_2026[["entidad", "lugar_unidades", "lugar_penetracion", "penetracion"]]
        .sort_values("lugar_penetracion")
        .head(6)
        .round(1)
    )

    st.write(
        "La CDMX es el mercado más grande y el más electrificado. Sinaloa sube "
        "del lugar 11 al 2 y Querétaro del 8 al 3: son mercados medianos tan "
        "electrificados como Monterrey o Guadalajara. El Estado de México baja "
        "del lugar 2 al 6."
    )


    # --------------------------------------------------------
    # INGRESO DE LOS HOGARES
    # --------------------------------------------------------

    st.subheader(
        "Ingreso de los hogares"
    )

    jm_2026 = jm_2026.merge(
        jm_ingreso[jm_ingreso["es_nacional"] == False][["entidad", "ingreso_total"]],
        on="entidad"
    )

    jm_r_mezcla = jm_2026["ingreso_total"].corr(jm_2026["share_ele"])
    jm_r_adopcion = jm_2026["ingreso_total"].corr(jm_2026["penetracion"])

    jm_fig6, jm_ax6 = plt.subplots(
        figsize=(10, 5)
    )

    jm_ax6.scatter(
        jm_2026["ingreso_total"],
        jm_2026["share_ele"],
        color="#1f3a5f"
    )

    for jm_edo in ["Sinaloa", "Baja California Sur", "Ciudad de México", "Chiapas"]:
        jm_fila = jm_2026[jm_2026["entidad"] == jm_edo].iloc[0]
        jm_ax6.text(jm_fila["ingreso_total"], jm_fila["share_ele"] + 0.4, jm_edo)

    jm_ax6.set_title(
        "Ingreso por hogar contra participación eléctrica, por estado"
    )

    jm_ax6.set_xlabel("Ingreso trimestral por hogar (ENIGH 2024)")
    jm_ax6.set_ylabel("Participación eléctrica 2026 (%)")
    jm_ax6.margins(0.1)

    st.pyplot(jm_fig6)

    st.write(
        f"El ingreso explica el {jm_r_mezcla ** 2 * 100:.0f} % de la mezcla "
        "(qué parte de lo electrificado es eléctrico puro) y el "
        f"{jm_r_adopcion ** 2 * 100:.0f} % de la adopción (penetración). "
        "Predice dónde se electrifica, no qué se compra."
    )

    jm_sin = jm_2026[jm_2026["entidad"] == "Sinaloa"].iloc[0]
    jm_bcs_ing = jm_2026[jm_2026["entidad"] == "Baja California Sur"].iloc[0]

    st.write(
        "Dos estados rompen la regla: Sinaloa, con ingreso medio y la "
        f"participación eléctrica más alta del país ({jm_sin['share_ele']:.1f} %), "
        "y Baja California Sur, con el tercer ingreso más alto y "
        f"{jm_bcs_ing['share_ele']:.1f} %. Es una correlación entre estados, "
        "no entre personas."
    )


    # --------------------------------------------------------
    # INFRAESTRUCTURA DE CARGA
    # --------------------------------------------------------

    st.subheader(
        "Infraestructura de carga"
    )

    # Limpieza del archivo de la CFE: nombres de estado y sitios repetidos
    jm_carga["entidad"] = jm_carga["estado"].replace({
        "Ciudad de Mexico": "Ciudad de México",
        "Estado de Mexico": "México",
        "Nuevo Leon": "Nuevo León",
        "Queretaro": "Querétaro",
        "San Luis Potosi": "San Luis Potosí",
        "Yucatan": "Yucatán",
        "Coahuila": "Coahuila de Zaragoza",
        "Michoacan": "Michoacán de Ocampo",
        "Veracruz": "Veracruz de Ignacio de la Llave"
    })

    jm_carga["nombre_simple"] = (
        jm_carga["nombre_estacion"]
        .str.normalize("NFKD")
        .str.encode("ascii", "ignore")
        .str.decode("ascii")
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )

    jm_carga = jm_carga.drop_duplicates(
        subset=["entidad", "nombre_simple", "tipo_01", "electrolineras_totales"]
    ).copy()

    # Cargadores de Tesla (conector propio) en los cinco grupos de cada sitio
    jm_carga["tesla"] = 0
    for jm_i in range(1, 6):
        jm_n = pd.to_numeric(jm_carga[f"cargadores_0{jm_i}"], errors="coerce").fillna(0)
        jm_es_tesla = jm_carga[f"tipo_0{jm_i}"].str.startswith("Tesla", na=False)
        jm_carga["tesla"] = jm_carga["tesla"] + jm_n.where(jm_es_tesla, 0)

    # Cargadores por cada mil autos nuevos de 2025, por estado
    jm_carga_edo = jm_carga.groupby("entidad")[["electrolineras_totales", "tesla"]].sum().reset_index()
    jm_carga_edo = jm_carga_edo.merge(
        jm_totales[jm_totales["anio"] == 2025][["entidad", "ventas_totales"]],
        on="entidad"
    )
    jm_carga_edo["por_mil_autos"] = jm_carga_edo["electrolineras_totales"] / jm_carga_edo["ventas_totales"] * 1000
    jm_carga_edo["sin_tesla_por_mil"] = (
        (jm_carga_edo["electrolineras_totales"] - jm_carga_edo["tesla"])
        / jm_carga_edo["ventas_totales"] * 1000
    )
    jm_carga_edo["pct_tesla"] = jm_carga_edo["tesla"] / jm_carga_edo["electrolineras_totales"] * 100

    jm_2026 = jm_2026.merge(
        jm_carga_edo[["entidad", "por_mil_autos", "sin_tesla_por_mil", "pct_tesla"]],
        on="entidad"
    )

    jm_total_carg = jm_carga["electrolineras_totales"].sum()
    jm_bcs = jm_2026[jm_2026["entidad"] == "Baja California Sur"].iloc[0]

    st.write(
        f"Quitando sitios repetidos quedan {jm_total_carg:,} cargadores públicos, "
        f"y el {jm_carga['tesla'].sum() / jm_total_carg * 100:.1f} % son de Tesla. "
        "La carga explica menos que el ingreso: r = "
        f"{jm_2026['por_mil_autos'].corr(jm_2026['share_ele']):.3f} contra "
        f"{jm_r_mezcla:.3f}."
    )

    st.write(
        "Baja California Sur no es falta de cargadores: tiene "
        f"{jm_bcs['por_mil_autos']:.1f} por cada mil autos nuevos, el segundo "
        f"lugar del país. Pero el {jm_bcs['pct_tesla']:.0f} % son de Tesla, en "
        f"hoteles; sin ellos le quedan {jm_bcs['sin_tesla_por_mil']:.1f}."
    )


    # --------------------------------------------------------
    # RECOMENDACIÓN POR ESTADO
    # --------------------------------------------------------

    st.subheader(
        "Recomendación por estado"
    )

    st.markdown(
        "Las reglas se escriben antes de aplicarlas, con cortes que ya salieron arriba:\n\n"
        "* **Eléctrico ya:** penetración arriba del nacional, se enchufa arriba "
        "del nacional y su participación eléctrica subió desde 2022.\n"
        "* **Híbrido:** mercado grande (arriba de la mediana) que se enchufa "
        "abajo del nacional.\n"
        "* **Todavía no:** el resto."
    )

    # Qué parte de lo electrificado se enchufa (eléctrico + enchufable), y los cortes nacionales
    jm_2026["enchufa"] = (jm_2026["electrico"] + jm_2026["hibrido_enchufable"]) / jm_2026["total_hye"] * 100
    jm_enchufa_nal = (jm_2026["electrico"].sum() + jm_2026["hibrido_enchufable"].sum()) / jm_2026["total_hye"].sum() * 100
    jm_mediana = jm_2026["total_hye"].median()

    # Cambio de participación eléctrica desde 2022 (estados con al menos 300 electrificados en 2022)
    jm_base_2022 = jm_autos[jm_autos["anio"] == 2022].set_index("entidad")["total_hye"]
    jm_2026["cambio_share"] = jm_2026["entidad"].map((jm_tabla[2026] - jm_tabla[2022])[jm_base_2022 >= 300])


    def jm_clasificar(jm_fila):
        if jm_fila["penetracion"] > jm_pen_nal and jm_fila["enchufa"] > jm_enchufa_nal and jm_fila["cambio_share"] > 0:
            return "Eléctrico ya"
        if jm_fila["total_hye"] > jm_mediana and jm_fila["enchufa"] < jm_enchufa_nal:
            return "Híbrido"
        return "Todavía no"


    jm_2026["recomendacion"] = jm_2026.apply(jm_clasificar, axis=1)

    # Mapa de la recomendación (dibujando el contorno de cada estado)
    jm_geo = pd.read_json("https://raw.githubusercontent.com/angelnmara/geojson/master/mexicoHigh.json")
    jm_colores = {"Eléctrico ya": "#e8c468", "Híbrido": "#1f3a5f", "Todavía no": "#d9d9d9"}
    jm_cat = jm_2026.set_index(jm_2026["entidad"].replace(jm_cortos))["recomendacion"]

    jm_fig7, jm_ax7 = plt.subplots(
        figsize=(10, 6)
    )

    for jm_f in jm_geo["features"]:
        jm_geom = jm_f["geometry"]
        jm_partes = [jm_geom["coordinates"]] if jm_geom["type"] == "Polygon" else jm_geom["coordinates"]   # estados con islas traen varias partes
        for jm_parte in jm_partes:
            jm_ax7.add_patch(plt.Polygon(
                jm_parte[0],
                facecolor=jm_colores[jm_cat[jm_f["properties"]["name"]]],
                edgecolor="black",
                linewidth=0.3
            ))

    jm_ax7.autoscale()
    jm_ax7.set_aspect(1.1)      # para que el país no salga aplastado
    jm_ax7.axis("off")

    jm_ax7.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=jm_c, label=jm_n) for jm_n, jm_c in jm_colores.items()],
        loc="lower left"
    )

    jm_ax7.set_title(
        "Recomendación por estado"
    )

    st.pyplot(jm_fig7)

    st.dataframe(
        jm_2026[jm_2026["recomendacion"] != "Todavía no"][["entidad", "recomendacion", "total_hye", "penetracion", "enchufa"]]
        .sort_values(["recomendacion", "total_hye"], ascending=[True, False])
        .round(1)
    )

    # Selector de estado
    jm_estado = st.selectbox(
        "Selecciona un estado para ver su recomendación",
        sorted(jm_2026["entidad"])
    )

    jm_sel = jm_2026[jm_2026["entidad"] == jm_estado].iloc[0]

    st.write(
        f"**{jm_estado}: {jm_sel['recomendacion']}.** Penetración "
        f"{jm_sel['penetracion']:.1f} % (nacional {jm_pen_nal:.1f} %), se enchufa "
        f"{jm_sel['enchufa']:.1f} % (nacional {jm_enchufa_nal:.1f} %), "
        f"{jm_sel['total_hye']:,} electrificados en 2026."
    )


    # --------------------------------------------------------
    # DÓNDE ABRIR AGENCIAS
    # --------------------------------------------------------

    st.subheader(
        "Dónde abrir agencias"
    )

    st.write(
        "Estados donde la penetración crece más que en el país y cada agencia "
        "vende más que el promedio:"
    )

    jm_2026 = jm_2026.merge(jm_agencias[["entidad", "agencias"]], on="entidad")
    jm_2026 = jm_2026.merge(
        jm_totales[jm_totales["anio"] == 2025][["entidad", "ventas_totales"]].rename(columns={"ventas_totales": "ventas_2025"}),
        on="entidad"
    )
    jm_2026 = jm_2026.merge(
        jm_autos[jm_autos["anio"] == 2025][["entidad", "total_hye"]].rename(columns={"total_hye": "hye_2025"}),
        on="entidad"
    )

    jm_2026["ventas_por_agencia"] = jm_2026["ventas_totales"] / jm_2026["agencias"]
    jm_2026["cambio_pen"] = jm_2026["penetracion"] - jm_2026["hye_2025"] / jm_2026["ventas_2025"] * 100
    jm_vpa_nal = jm_2026["ventas_totales"].sum() / jm_2026["agencias"].sum()
    jm_cambio_nal = jm_pen_nal - jm_2026["hye_2025"].sum() / jm_2026["ventas_2025"].sum() * 100

    st.dataframe(
        jm_2026[(jm_2026["cambio_pen"] > jm_cambio_nal) & (jm_2026["ventas_por_agencia"] > jm_vpa_nal)][
            ["entidad", "cambio_pen", "ventas_por_agencia"]
        ]
        .sort_values("ventas_por_agencia", ascending=False)
        .round(1)
    )


    # --------------------------------------------------------
    # CONCLUSIÓN
    # --------------------------------------------------------

    st.markdown(
        "**Conclusión del análisis complementario:** El mercado de "
        "electrificados sí crece: nueve de cada diez autos que México vendió "
        "de más en 2026 fueron híbridos o eléctricos, y el cierre estimado del "
        "año es de al menos 211,405 unidades. Dentro de los híbridos, el que "
        "más crece es el enchufable (+222.8 %), que en 2026 ya vende más que el "
        "eléctrico puro. Los eléctricos puros cayeron 13.9 % en 2025, en casi "
        "todos los estados.\n\n"
        "Por estado, la recomendación es: eléctrico y enchufable en la Ciudad "
        "de México, Estado de México, Nuevo León, Jalisco y Sinaloa; híbrido "
        "convencional en Guanajuato, Puebla, Veracruz, Coahuila, Yucatán, "
        "Chihuahua, Baja California, Michoacán, Sonora y San Luis Potosí. Las "
        "agencias nuevas convienen en la CDMX, Estado de México, Nuevo León, "
        "Querétaro y Jalisco.\n\n"
        "**Límites:** BYD no está en los datos de INEGI, 2026 son ocho meses, "
        "la carga es una sola foto de 2026 y el cruce con ingreso es entre "
        "estados, no entre personas."
    )


    # --------------------------------------------------------
    # CONCLUSIÓN DE INVERSIÓN
    # --------------------------------------------------------

    st.subheader(
        "Conclusión de inversión: dónde, cómo y en qué orden"
    )

    # Capas de la cartera, con lo que ya se calculó arriba
    jm_capa1 = ["Ciudad de México", "México", "Nuevo León", "Jalisco", "Sinaloa"]
    jm_capa2 = ["Guanajuato", "Puebla", "Veracruz de Ignacio de la Llave",
                "Coahuila de Zaragoza", "Yucatán", "Chihuahua",
                "Baja California", "Michoacán de Ocampo", "Sonora",
                "San Luis Potosí"]

    jm_2026["capa"] = np.where(jm_2026["entidad"].isin(jm_capa1), "1. Eléctrico y enchufable",
                      np.where(jm_2026["entidad"].isin(jm_capa2), "2. Híbrido convencional",
                               "3. Todavía no"))

    # Cuántas agencias faltarían para llegar al promedio nacional de ventas por agencia
    jm_2026["agencias_faltantes"] = jm_2026["ventas_totales"] / jm_vpa_nal - jm_2026["agencias"]

    jm_resumen_capas = (
        jm_2026
        .groupby("capa")
        .agg(estados=("entidad", "count"),
             electrificados=("total_hye", "sum"),
             penetracion=("penetracion", "mean"),
             se_enchufa=("enchufa", "mean"))
        .round(1)
    )
    jm_resumen_capas["% del mercado"] = (
        jm_resumen_capas["electrificados"] / jm_2026["total_hye"].sum() * 100
    ).round(1)

    st.dataframe(jm_resumen_capas)

    st.write(
        f"**Dónde.** Los {len(jm_capa1) + len(jm_capa2)} estados de las dos "
        "primeras capas son el "
        f"{jm_2026[jm_2026['capa'] != '3. Todavía no']['total_hye'].sum() / jm_2026['total_hye'].sum() * 100:.0f} % "
        "del mercado electrificado del país. En la primera capa el producto es "
        "eléctrico puro y enchufable, porque ahí ya se enchufa más que en el "
        "promedio nacional; en la segunda, híbrido convencional, que es donde "
        "está el volumen y no hace falta cable."
    )

    jm_cobertura = jm_2026[jm_2026["agencias_faltantes"] > 0][
        ["entidad", "agencias", "ventas_por_agencia", "agencias_faltantes",
         "cambio_pen"]
    ].sort_values("agencias_faltantes", ascending=False).round(1)

    st.write(
        f"**Cómo.** El cuello de botella en los mercados grandes es cobertura, "
        f"no demanda: el promedio nacional es de {jm_vpa_nal:.0f} autos por "
        "agencia al año, y estos estados venden más que eso por punto de "
        "venta. La última columna es cuántas agencias faltarían para igualar "
        "el promedio:"
    )

    st.dataframe(jm_cobertura)

    jm_nl = jm_2026[jm_2026["entidad"] == "Nuevo León"].iloc[0]
    jm_sin_tesla_nal = jm_2026["sin_tesla_por_mil"].median()

    st.write(
        f"**Y la carga va junto con el producto.** Nuevo León está en la "
        f"primera capa pero tiene {jm_nl['sin_tesla_por_mil']:.1f} cargadores "
        "no-Tesla por cada mil autos nuevos, por debajo de la mediana nacional "
        f"({jm_sin_tesla_nal:.1f}). Quien lleve eléctricos a Monterrey tiene "
        "que llevar dónde cargarlos, o vender enchufable. Sinaloa es el caso "
        "contrario: la participación eléctrica más alta del país con agencias "
        "por debajo del promedio, así que ahí la oportunidad es de producto, "
        "no de cobertura."
    )

    jm_izq, jm_der = st.columns(2)

    with jm_izq:
        st.markdown("**Si el dinero lo pone un distribuidor**")
        st.write(
            "Eléctrico y enchufable en la primera capa; híbrido convencional en "
            "la segunda. La inversión que va primero es cobertura: en los "
            "mercados grandes cada agencia ya vende por encima del promedio "
            "nacional, así que el cuello de botella no es la demanda sino dónde "
            "comprarlos. En Nuevo León la carga es parte del costo de entrada; "
            "en Sinaloa falta producto, no puntos de venta."
        )

    with jm_der:
        st.markdown("**Si el dinero lo pone una financiera**")
        st.write(
            "El crédito conviene donde la penetración crece más rápido "
            "(Querétaro, Jalisco, Sinaloa y Nuevo León, todos por arriba del "
            f"{jm_cambio_nal:.1f} nacional). El híbrido es el producto de menor "
            "riesgo porque no depende de la red; el eléctrico puro cayó 13.9 % "
            "en 2025, así que donde la carga es baja conviene plazo más corto o "
            "enganche mayor. Financiar cargadores es un negocio aparte en los "
            "estados con demanda y poca red."
        )

    st.write(
        "**Cuándo reevaluar, para los dos.** Tres señales cambian esta "
        "recomendación: que el eléctrico puro vuelva a crecer dos años "
        "seguidos, que BYD empiece a reportar ventas por estado —vendió 45,296 "
        "unidades en 2026 sin aparecer en el registro—, y el cierre real de "
        f"2026 contra el piso estimado de {jm_cierre:,.0f} unidades."
    )


    # Cerramos las figuras para que no se acumulen en memoria
    plt.close("all")
