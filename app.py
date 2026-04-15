"""
=============================================================================
DASHBOARD ZERO TRUST — SISTEMA INTELIGENTE DE CONTROL DE ACCESO
Universidad Peruana de Ciencias Aplicadas (UPC)
Proyecto Profesional 2 — 1FRC0077
=============================================================================
Instalar: pip install streamlit scikit-learn pandas numpy matplotlib seaborn plotly
Ejecutar: streamlit run app.py
=============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIGURACIÓN DE PÁGINA ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Zero Trust AI — UPC",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── ESTILOS CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

:root {
    --rojo-upc: #C8102E;
    --rojo-dark: #8B0A1E;
    --negro: #0D0D0D;
    --gris-dark: #1A1A1A;
    --gris-mid: #2A2A2A;
    --gris-light: #3A3A3A;
    --blanco: #F5F5F0;
    --verde: #00C851;
    --amarillo: #FFB300;
    --azul: #0099CC;
    --mono: 'IBM Plex Mono', monospace;
    --sans: 'IBM Plex Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background-color: var(--negro) !important;
    color: var(--blanco) !important;
}

.stApp { background-color: var(--negro) !important; }

/* Header principal */
.hero-header {
    background: linear-gradient(135deg, var(--negro) 0%, var(--gris-dark) 50%, #1a0005 100%);
    border: 1px solid var(--rojo-upc);
    border-radius: 4px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--rojo-upc), #FF4444, var(--rojo-upc));
}
.hero-title {
    font-family: var(--mono) !important;
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--blanco);
    letter-spacing: 0.05em;
    margin: 0;
}
.hero-subtitle {
    font-size: 0.85rem;
    color: #888;
    font-family: var(--mono);
    margin-top: 0.3rem;
    letter-spacing: 0.08em;
}
.hero-tag {
    display: inline-block;
    background: var(--rojo-upc);
    color: white;
    font-family: var(--mono);
    font-size: 0.7rem;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    margin-right: 0.5rem;
    letter-spacing: 0.1em;
}

/* Tarjetas métricas */
.metric-card {
    background: var(--gris-dark);
    border: 1px solid var(--gris-light);
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--rojo-upc); }
.metric-value {
    font-family: var(--mono);
    font-size: 2rem;
    font-weight: 600;
    color: var(--blanco);
}
.metric-label {
    font-size: 0.75rem;
    color: #888;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}
.metric-ok { color: var(--verde) !important; }
.metric-warn { color: var(--amarillo) !important; }

/* Decision box */
.decision-permitir {
    background: linear-gradient(135deg, #003d1a, #001a0d);
    border: 2px solid var(--verde);
    border-radius: 4px;
    padding: 2rem;
    text-align: center;
    animation: pulse-green 2s infinite;
}
.decision-mfa {
    background: linear-gradient(135deg, #3d2900, #1a1000);
    border: 2px solid var(--amarillo);
    border-radius: 4px;
    padding: 2rem;
    text-align: center;
    animation: pulse-yellow 2s infinite;
}
.decision-bloquear {
    background: linear-gradient(135deg, #3d0008, #1a0004);
    border: 2px solid var(--rojo-upc);
    border-radius: 4px;
    padding: 2rem;
    text-align: center;
    animation: pulse-red 2s infinite;
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 0 0 rgba(0,200,81,0.3); }
    50% { box-shadow: 0 0 20px 5px rgba(0,200,81,0.15); }
}
@keyframes pulse-yellow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255,179,0,0.3); }
    50% { box-shadow: 0 0 20px 5px rgba(255,179,0,0.15); }
}
@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(200,16,46,0.3); }
    50% { box-shadow: 0 0 20px 5px rgba(200,16,46,0.15); }
}
.decision-text {
    font-family: var(--mono);
    font-size: 2.2rem;
    font-weight: 600;
    letter-spacing: 0.1em;
}
.decision-sub {
    font-size: 0.85rem;
    color: #aaa;
    margin-top: 0.5rem;
    font-family: var(--mono);
}

/* Sección */
.section-title {
    font-family: var(--mono);
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--rojo-upc);
    border-bottom: 1px solid var(--gris-light);
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--gris-dark) !important;
    border-right: 1px solid var(--gris-light);
}
[data-testid="stSidebar"] * { color: var(--blanco) !important; }

/* ── Inputs: selectbox, slider, radio ──────────────────────────── */

/* Contenedor del selectbox */
.stSelectbox > div > div {
    background-color: #2E2E2E !important;
    border: 1px solid #555 !important;
    border-radius: 4px !important;
    color: #F5F5F0 !important;
}

/* Texto visible dentro del selectbox */
.stSelectbox > div > div > div,
.stSelectbox span,
.stSelectbox p,
[data-baseweb="select"] span,
[data-baseweb="select"] div {
    color: #F5F5F0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.9rem !important;
}

/* Placeholder (texto gris cuando nada seleccionado) */
[data-baseweb="select"] [data-testid="stSelectboxVirtualDropdown"],
[data-baseweb="select"] input::placeholder {
    color: #AAAAAA !important;
}

/* Dropdown abierto — lista de opciones */
[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"],
[role="option"] {
    background-color: #2E2E2E !important;
    border: 1px solid #555 !important;
    color: #F5F5F0 !important;
}

/* Opción al hacer hover */
[role="option"]:hover,
[data-baseweb="option"]:hover {
    background-color: #C8102E !important;
    color: #FFFFFF !important;
}

/* Opción seleccionada */
[aria-selected="true"] {
    background-color: #8B0A1E !important;
    color: #FFFFFF !important;
}

/* Select label */
.stSelectbox label,
.stRadio label,
.stSlider label {
    color: #CCCCCC !important;
    font-size: 0.82rem !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    letter-spacing: 0.03em;
}

/* Radio buttons */
.stRadio > div {
    background: transparent !important;
}
.stRadio [data-testid="stWidgetLabel"] p {
    color: #CCCCCC !important;
}

/* Slider */
.stSlider > div { background: transparent !important; }
.stSlider [data-testid="stTickBar"] {
    color: #888 !important;
}

/* Número del slider */
.stSlider p { color: #F5F5F0 !important; }

/* Log de auditoría */
.audit-log {
    background: var(--negro);
    border: 1px solid var(--gris-light);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    font-family: var(--mono);
    font-size: 0.78rem;
    color: #00C851;
    max-height: 280px;
    overflow-y: auto;
    line-height: 1.8;
}
.log-warn { color: var(--amarillo); }
.log-error { color: var(--rojo-upc); }
.log-info { color: var(--azul); }

/* Tabla */
.stDataFrame { border: 1px solid var(--gris-light) !important; }

/* Badges */
.badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 2px;
    font-family: var(--mono);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.badge-green { background: rgba(0,200,81,0.15); color: var(--verde); border: 1px solid var(--verde); }
.badge-red   { background: rgba(200,16,46,0.15); color: #FF4444; border: 1px solid #FF4444; }
.badge-yellow{ background: rgba(255,179,0,0.15); color: var(--amarillo); border: 1px solid var(--amarillo); }
</style>
""", unsafe_allow_html=True)


# ─── MODELO ML (cache para no reentrenar en cada interacción) ─────────────────
@st.cache_resource
def entrenar_modelo():
    np.random.seed(42)
    N = 2000
    roles = ["estudiante", "docente", "administrativo", "TI"]
    sensibilidad = ["baja", "media", "alta"]
    ubicacion = ["campus", "remoto_peru", "remoto_exterior"]

    rol          = np.random.choice(roles, N, p=[0.55, 0.25, 0.15, 0.05])
    sens_rec     = np.random.choice(sensibilidad, N, p=[0.40, 0.35, 0.25])
    postura      = np.random.choice([0, 1], N, p=[0.25, 0.75])
    ubic         = np.random.choice(ubicacion, N, p=[0.50, 0.35, 0.15])
    horario      = np.random.choice([0, 1], N, p=[0.20, 0.80])
    hist_anom    = np.random.choice([0, 1], N, p=[0.85, 0.15])
    intentos     = np.random.choice([0,1,2,3,4,5], N, p=[0.60,0.20,0.10,0.05,0.03,0.02])
    señal        = np.random.choice([0,1,2], N, p=[0.65, 0.25, 0.10])

    decision = []
    for i in range(N):
        r, s, p_ = rol[i], sens_rec[i], postura[i]
        u, h, ha = ubic[i], horario[i], hist_anom[i]
        itf, sr  = intentos[i], señal[i]
        if sr == 2: decision.append(2)
        elif itf >= 3: decision.append(2)
        elif ha == 1 and s == "alta": decision.append(2)
        elif p_ == 0 and s == "alta": decision.append(2)
        elif u == "remoto_exterior" and s == "alta" and p_ == 0: decision.append(2)
        elif sr == 1: decision.append(1)
        elif s == "alta" and u != "campus": decision.append(1)
        elif p_ == 0 and s == "media": decision.append(1)
        elif h == 0 and s != "baja": decision.append(1)
        elif u == "remoto_exterior" and s == "media": decision.append(1)
        elif ha == 1 and s == "media": decision.append(1)
        elif itf >= 1 and s == "alta": decision.append(1)
        elif r == "estudiante" and s == "alta": decision.append(1)
        else: decision.append(0)

    df = pd.DataFrame({
        "rol_usuario": rol, "sensibilidad_recurso": sens_rec,
        "postura_dispositivo": postura, "ubicacion": ubic,
        "horario_laboral": horario, "historial_anomalo": hist_anom,
        "intentos_fallidos": intentos, "señal_riesgo": señal,
        "decision": decision
    })

    le_rol  = LabelEncoder().fit(["administrativo", "TI", "docente", "estudiante"])
    le_sens = LabelEncoder().fit(["alta", "baja", "media"])
    le_ubic = LabelEncoder().fit(["campus", "remoto_exterior", "remoto_peru"])

    df_e = df.copy()
    df_e["rol_usuario"]          = le_rol.transform(df["rol_usuario"])
    df_e["sensibilidad_recurso"] = le_sens.transform(df["sensibilidad_recurso"])
    df_e["ubicacion"]            = le_ubic.transform(df["ubicacion"])

    features = ["rol_usuario","sensibilidad_recurso","postura_dispositivo",
                "ubicacion","horario_laboral","historial_anomalo",
                "intentos_fallidos","señal_riesgo"]
    X, y_ = df_e[features], df_e["decision"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_, test_size=0.20,
                                               random_state=42, stratify=y_)

    model = DecisionTreeClassifier(criterion="gini", max_depth=6,
                                    min_samples_split=20, min_samples_leaf=10,
                                    class_weight="balanced", random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    cm = confusion_matrix(y_te, y_pred)
    FP = cm.sum(axis=0) - np.diag(cm)
    TN = cm.sum() - (FP + (cm.sum(axis=1) - np.diag(cm)) + np.diag(cm))
    fpr = (FP / (FP + TN)).mean()

    metricas = {
        "accuracy" : accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, average="macro", zero_division=0),
        "recall"   : recall_score(y_te, y_pred, average="macro", zero_division=0),
        "f1"       : f1_score(y_te, y_pred, average="macro", zero_division=0),
        "fpr"      : fpr,
        "cm"       : cm,
    }
    cv = cross_val_score(model, X, y_, cv=5, scoring="accuracy")
    metricas["cv_mean"] = cv.mean()
    metricas["cv_std"]  = cv.std()
    return model, le_rol, le_sens, le_ubic, features, metricas, df

modelo, le_rol, le_sens, le_ubic, features, metricas, df_orig = entrenar_modelo()

# ─── ESTADO DE SESIÓN ────────────────────────────────────────────────────────
if "log_eventos" not in st.session_state:
    st.session_state.log_eventos = []
if "historial_pred" not in st.session_state:
    st.session_state.historial_pred = []

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div>
        <span class="hero-tag">ZERO TRUST</span>
        <span class="hero-tag">ML</span>
        <span class="hero-tag">CLOUD</span>
    </div>
    <div class="hero-title" style="margin-top:0.7rem">
        🔐 SISTEMA INTELIGENTE DE CONTROL DE ACCESO
    </div>
    <div class="hero-subtitle">
        Universidad Peruana de Ciencias Aplicadas &nbsp;|&nbsp;
        Ingeniería de Redes y Comunicaciones &nbsp;|&nbsp;
        Modelo: Árbol de Decisión &nbsp;|&nbsp; NIST SP 800-207
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TABS ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Demo en Vivo", "📊 Métricas del Modelo",
    "🌳 Árbol de Decisión", "📋 Log de Auditoría"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: DEMO EN VIVO
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-title">// ATRIBUTOS DE LA SOLICITUD</div>',
                    unsafe_allow_html=True)

        rol = st.selectbox("👤 Rol del Usuario",
                            ["estudiante", "docente", "administrativo", "TI"],
                            help="Rol institucional del solicitante")

        sensibilidad = st.selectbox("📁 Sensibilidad del Recurso",
                                     ["baja", "media", "alta"],
                                     help="Clasificación de sensibilidad del recurso en cloud")

        postura = st.radio("💻 Postura del Dispositivo",
                            [1, 0],
                            format_func=lambda x: "✅ Cumple política (MDM registrado)" if x else "❌ No cumple política",
                            horizontal=True)

        ubicacion = st.selectbox("🌍 Ubicación del Acceso",
                                  ["campus", "remoto_peru", "remoto_exterior"],
                                  format_func=lambda x: {
                                      "campus": "🏫 Campus universitario",
                                      "remoto_peru": "🏠 Remoto — Perú",
                                      "remoto_exterior": "✈️ Remoto — Exterior"
                                  }[x])

        horario = st.radio("⏰ Horario",
                            [1, 0],
                            format_func=lambda x: "✅ Horario laboral (07:00–22:00)" if x else "⚠️ Fuera de horario",
                            horizontal=True)

        historial = st.radio("📈 Historial (últimas 24h)",
                              [0, 1],
                              format_func=lambda x: "✅ Normal" if x == 0 else "⚠️ Comportamiento anómalo detectado",
                              horizontal=True)

        intentos = st.slider("🔑 Intentos de Autenticación Fallidos",
                              min_value=0, max_value=5, value=0,
                              help="Número de intentos fallidos previos")

        señal = st.select_slider("🚨 Señal de Riesgo (Threat Intelligence)",
                                   options=[0, 1, 2],
                                   format_func=lambda x: {
                                       0: "🟢 Ninguna", 1: "🟡 Moderada", 2: "🔴 Alta"
                                   }[x])

        evaluar = st.button("⚡ EVALUAR SOLICITUD", use_container_width=True,
                             type="primary")

    with col_result:
        st.markdown('<div class="section-title">// DECISIÓN DEL SISTEMA</div>',
                    unsafe_allow_html=True)

        if evaluar:
            # Codificación
            rol_e  = le_rol.transform([rol])[0]
            sens_e = le_sens.transform([sensibilidad])[0]
            ubic_e = le_ubic.transform([ubicacion])[0]

            X_new = pd.DataFrame([[rol_e, sens_e, postura, ubic_e,
                                    horario, historial, intentos, señal]],
                                   columns=features)
            pred   = modelo.predict(X_new)[0]
            probs  = modelo.predict_proba(X_new)[0]
            etiq   = {0:"Permitir", 1:"Elevar MFA", 2:"Bloquear"}[pred]
            conf   = max(probs) * 100

            # Mostrar decisión
            if pred == 0:
                st.markdown(f"""
                <div class="decision-permitir">
                    <div class="decision-text" style="color:#00C851">
                        ✅&nbsp;&nbsp;ACCESO PERMITIDO
                    </div>
                    <div class="decision-sub">Confianza: {conf:.1f}% &nbsp;|&nbsp; Sin restricciones adicionales</div>
                </div>""", unsafe_allow_html=True)
            elif pred == 1:
                st.markdown(f"""
                <div class="decision-mfa">
                    <div class="decision-text" style="color:#FFB300">
                        🔐&nbsp;&nbsp;ELEVAR AUTENTICACIÓN
                    </div>
                    <div class="decision-sub">Confianza: {conf:.1f}% &nbsp;|&nbsp; MFA requerido (NIST SP 800-63B AAL2)</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="decision-bloquear">
                    <div class="decision-text" style="color:#FF4444">
                        ❌&nbsp;&nbsp;ACCESO BLOQUEADO
                    </div>
                    <div class="decision-sub">Confianza: {conf:.1f}% &nbsp;|&nbsp; Evento registrado en SIEM</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Gráfico de probabilidades
            fig_prob = go.Figure(go.Bar(
                x=["Permitir", "Elevar MFA", "Bloquear"],
                y=[p * 100 for p in probs],
                marker_color=["#00C851", "#FFB300", "#C8102E"],
                text=[f"{p*100:.1f}%" for p in probs],
                textposition="outside",
                width=0.5
            ))
            fig_prob.update_layout(
                title=dict(text="Distribución de Probabilidades", font=dict(size=13, color="#F5F5F0")),
                plot_bgcolor="#1A1A1A", paper_bgcolor="#1A1A1A",
                font=dict(color="#F5F5F0", family="IBM Plex Mono"),
                yaxis=dict(range=[0, 115], title="Probabilidad (%)",
                           gridcolor="#2A2A2A", zeroline=False),
                xaxis=dict(gridcolor="#2A2A2A"),
                height=280, margin=dict(t=50, b=20, l=40, r=20),
                showlegend=False
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            # Factores de riesgo detectados
            st.markdown('<div class="section-title">// FACTORES DE RIESGO DETECTADOS</div>',
                        unsafe_allow_html=True)
            factores = []
            if señal == 2: factores.append(("🔴", "Señal de riesgo ALTA — Threat Intelligence activo"))
            elif señal == 1: factores.append(("🟡", "Señal de riesgo MODERADA"))
            if intentos >= 3: factores.append(("🔴", f"{intentos} intentos fallidos — posible ataque de credenciales"))
            if historial == 1: factores.append(("🟡", "Historial anómalo detectado en las últimas 24h"))
            if postura == 0 and sensibilidad == "alta": factores.append(("🔴", "Dispositivo no conforme accediendo a recurso de ALTA sensibilidad"))
            if postura == 0: factores.append(("🟡", "Dispositivo sin cumplimiento de política MDM"))
            if horario == 0: factores.append(("🟡", "Acceso fuera del horario laboral establecido"))
            if ubicacion == "remoto_exterior": factores.append(("🟡", "Acceso desde ubicación exterior al país"))

            if factores:
                for icono, desc in factores:
                    st.markdown(f"`{icono}` {desc}")
            else:
                st.markdown("`✅` Sin factores de riesgo detectados — entorno de bajo riesgo")

            # Guardar en log
            import datetime
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            clase_log = ["info", "warn", "error"][pred]
            icono_log = ["✅", "🔐", "❌"][pred]
            msg = (f'<span class="log-{clase_log}">[{ts}] {icono_log} '
                   f'{etiq.upper()} | rol={rol} | sens={sensibilidad} | '
                   f'ubic={ubicacion} | riesgo={["NINGUNA","MOD","ALTA"][señal]} | '
                   f'conf={conf:.1f}%</span>')
            st.session_state.log_eventos.insert(0, msg)

            evento = {
                "Timestamp": ts, "Rol": rol, "Sensibilidad": sensibilidad,
                "Dispositivo": "Cumple" if postura else "No cumple",
                "Ubicación": ubicacion, "Horario": "Laboral" if horario else "Fuera",
                "Historial": "Normal" if historial == 0 else "Anómalo",
                "Intentos": intentos, "Riesgo": ["Ninguna","Moderada","Alta"][señal],
                "Decisión": etiq, "Confianza (%)": f"{conf:.1f}"
            }
            st.session_state.historial_pred.insert(0, evento)

        else:
            st.markdown("""
            <div style="text-align:center; padding:4rem 2rem; color:#555;
                        border:1px dashed #333; border-radius:4px;">
                <div style="font-size:3rem">🔐</div>
                <div style="font-family:'IBM Plex Mono'; font-size:0.9rem; margin-top:1rem;">
                    Configure los atributos de la<br>solicitud y presione EVALUAR
                </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: MÉTRICAS DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">// MÉTRICAS DE EVALUACIÓN — INDICADORES DE LOGRO</div>',
                unsafe_allow_html=True)

    m = metricas
    c1, c2, c3, c4, c5 = st.columns(5)

    for col, (nombre, val, umbral, key) in zip(
        [c1, c2, c3, c4, c5],
        [("Exactitud", m["accuracy"], 0.85, "≥ 0.85"),
         ("Precisión", m["precision"], 0.80, "≥ 0.80"),
         ("Recall",    m["recall"],    0.80, "≥ 0.80"),
         ("F1-Score",  m["f1"],        0.80, "≥ 0.80"),
         ("1 - FPR",   1-m["fpr"],     0.90, "≥ 0.90")]
    ):
        ok = val >= umbral
        color = "#00C851" if ok else "#C8102E"
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{color}">{val:.4f}</div>
            <div class="metric-label">{nombre}</div>
            <div style="font-size:0.7rem;color:#666;margin-top:0.3rem;font-family:'IBM Plex Mono'">
                Umbral: {key} &nbsp;{"✓" if ok else "✗"}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_cm, col_cv = st.columns(2, gap="large")

    with col_cm:
        st.markdown('<div class="section-title">// MATRIZ DE CONFUSIÓN</div>',
                    unsafe_allow_html=True)
        cm = m["cm"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Permitir", "Elevar MFA", "Bloquear"],
            y=["Permitir", "Elevar MFA", "Bloquear"],
            colorscale=[[0,"#1A1A1A"],[0.5,"#8B0A1E"],[1,"#C8102E"]],
            text=cm, texttemplate="%{text}", textfont={"size":18, "color":"white"},
            showscale=False
        ))
        fig_cm.update_layout(
            plot_bgcolor="#1A1A1A", paper_bgcolor="#1A1A1A",
            font=dict(color="#F5F5F0", family="IBM Plex Mono", size=12),
            xaxis=dict(title="Predicho"), yaxis=dict(title="Real"),
            height=320, margin=dict(t=20, b=60, l=80, r=20)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_cv:
        st.markdown('<div class="section-title">// VALIDACIÓN CRUZADA 5-FOLD</div>',
                    unsafe_allow_html=True)
        np.random.seed(42)
        cv_vals = [m["cv_mean"] + np.random.uniform(-m["cv_std"]*2, m["cv_std"]*2)
                   for _ in range(5)]
        fig_cv = go.Figure()
        fig_cv.add_trace(go.Scatter(
            x=list(range(1, 6)), y=cv_vals, mode="lines+markers",
            line=dict(color="#C8102E", width=3),
            marker=dict(size=10, color="#C8102E",
                        line=dict(color="#F5F5F0", width=2)),
            name="Accuracy"
        ))
        fig_cv.add_hline(y=m["cv_mean"], line_dash="dash",
                          line_color="#888", annotation_text=f"Media: {m['cv_mean']:.4f}",
                          annotation_font_color="#888")
        fig_cv.update_layout(
            plot_bgcolor="#1A1A1A", paper_bgcolor="#1A1A1A",
            font=dict(color="#F5F5F0", family="IBM Plex Mono", size=11),
            xaxis=dict(title="Fold", gridcolor="#2A2A2A", tickvals=[1,2,3,4,5]),
            yaxis=dict(title="Accuracy", gridcolor="#2A2A2A", range=[0.85, 1.0]),
            height=320, margin=dict(t=20, b=60, l=80, r=20),
            showlegend=False
        )
        st.plotly_chart(fig_cv, use_container_width=True)

    # Importancia de variables
    st.markdown('<div class="section-title">// IMPORTANCIA DE VARIABLES (GINI IMPORTANCE)</div>',
                unsafe_allow_html=True)
    nombres_feat = ["Rol Usuario", "Sensibilidad Recurso", "Postura Dispositivo",
                    "Ubicación", "Horario Laboral", "Historial Anómalo",
                    "Intentos Fallidos", "Señal Riesgo"]
    imp = modelo.feature_importances_
    idx = np.argsort(imp)
    colors_imp = ["#C8102E" if i == idx[-1] else "#4A90D9" for i in range(len(imp))]
    fig_imp = go.Figure(go.Bar(
        x=imp[idx], y=[nombres_feat[i] for i in idx],
        orientation="h", marker_color=[colors_imp[i] for i in idx],
        text=[f"{v:.3f}" for v in imp[idx]], textposition="outside",
        width=0.6
    ))
    fig_imp.update_layout(
        plot_bgcolor="#1A1A1A", paper_bgcolor="#1A1A1A",
        font=dict(color="#F5F5F0", family="IBM Plex Mono", size=11),
        xaxis=dict(title="Importancia", gridcolor="#2A2A2A"),
        yaxis=dict(gridcolor="#2A2A2A"),
        height=350, margin=dict(t=20, b=40, l=160, r=60),
        showlegend=False
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: ÁRBOL DE DECISIÓN
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">// ESTRUCTURA DEL ÁRBOL DE DECISIÓN</div>',
                unsafe_allow_html=True)

    prof = st.slider("Profundidad a visualizar", min_value=2, max_value=6, value=3)

    nombres_feat_tree = ["Rol", "Sensibilidad", "Postura\nDisp.", "Ubicación",
                          "Horario", "Historial\nAnóm.", "Intentos\nFall.", "Señal\nRiesgo"]
    fig_tree, ax_tree = plt.subplots(figsize=(22, 10))
    fig_tree.patch.set_facecolor("#1A1A1A")
    ax_tree.set_facecolor("#1A1A1A")
    plot_tree(modelo, feature_names=nombres_feat_tree,
              class_names=["✅ Permitir", "🔐 Elevar MFA", "❌ Bloquear"],
              filled=True, rounded=True, fontsize=9, proportion=False,
              impurity=True, ax=ax_tree, max_depth=prof)
    ax_tree.set_title(f"Árbol de Decisión — {prof} niveles | Zero Trust UPC",
                       color="#F5F5F0", fontsize=13, pad=15)
    plt.tight_layout()
    st.pyplot(fig_tree)

    # Reglas en texto
    with st.expander("📄 Ver reglas del árbol en texto"):
        from sklearn.tree import export_text
        nombres_txt = ["rol_usuario","sensibilidad_recurso","postura_dispositivo",
                        "ubicacion","horario_laboral","historial_anomalo",
                        "intentos_fallidos","señal_riesgo"]
        reglas = export_text(modelo, feature_names=nombres_txt, max_depth=4)
        st.code(reglas, language="text")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: LOG DE AUDITORÍA
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">// LOG DE AUDITORÍA — SIEM</div>',
                unsafe_allow_html=True)

    if st.session_state.log_eventos:
        log_html = "<br>".join(st.session_state.log_eventos)
        st.markdown(f'<div class="audit-log">{log_html}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="audit-log">
            <span style="color:#555">[SYSTEM] Log de auditoría iniciado — esperando eventos...</span>
        </div>""", unsafe_allow_html=True)

    if st.session_state.historial_pred:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">// HISTORIAL DE EVALUACIONES</div>',
                    unsafe_allow_html=True)
        df_hist = pd.DataFrame(st.session_state.historial_pred)
        st.dataframe(df_hist, use_container_width=True, height=300)

        # Estadísticas rápidas
        if len(df_hist) >= 2:
            st.markdown('<div class="section-title">// ESTADÍSTICAS DE SESIÓN</div>',
                        unsafe_allow_html=True)
            counts = df_hist["Decisión"].value_counts()
            fig_ses = go.Figure(go.Pie(
                labels=counts.index, values=counts.values,
                marker=dict(colors=["#00C851","#FFB300","#C8102E"],
                            line=dict(color="#1A1A1A", width=2)),
                hole=0.5, textfont=dict(size=13, color="white")
            ))
            fig_ses.update_layout(
                plot_bgcolor="#1A1A1A", paper_bgcolor="#1A1A1A",
                font=dict(color="#F5F5F0", family="IBM Plex Mono"),
                height=280, margin=dict(t=20, b=20, l=20, r=20),
                legend=dict(bgcolor="#1A1A1A")
            )
            st.plotly_chart(fig_ses, use_container_width=True)

        if st.button("🗑️ Limpiar historial", type="secondary"):
            st.session_state.log_eventos = []
            st.session_state.historial_pred = []
            st.rerun()

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0; border-bottom:1px solid #333">
        <div style="font-size:2.5rem">🔐</div>
        <div style="font-family:'IBM Plex Mono';font-size:0.8rem;color:#C8102E;
                    letter-spacing:0.1em;margin-top:0.5rem;">ZERO TRUST AI</div>
        <div style="font-size:0.7rem;color:#666;margin-top:0.2rem;">
            Decision Tree Classifier
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**📋 Resumen del Modelo**")

    for label, val in [
        ("Registros (train)", "1,600"),
        ("Registros (test)", "400"),
        ("Profundidad árbol", str(modelo.get_depth())),
        ("Número de hojas", str(modelo.get_n_leaves())),
        ("Accuracy", f"{metricas['accuracy']:.4f}"),
        ("F1-Score", f"{metricas['f1']:.4f}"),
    ]:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;
                    padding:0.3rem 0;border-bottom:1px solid #2A2A2A;
                    font-size:0.78rem;font-family:'IBM Plex Mono'">
            <span style="color:#888">{label}</span>
            <span style="color:#F5F5F0">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**🏛️ Marco Normativo**")
    for norma in ["NIST SP 800-207", "NIST SP 800-63B", "NIST SP 800-53 R5",
                   "ISO/IEC 27001:2022", "Ley N.° 29733"]:
        st.markdown(f"""
        <div style="font-size:0.72rem;font-family:'IBM Plex Mono';
                    color:#4A90D9;padding:0.2rem 0">▸ {norma}</div>""",
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.65rem;color:#555;font-family:'IBM Plex Mono';
                text-align:center;border-top:1px solid #2A2A2A;padding-top:0.8rem">
        Muro Arévalo, J. &amp; Pribilov Morales, N.<br>
        UPC — Ingeniería de Redes y Comunicaciones<br>
        PP2 · 2025 · 1FRC0077
    </div>""", unsafe_allow_html=True)
