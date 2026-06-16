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
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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

/* Inputs */
.stSelectbox > div > div {
    background-color: #2E2E2E !important;
    border: 1px solid #555 !important;
    border-radius: 4px !important;
    color: #F5F5F0 !important;
}
.stSelectbox > div > div > div,
.stSelectbox span,
.stSelectbox p,
[data-baseweb="select"] span,
[data-baseweb="select"] div {
    color: #F5F5F0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.9rem !important;
}
[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"],
[role="option"] {
    background-color: #2E2E2E !important;
    border: 1px solid #555 !important;
    color: #F5F5F0 !important;
}
[role="option"]:hover,
[data-baseweb="option"]:hover {
    background-color: #C8102E !important;
    color: #FFFFFF !important;
}
[aria-selected="true"] {
    background-color: #8B0A1E !important;
    color: #FFFFFF !important;
}
.stSelectbox label,
.stRadio label,
.stSlider label {
    color: #CCCCCC !important;
    font-size: 0.82rem !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    letter-spacing: 0.03em;
}
.stRadio > div { background: transparent !important; }
.stRadio [data-testid="stWidgetLabel"] p { color: #CCCCCC !important; }
.stSlider > div { background: transparent !important; }
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


# ─── MODELO ML ───────────────────────────────────────────────────────────────
# Dataset generado con etiquetado determinístico basado en reglas Zero Trust.
# Fuente: Rose et al. (2020) NIST SP 800-207; Wang et al. (2025);
#         Lukaseder et al. (2020). random_state=42 garantiza reproducibilidad.
@st.cache_resource
def entrenar_modelo():
    rng = np.random.RandomState(42)
    N   = 2000

    rol               = rng.randint(0, 4, N)   # 0=estudiante,1=docente,2=admin,3=ti
    sensibilidad      = rng.randint(0, 3, N)   # 0=baja,1=media,2=alta
    postura           = rng.randint(0, 2, N)   # 0=no conforme,1=conforme
    ubicacion         = rng.randint(0, 3, N)   # 0=campus,1=remoto_peru,2=remoto_ext
    horario           = rng.randint(0, 2, N)   # 0=fuera,1=en horario
    historial         = rng.randint(0, 2, N)   # 0=normal,1=anomalo
    intentos_fallidos = rng.randint(0, 6, N)   # 0-5
    senal_riesgo      = rng.randint(0, 3, N)   # 0=ninguna,1=moderada,2=alta

    # Etiquetado por reglas Zero Trust (NIST SP 800-207 - Rose et al., 2020)
    # BLOQUEAR: señal alta o intentos >= 4 (fail-secure)
    # PERMITIR: señal nula + postura conforme + historial normal
    # ELEVAR MFA: resto de casos
    etiqueta = np.where(
        (senal_riesgo == 2) | (intentos_fallidos >= 4), 2,
        np.where(
            (senal_riesgo == 0) & (postura == 1) & (historial == 0), 0,
            1
        )
    )
    # Ruido controlado 3% para simular casos límite reales
    mask     = rng.random(N) < 0.03
    etiqueta = np.where(mask, rng.randint(0, 3, N), etiqueta)

    df = pd.DataFrame({
        "rol":                rol,
        "sensibilidad":       sensibilidad,
        "postura_dispositivo": postura,
        "ubicacion":          ubicacion,
        "horario_laboral":    horario,
        "historial_anomalo":  historial,
        "intentos_fallidos":  intentos_fallidos,
        "senal_riesgo":       senal_riesgo,
        "decision":           etiqueta
    })

    features = ["rol", "sensibilidad", "postura_dispositivo", "ubicacion",
                "horario_laboral", "historial_anomalo", "intentos_fallidos", "senal_riesgo"]
    X  = df[features]
    y_ = df["decision"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_, test_size=0.20, random_state=42, stratify=y_)

    # Hiperparámetros documentados en sección 4.3.3.9 del Capítulo 4
    # (Mahbooba et al., 2021; Funmi et al., 2025)
    model = DecisionTreeClassifier(
        criterion="gini", max_depth=6,
        min_samples_split=20, min_samples_leaf=10,
        class_weight="balanced", random_state=42
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    # Métricas con promedio 'weighted' para clases desbalanceadas
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_te, y_pred,    average="weighted", zero_division=0)
    f1   = f1_score(y_te, y_pred,         average="weighted", zero_division=0)

    cm = confusion_matrix(y_te, y_pred)
    fpr_lista = []
    for i in range(cm.shape[0]):
        FP = cm[:, i].sum() - cm[i, i]
        TN = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fpr_lista.append(FP / (FP + TN) if (FP + TN) > 0 else 0)
    fpr = float(np.mean(fpr_lista))

    # Validación cruzada 5-fold estratificada
    cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_vals  = cross_val_score(model, X, y_, cv=cv, scoring="accuracy")

    importancias = pd.Series(
        model.feature_importances_, index=features
    ).sort_values(ascending=False)

    metricas = {
        "accuracy" : acc,
        "precision": prec,
        "recall"   : rec,
        "f1"       : f1,
        "fpr"      : fpr,
        "cm"       : cm,
        "cv_mean"  : float(cv_vals.mean()),
        "cv_std"   : float(cv_vals.std()),
        "cv_vals"  : cv_vals.tolist(),
    }
    return model, features, metricas, importancias, X, y_

modelo, features, metricas, importancias, X_full, y_full = entrenar_modelo()

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

        # Escenarios predefinidos documentados en Tabla 44 del Capítulo 4
        st.markdown("**Escenarios documentados (Tabla 44 - Capítulo 4):**")
        col_e1, col_e2, col_e3, col_e4 = st.columns(4)
        esc_sel = None
        with col_e1:
            if st.button("E1 Estudiante", use_container_width=True):
                esc_sel = 1
        with col_e2:
            if st.button("E2 Docente", use_container_width=True):
                esc_sel = 2
        with col_e3:
            if st.button("E3 Credential", use_container_width=True):
                esc_sel = 3
        with col_e4:
            if st.button("E4 TI", use_container_width=True):
                esc_sel = 4

        escenarios_def = {
            1: {"rol": 0, "sens": 0, "postura": 1, "ubi": 0, "horario": 1, "hist": 0, "intentos": 0, "senal": 0},
            2: {"rol": 1, "sens": 2, "postura": 0, "ubi": 2, "horario": 0, "hist": 0, "intentos": 0, "senal": 2},
            3: {"rol": 2, "sens": 1, "postura": 0, "ubi": 1, "horario": 1, "hist": 0, "intentos": 4, "senal": 2},
            4: {"rol": 3, "sens": 2, "postura": 1, "ubi": 0, "horario": 0, "hist": 1, "intentos": 0, "senal": 1},
        }
        defs = escenarios_def.get(esc_sel, escenarios_def[1]) if esc_sel else {
            "rol": 0, "sens": 0, "postura": 1, "ubi": 0,
            "horario": 1, "hist": 0, "intentos": 0, "senal": 0
        }

        st.divider()

        rol_val = st.selectbox("👤 Rol del Usuario",
                    [0, 1, 2, 3],
                    index=defs["rol"],
                    format_func=lambda x: ["Estudiante (0)", "Docente (1)",
                                            "Administrativo (2)", "TI (3)"][x])

        sens_val = st.selectbox("📁 Sensibilidad del Recurso",
                     [0, 1, 2],
                     index=defs["sens"],
                     format_func=lambda x: ["Baja (0) — datos públicos",
                                             "Media (1) — datos académicos",
                                             "Alta (2) — datos financieros/personales"][x])

        postura_val = st.radio("💻 Postura del Dispositivo",
                        [1, 0],
                        index=0 if defs["postura"] == 1 else 1,
                        format_func=lambda x: "✅ Cumple política (MDM registrado)" if x else "❌ No cumple política",
                        horizontal=True)

        ubi_val = st.selectbox("🌍 Ubicación del Acceso",
                    [0, 1, 2],
                    index=defs["ubi"],
                    format_func=lambda x: ["🏫 Campus universitario",
                                            "🏠 Remoto — Perú",
                                            "✈️ Remoto — Exterior"][x])

        horario_val = st.radio("⏰ Horario",
                        [1, 0],
                        index=0 if defs["horario"] == 1 else 1,
                        format_func=lambda x: "✅ Horario laboral (07:00–22:00)" if x else "⚠️ Fuera de horario",
                        horizontal=True)

        hist_val = st.radio("📈 Historial (últimas 24h)",
                     [0, 1],
                     index=defs["hist"],
                     format_func=lambda x: "✅ Normal" if x == 0 else "⚠️ Comportamiento anómalo detectado",
                     horizontal=True)

        intentos_val = st.slider("🔑 Intentos de Autenticación Fallidos",
                          min_value=0, max_value=5, value=defs["intentos"],
                          help="4+ activa bloqueo inmediato (regla fail-secure NIST SP 800-207)")

        senal_val = st.select_slider("🚨 Señal de Riesgo (Threat Intelligence)",
                          options=[0, 1, 2],
                          value=defs["senal"],
                          format_func=lambda x: {
                              0: "🟢 Ninguna", 1: "🟡 Moderada", 2: "🔴 Alta"
                          }[x])

        evaluar = st.button("⚡ EVALUAR SOLICITUD", use_container_width=True, type="primary")

    with col_result:
        st.markdown('<div class="section-title">// DECISIÓN DEL SISTEMA</div>',
                    unsafe_allow_html=True)

        if evaluar:
            # Calcular Trust Score R = 0.35Ru + 0.25Rd + 0.20Rc + 0.20Rr
            # (Rose et al., 2020; Wang et al., 2025; Lukaseder et al., 2020)
            Ru = hist_val * 0.6 + (intentos_val / 5) * 0.4
            Rd = (1 - postura_val) * 0.7 + (ubi_val / 2) * 0.3
            Rc = (1 - horario_val) * 0.3 + (sens_val / 2) * 0.7
            Rr = senal_val / 2
            R  = round(0.35 * Ru + 0.25 * Rd + 0.20 * Rc + 0.20 * Rr, 3)

            # Predicción del modelo ML
            X_new = pd.DataFrame([[rol_val, sens_val, postura_val, ubi_val,
                                    horario_val, hist_val, intentos_val, senal_val]],
                                   columns=features)
            pred  = modelo.predict(X_new)[0]
            probs = modelo.predict_proba(X_new)[0]
            conf  = max(probs) * 100

            etiq_map = {0: "Permitir", 1: "Elevar MFA", 2: "Bloquear"}
            etiq = etiq_map[pred]

            # Nivel AAL según NIST SP 800-63B
            aal_map = {0: "AAL1", 1: "AAL3" if sens_val == 2 else "AAL2", 2: "N/A"}
            aal = aal_map[pred]
            token_map = {0: "4 horas", 1: "15 min" if sens_val == 2 else "30 min", 2: "—"}
            token = token_map[pred]

            if pred == 0:
                st.markdown(f"""
                <div class="decision-permitir">
                    <div class="decision-text" style="color:#00C851">
                        ✅&nbsp;&nbsp;ACCESO PERMITIDO
                    </div>
                    <div class="decision-sub">
                        Trust Score R = {R} &nbsp;|&nbsp; {aal} &nbsp;|&nbsp; Token: {token}
                    </div>
                </div>""", unsafe_allow_html=True)
            elif pred == 1:
                st.markdown(f"""
                <div class="decision-mfa">
                    <div class="decision-text" style="color:#FFB300">
                        🔐&nbsp;&nbsp;ELEVAR AUTENTICACIÓN
                    </div>
                    <div class="decision-sub">
                        Trust Score R = {R} &nbsp;|&nbsp; {aal} requerido &nbsp;|&nbsp; Token: {token}
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                motivo = "Regla fail-secure: intentos >= 4" if intentos_val >= 4 else f"R = {R} ≥ 0.70"
                st.markdown(f"""
                <div class="decision-bloquear">
                    <div class="decision-text" style="color:#FF4444">
                        ❌&nbsp;&nbsp;ACCESO BLOQUEADO
                    </div>
                    <div class="decision-sub">
                        Trust Score R = {R} &nbsp;|&nbsp; {motivo} &nbsp;|&nbsp; Evento en SIEM
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Desglose del Trust Score
            st.markdown(f"""
            <div style="font-family:'IBM Plex Mono';font-size:0.78rem;
                        background:#1A1A1A;border:1px solid #3A3A3A;
                        border-radius:4px;padding:1rem;line-height:2;color:#888">
                <span style="color:#C8102E">R</span> = 
                0.35 × Ru({Ru:.3f}) + 0.25 × Rd({Rd:.3f}) + 
                0.20 × Rc({Rc:.3f}) + 0.20 × Rr({Rr:.3f})<br>
                <span style="color:#C8102E">R</span> = <b style="color:#F5F5F0">{R:.3f}</b>
                &nbsp;|&nbsp; Umbral PERMITIR &lt; 0.30 &nbsp;|&nbsp; 
                Umbral BLOQUEAR ≥ 0.70<br>
                Norma: NIST SP 800-207 (Rose et al., 2020)
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Gráfico probabilidades
            fig_prob = go.Figure(go.Bar(
                x=["Permitir", "Elevar MFA", "Bloquear"],
                y=[p * 100 for p in probs],
                marker_color=["#00C851", "#FFB300", "#C8102E"],
                text=[f"{p*100:.1f}%" for p in probs],
                textposition="outside",
                width=0.5
            ))
            fig_prob.update_layout(
                title=dict(text="Distribución de Probabilidades (Modelo CART)",
                           font=dict(size=13, color="#F5F5F0")),
                plot_bgcolor="#1A1A1A", paper_bgcolor="#1A1A1A",
                font=dict(color="#F5F5F0", family="IBM Plex Mono"),
                yaxis=dict(range=[0, 115], title="Probabilidad (%)",
                           gridcolor="#2A2A2A", zeroline=False),
                xaxis=dict(gridcolor="#2A2A2A"),
                height=280, margin=dict(t=50, b=20, l=40, r=20),
                showlegend=False
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            # Factores de riesgo
            st.markdown('<div class="section-title">// FACTORES DE RIESGO DETECTADOS</div>',
                        unsafe_allow_html=True)
            factores = []
            if senal_val == 2:   factores.append(("🔴", "Señal de riesgo ALTA — Threat Intelligence activo"))
            elif senal_val == 1: factores.append(("🟡", "Señal de riesgo MODERADA"))
            if intentos_val >= 4: factores.append(("🔴", f"{intentos_val} intentos fallidos — activó regla fail-secure (NIST SP 800-207 tenet 5)"))
            elif intentos_val >= 2: factores.append(("🟡", f"{intentos_val} intentos fallidos — patrón sospechoso"))
            if hist_val == 1:    factores.append(("🟡", "Historial anómalo detectado en las últimas 24h"))
            if postura_val == 0 and sens_val == 2: factores.append(("🔴", "Dispositivo no conforme accediendo a recurso de ALTA sensibilidad"))
            elif postura_val == 0: factores.append(("🟡", "Dispositivo sin cumplimiento de política MDM"))
            if horario_val == 0: factores.append(("🟡", "Acceso fuera del horario laboral establecido"))
            if ubi_val == 2:     factores.append(("🟡", "Acceso desde ubicación exterior al país"))

            if factores:
                for icono, desc in factores:
                    st.markdown(f"`{icono}` {desc}")
            else:
                st.markdown("`✅` Sin factores de riesgo detectados — entorno de bajo riesgo")

            # Log
            import datetime
            ts        = datetime.datetime.now().strftime("%H:%M:%S")
            clase_log = ["info", "warn", "error"][pred]
            icono_log = ["✅", "🔐", "❌"][pred]
            roles_txt = ["estudiante", "docente", "administrativo", "ti"]
            senal_txt = ["NINGUNA", "MOD", "ALTA"]
            msg = (f'<span class="log-{clase_log}">[{ts}] {icono_log} '
                   f'{etiq.upper()} | rol={roles_txt[rol_val]} | '
                   f'trust_score={R} | aal={aal} | '
                   f'riesgo={senal_txt[senal_val]} | conf={conf:.1f}%</span>')
            st.session_state.log_eventos.insert(0, msg)

            evento = {
                "Timestamp": ts,
                "Rol": roles_txt[rol_val],
                "Sensibilidad": ["baja","media","alta"][sens_val],
                "Dispositivo": "Cumple" if postura_val else "No cumple",
                "Ubicación": ["campus","remoto_peru","remoto_exterior"][ubi_val],
                "Horario": "Laboral" if horario_val else "Fuera",
                "Historial": "Normal" if hist_val == 0 else "Anómalo",
                "Intentos": intentos_val,
                "Señal": senal_txt[senal_val],
                "Trust Score R": R,
                "Decisión": etiq,
                "AAL": aal,
                "Confianza (%)": f"{conf:.1f}"
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

    # Mostrar FPR directamente (no 1-FPR) con umbral <= 0.10
    for col, (nombre, val, umbral, tipo, key) in zip(
        [c1, c2, c3, c4, c5],
        [("Exactitud",  m["accuracy"],  0.85, ">=", "≥ 0.85"),
         ("Precisión",  m["precision"], 0.80, ">=", "≥ 0.80"),
         ("Recall",     m["recall"],    0.80, ">=", "≥ 0.80"),
         ("F1-Score",   m["f1"],        0.80, ">=", "≥ 0.80"),
         ("FPR",        m["fpr"],       0.10, "<=", "≤ 0.10")]
    ):
        ok    = (val >= umbral) if tipo == ">=" else (val <= umbral)
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
        cv_vals_plot = m["cv_vals"]
        fig_cv = go.Figure()
        fig_cv.add_trace(go.Scatter(
            x=list(range(1, 6)), y=cv_vals_plot, mode="lines+markers",
            line=dict(color="#C8102E", width=3),
            marker=dict(size=10, color="#C8102E",
                        line=dict(color="#F5F5F0", width=2)),
            name="Accuracy"
        ))
        fig_cv.add_hline(
            y=m["cv_mean"], line_dash="dash", line_color="#888",
            annotation_text=f"Media: {m['cv_mean']:.4f} ± {m['cv_std']:.4f}",
            annotation_font_color="#888"
        )
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
    nombres_feat_display = {
        "rol": "Rol Usuario",
        "sensibilidad": "Sensibilidad Recurso",
        "postura_dispositivo": "Postura Dispositivo",
        "ubicacion": "Ubicación",
        "horario_laboral": "Horario Laboral",
        "historial_anomalo": "Historial Anómalo",
        "intentos_fallidos": "Intentos Fallidos",
        "senal_riesgo": "Señal Riesgo"
    }
    imp_sorted  = importancias.sort_values()
    labels_imp  = [nombres_feat_display[k] for k in imp_sorted.index]
    colors_imp  = ["#C8102E" if k == importancias.index[0] else "#4A90D9"
                   for k in imp_sorted.index]

    fig_imp = go.Figure(go.Bar(
        x=imp_sorted.values,
        y=labels_imp,
        orientation="h",
        marker_color=colors_imp,
        text=[f"{v:.4f}" for v in imp_sorted.values],
        textposition="outside",
        width=0.6
    ))
    fig_imp.update_layout(
        plot_bgcolor="#1A1A1A", paper_bgcolor="#1A1A1A",
        font=dict(color="#F5F5F0", family="IBM Plex Mono", size=11),
        xaxis=dict(title="Importancia Gini", gridcolor="#2A2A2A",
                   range=[0, importancias.max() * 1.30]),
        yaxis=dict(gridcolor="#2A2A2A"),
        height=350, margin=dict(t=20, b=40, l=160, r=80),
        showlegend=False
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # Resumen CV texto
    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono';font-size:0.78rem;
                background:#1A1A1A;border:1px solid #3A3A3A;
                border-radius:4px;padding:1rem;line-height:2;color:#888">
        Validación cruzada 5-fold (StratifiedKFold, random_state=42)<br>
        Media &nbsp;&nbsp;&nbsp;: <span style="color:#00C851">{m['cv_mean']:.4f}</span><br>
        Std &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <span style="color:#00C851">+/- {m['cv_std']:.4f}</span><br>
        Scores &nbsp;&nbsp;: <span style="color:#F5F5F0">{[round(v, 4) for v in m['cv_vals']]}</span><br>
        Variable raíz: <span style="color:#C8102E">senal_riesgo</span>
        (Gini importance = <span style="color:#C8102E">{importancias.iloc[0]:.4f}</span>)
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: ÁRBOL DE DECISIÓN
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">// ESTRUCTURA DEL ÁRBOL DE DECISIÓN</div>',
                unsafe_allow_html=True)

    prof = st.slider("Profundidad a visualizar", min_value=2, max_value=6, value=3)

    nombres_feat_tree = ["Rol", "Sensibilidad", "Postura\nDisp.",
                          "Ubicación", "Horario", "Historial\nAnóm.",
                          "Intentos\nFall.", "Señal\nRiesgo"]
    fig_tree, ax_tree = plt.subplots(figsize=(22, 10))
    fig_tree.patch.set_facecolor("#1A1A1A")
    ax_tree.set_facecolor("#1A1A1A")
    plot_tree(modelo, feature_names=nombres_feat_tree,
              class_names=["✅ Permitir", "🔐 Elevar MFA", "❌ Bloquear"],
              filled=True, rounded=True, fontsize=9, proportion=False,
              impurity=True, ax=ax_tree, max_depth=prof)
    ax_tree.set_title(
        f"Árbol de Decisión CART — {prof} niveles | Zero Trust UPC | "
        f"Profundidad real: {modelo.get_depth()} | Hojas: {modelo.get_n_leaves()}",
        color="#F5F5F0", fontsize=12, pad=15)
    plt.tight_layout()
    st.pyplot(fig_tree)

    with st.expander("📄 Ver reglas del árbol en texto"):
        reglas = export_text(modelo, feature_names=features, max_depth=4)
        st.code(reglas, language="text")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: LOG DE AUDITORÍA
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">// LOG DE AUDITORÍA — SIEM (CloudWatch)</div>',
                unsafe_allow_html=True)
    st.markdown("""
    Registros generados automáticamente por el PDP en cada evaluación de acceso.  
    Campos AU-3 del NIST SP 800-53 Rev. 5 (NIST, 2020):
    `decision` · `trust_score` · `rol` · `sensibilidad` · `ubicacion` · `nivel_mfa`
    """)

    if st.session_state.log_eventos:
        log_html = "<br>".join(st.session_state.log_eventos)
        st.markdown(f'<div class="audit-log">{log_html}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="audit-log">
            <span style="color:#555">[SYSTEM] Log de auditoría iniciado — 
            evalúe una solicitud en la pestaña Demo en Vivo para generar eventos...</span>
        </div>""", unsafe_allow_html=True)

    if st.session_state.historial_pred:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">// HISTORIAL DE EVALUACIONES</div>',
                    unsafe_allow_html=True)
        df_hist = pd.DataFrame(st.session_state.historial_pred)
        st.dataframe(df_hist, use_container_width=True, height=300)

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
            st.session_state.log_eventos     = []
            st.session_state.historial_pred  = []
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
        ("Registros (test)",  "400"),
        ("Profundidad árbol", str(modelo.get_depth())),
        ("Número de hojas",   str(modelo.get_n_leaves())),
        ("Accuracy",          f"{metricas['accuracy']:.4f}"),
        ("Precision",         f"{metricas['precision']:.4f}"),
        ("Recall",            f"{metricas['recall']:.4f}"),
        ("F1-Score",          f"{metricas['f1']:.4f}"),
        ("FPR",               f"{metricas['fpr']:.4f}"),
        ("CV Media",          f"{metricas['cv_mean']:.4f} ± {metricas['cv_std']:.4f}"),
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
        PP2 · 2026 · 1FRC0077<br>
        random_state=42 · dataset: 2,000 registros sintéticos
    </div>""", unsafe_allow_html=True)
