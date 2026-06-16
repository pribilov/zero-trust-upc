"""
=============================================================================
DASHBOARD ZERO TRUST — SISTEMA INTELIGENTE DE CONTROL DE ACCESO
Universidad Peruana de Ciencias Aplicadas (UPC)
Proyecto Profesional 2 — 1FRC0077
=============================================================================
Instalar: pip install streamlit scikit-learn pandas numpy matplotlib plotly
Ejecutar: streamlit run app.py
=============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    --rojo-upc: #C8102E; --rojo-dark: #8B0A1E;
    --negro: #0D0D0D; --gris-dark: #1A1A1A;
    --gris-mid: #2A2A2A; --gris-light: #3A3A3A;
    --blanco: #F5F5F0; --verde: #00C851;
    --amarillo: #FFB300; --azul: #0099CC;
    --mono: 'IBM Plex Mono', monospace;
    --sans: 'IBM Plex Sans', sans-serif;
}
html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background-color: var(--negro) !important;
    color: var(--blanco) !important;
}
.stApp { background-color: var(--negro) !important; }
.hero-header {
    background: linear-gradient(135deg, var(--negro) 0%, var(--gris-dark) 50%, #1a0005 100%);
    border: 1px solid var(--rojo-upc); border-radius: 4px;
    padding: 2rem 2.5rem; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.hero-header::before {
    content: ''; position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--rojo-upc), #FF4444, var(--rojo-upc));
}
.hero-title { font-family: var(--mono) !important; font-size: 1.6rem; font-weight: 600; color: var(--blanco); letter-spacing: 0.05em; margin: 0; }
.hero-subtitle { font-size: 0.85rem; color: #888; font-family: var(--mono); margin-top: 0.3rem; letter-spacing: 0.08em; }
.hero-tag { display: inline-block; background: var(--rojo-upc); color: white; font-family: var(--mono); font-size: 0.7rem; padding: 0.2rem 0.6rem; border-radius: 2px; margin-right: 0.5rem; letter-spacing: 0.1em; }
.metric-card { background: var(--gris-dark); border: 1px solid var(--gris-light); border-radius: 4px; padding: 1.2rem 1.5rem; text-align: center; transition: border-color 0.2s; }
.metric-card:hover { border-color: var(--rojo-upc); }
.metric-value { font-family: var(--mono); font-size: 2rem; font-weight: 600; }
.metric-label { font-size: 0.75rem; color: #888; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 0.2rem; }
.decision-permitir { background: linear-gradient(135deg, #003d1a, #001a0d); border: 2px solid var(--verde); border-radius: 4px; padding: 2rem; text-align: center; animation: pulse-green 2s infinite; }
.decision-mfa { background: linear-gradient(135deg, #3d2900, #1a1000); border: 2px solid var(--amarillo); border-radius: 4px; padding: 2rem; text-align: center; animation: pulse-yellow 2s infinite; }
.decision-bloquear { background: linear-gradient(135deg, #3d0008, #1a0004); border: 2px solid var(--rojo-upc); border-radius: 4px; padding: 2rem; text-align: center; animation: pulse-red 2s infinite; }
@keyframes pulse-green { 0%,100%{box-shadow:0 0 0 0 rgba(0,200,81,0.3);} 50%{box-shadow:0 0 20px 5px rgba(0,200,81,0.15);} }
@keyframes pulse-yellow { 0%,100%{box-shadow:0 0 0 0 rgba(255,179,0,0.3);} 50%{box-shadow:0 0 20px 5px rgba(255,179,0,0.15);} }
@keyframes pulse-red { 0%,100%{box-shadow:0 0 0 0 rgba(200,16,46,0.3);} 50%{box-shadow:0 0 20px 5px rgba(200,16,46,0.15);} }
.decision-text { font-family: var(--mono); font-size: 2.2rem; font-weight: 600; letter-spacing: 0.1em; }
.decision-sub { font-size: 0.85rem; color: #aaa; margin-top: 0.5rem; font-family: var(--mono); }
.section-title { font-family: var(--mono); font-size: 0.75rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--rojo-upc); border-bottom: 1px solid var(--gris-light); padding-bottom: 0.5rem; margin-bottom: 1.2rem; }
[data-testid="stSidebar"] { background-color: var(--gris-dark) !important; border-right: 1px solid var(--gris-light); }
[data-testid="stSidebar"] * { color: var(--blanco) !important; }
.stSelectbox > div > div { background-color: #2E2E2E !important; border: 1px solid #555 !important; border-radius: 4px !important; color: #F5F5F0 !important; }
.stSelectbox > div > div > div, .stSelectbox span, .stSelectbox p, [data-baseweb="select"] span, [data-baseweb="select"] div { color: #F5F5F0 !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 0.9rem !important; }
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"], [role="option"] { background-color: #2E2E2E !important; border: 1px solid #555 !important; color: #F5F5F0 !important; }
[role="option"]:hover, [data-baseweb="option"]:hover { background-color: #C8102E !important; color: #FFFFFF !important; }
[aria-selected="true"] { background-color: #8B0A1E !important; color: #FFFFFF !important; }
.stSelectbox label, .stRadio label, .stSlider label { color: #CCCCCC !important; font-size: 0.82rem !important; font-family: 'IBM Plex Sans', sans-serif !important; letter-spacing: 0.03em; }
.stRadio > div { background: transparent !important; }
.stRadio [data-testid="stWidgetLabel"] p { color: #CCCCCC !important; }
.stSlider > div { background: transparent !important; }
.stSlider p { color: #F5F5F0 !important; }
.audit-log { background: var(--negro); border: 1px solid var(--gris-light); border-radius: 4px; padding: 1rem 1.2rem; font-family: var(--mono); font-size: 0.78rem; color: #00C851; max-height: 280px; overflow-y: auto; line-height: 1.8; }
.log-warn { color: var(--amarillo); }
.log-error { color: var(--rojo-upc); }
.log-info { color: var(--azul); }
.stDataFrame { border: 1px solid var(--gris-light) !important; }
</style>
""", unsafe_allow_html=True)


# ─── MODELO ML ───────────────────────────────────────────────────────────────
@st.cache_resource
def entrenar_modelo():
    rng = np.random.RandomState(42)
    N   = 2000
    rol               = rng.randint(0, 4, N)
    sensibilidad      = rng.randint(0, 3, N)
    postura           = rng.randint(0, 2, N)
    ubicacion         = rng.randint(0, 3, N)
    horario           = rng.randint(0, 2, N)
    historial         = rng.randint(0, 2, N)
    intentos_fallidos = rng.randint(0, 6, N)
    senal_riesgo      = rng.randint(0, 3, N)

    etiqueta = np.where(
        (senal_riesgo == 2) | (intentos_fallidos >= 4), 2,
        np.where((senal_riesgo == 0) & (postura == 1) & (historial == 0), 0, 1)
    )
    mask     = rng.random(N) < 0.03
    etiqueta = np.where(mask, rng.randint(0, 3, N), etiqueta)

    features = ["rol", "sensibilidad", "postura_dispositivo", "ubicacion",
                "horario_laboral", "historial_anomalo", "intentos_fallidos", "senal_riesgo"]
    df = pd.DataFrame({
        "rol": rol, "sensibilidad": sensibilidad,
        "postura_dispositivo": postura, "ubicacion": ubicacion,
        "horario_laboral": horario, "historial_anomalo": historial,
        "intentos_fallidos": intentos_fallidos, "senal_riesgo": senal_riesgo,
        "decision": etiqueta
    })
    X  = df[features]
    y_ = df["decision"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_, test_size=0.20, random_state=42, stratify=y_)

    model = DecisionTreeClassifier(
        criterion="gini", max_depth=6, min_samples_split=20,
        min_samples_leaf=10, class_weight="balanced", random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_te,   y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_te,       y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_te, y_pred)
    fpr_lista = []
    for i in range(cm.shape[0]):
        FP = cm[:, i].sum() - cm[i, i]
        TN = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fpr_lista.append(FP / (FP + TN) if (FP + TN) > 0 else 0)
    fpr = float(np.mean(fpr_lista))

    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_vals = cross_val_score(model, X, y_, cv=cv, scoring="accuracy")
    imp     = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

    metricas = {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "fpr": fpr, "cm": cm,
        "cv_mean": float(cv_vals.mean()), "cv_std": float(cv_vals.std()),
        "cv_vals": cv_vals.tolist(),
    }
    return model, features, metricas, imp, X, y_

modelo, features, metricas, importancias, X_full, y_full = entrenar_modelo()

# ─── ESTADO DE SESIÓN ────────────────────────────────────────────────────────
# Valores por defecto del formulario
DEFAULTS = {"rol": 0, "sens": 0, "postura": 1, "ubi": 0,
            "horario": 1, "hist": 0, "intentos": 0, "senal": 0}

ESCENARIOS = {
    1: {"rol": 0, "sens": 0, "postura": 1, "ubi": 0, "horario": 1, "hist": 0, "intentos": 0, "senal": 0},
    2: {"rol": 1, "sens": 2, "postura": 0, "ubi": 2, "horario": 0, "hist": 0, "intentos": 0, "senal": 2},
    3: {"rol": 2, "sens": 1, "postura": 0, "ubi": 1, "horario": 1, "hist": 0, "intentos": 4, "senal": 2},
    4: {"rol": 3, "sens": 2, "postura": 1, "ubi": 0, "horario": 0, "hist": 1, "intentos": 0, "senal": 1},
}

# Inicializar session_state con defaults
for k, v in DEFAULTS.items():
    if f"inp_{k}" not in st.session_state:
        st.session_state[f"inp_{k}"] = v
if "log_eventos" not in st.session_state:
    st.session_state.log_eventos = []
if "historial_pred" not in st.session_state:
    st.session_state.historial_pred = []
if "resultado" not in st.session_state:
    st.session_state.resultado = None

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

    # ── Botones de escenarios predefinidos ──────────────────────────────────
    st.markdown('<div class="section-title">// ESCENARIOS DOCUMENTADOS (TABLA 44 - CAPÍTULO 4)</div>',
                unsafe_allow_html=True)
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)

    def cargar_escenario(n):
        e = ESCENARIOS[n]
        for k, v in e.items():
            st.session_state[f"inp_{k}"] = v
        st.session_state.resultado = None

    with col_e1:
        st.button("E1 — Estudiante campus\n✅ PERMITIR",
                  use_container_width=True,
                  on_click=cargar_escenario, args=(1,))
    with col_e2:
        st.button("E2 — Docente exterior\n❌ BLOQUEAR",
                  use_container_width=True,
                  on_click=cargar_escenario, args=(2,))
    with col_e3:
        st.button("E3 — Credential stuffing\n❌ BLOQUEAR",
                  use_container_width=True,
                  on_click=cargar_escenario, args=(3,))
    with col_e4:
        st.button("E4 — TI fuera de horario\n🔐 ELEVAR MFA",
                  use_container_width=True,
                  on_click=cargar_escenario, args=(4,))

    st.divider()
    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<div class="section-title">// ATRIBUTOS DE LA SOLICITUD</div>',
                    unsafe_allow_html=True)

        rol_val = st.selectbox(
            "👤 Rol del Usuario",
            options=[0, 1, 2, 3],
            index=st.session_state.inp_rol,
            format_func=lambda x: ["Estudiante (0)", "Docente (1)",
                                    "Administrativo (2)", "TI (3)"][x],
            key="inp_rol"
        )
        sens_val = st.selectbox(
            "📁 Sensibilidad del Recurso",
            options=[0, 1, 2],
            index=st.session_state.inp_sens,
            format_func=lambda x: ["Baja (0) — datos públicos",
                                    "Media (1) — datos académicos",
                                    "Alta (2) — datos financieros/personales"][x],
            key="inp_sens"
        )
        postura_val = st.radio(
            "💻 Postura del Dispositivo",
            options=[1, 0],
            index=0 if st.session_state.inp_postura == 1 else 1,
            format_func=lambda x: "✅ Cumple política (MDM registrado)" if x else "❌ No cumple política",
            horizontal=True,
            key="inp_postura"
        )
        ubi_val = st.selectbox(
            "🌍 Ubicación del Acceso",
            options=[0, 1, 2],
            index=st.session_state.inp_ubi,
            format_func=lambda x: ["🏫 Campus universitario",
                                    "🏠 Remoto — Perú",
                                    "✈️ Remoto — Exterior"][x],
            key="inp_ubi"
        )
        horario_val = st.radio(
            "⏰ Horario",
            options=[1, 0],
            index=0 if st.session_state.inp_horario == 1 else 1,
            format_func=lambda x: "✅ Horario laboral (07:00–22:00)" if x else "⚠️ Fuera de horario",
            horizontal=True,
            key="inp_horario"
        )
        hist_val = st.radio(
            "📈 Historial (últimas 24h)",
            options=[0, 1],
            index=st.session_state.inp_hist,
            format_func=lambda x: "✅ Normal" if x == 0 else "⚠️ Comportamiento anómalo detectado",
            horizontal=True,
            key="inp_hist"
        )
        intentos_val = st.slider(
            "🔑 Intentos de Autenticación Fallidos",
            min_value=0, max_value=5,
            value=st.session_state.inp_intentos,
            help="4+ activa bloqueo inmediato (regla fail-secure NIST SP 800-207)",
            key="inp_intentos"
        )
        senal_val = st.select_slider(
            "🚨 Señal de Riesgo (Threat Intelligence)",
            options=[0, 1, 2],
            value=st.session_state.inp_senal,
            format_func=lambda x: {0: "🟢 Ninguna", 1: "🟡 Moderada", 2: "🔴 Alta"}[x],
            key="inp_senal"
        )

        evaluar = st.button("⚡ EVALUAR SOLICITUD", use_container_width=True, type="primary")

    # ── Panel de resultados ──────────────────────────────────────────────────
    with col_result:
        st.markdown('<div class="section-title">// DECISIÓN DEL SISTEMA</div>',
                    unsafe_allow_html=True)

        if evaluar:
            # Trust Score R = 0.35Ru + 0.25Rd + 0.20Rc + 0.20Rr
            # (Rose et al., 2020; Wang et al., 2025; Lukaseder et al., 2020)
            Ru = hist_val * 0.6 + (intentos_val / 5) * 0.4
            Rd = (1 - postura_val) * 0.7 + (ubi_val / 2) * 0.3
            Rc = (1 - horario_val) * 0.3 + (sens_val / 2) * 0.7
            Rr = senal_val / 2
            R  = round(0.35 * Ru + 0.25 * Rd + 0.20 * Rc + 0.20 * Rr, 3)

            X_new = pd.DataFrame(
                [[rol_val, sens_val, postura_val, ubi_val,
                  horario_val, hist_val, intentos_val, senal_val]],
                columns=features)
            pred  = modelo.predict(X_new)[0]
            probs = modelo.predict_proba(X_new)[0]
            conf  = max(probs) * 100

            aal   = {0: "AAL1", 1: "AAL3" if sens_val == 2 else "AAL2", 2: "N/A"}[pred]
            token = {0: "4 horas", 1: "15 min" if sens_val == 2 else "30 min", 2: "—"}[pred]
            etiq  = {0: "Permitir", 1: "Elevar MFA", 2: "Bloquear"}[pred]

            st.session_state.resultado = {
                "pred": pred, "R": R, "Ru": Ru, "Rd": Rd, "Rc": Rc, "Rr": Rr,
                "probs": probs.tolist(), "conf": conf,
                "aal": aal, "token": token, "etiq": etiq,
                "rol_val": rol_val, "sens_val": sens_val,
                "intentos_val": intentos_val, "senal_val": senal_val,
                "postura_val": postura_val, "horario_val": horario_val,
                "hist_val": hist_val, "ubi_val": ubi_val,
            }

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
            st.session_state.historial_pred.insert(0, {
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
                "Confianza (%)": f"{conf:.1f}",
            })

        # Mostrar resultado (persiste entre reruns)
        if st.session_state.resultado:
            r = st.session_state.resultado
            pred   = r["pred"]
            R      = r["R"]
            conf   = r["conf"]
            aal    = r["aal"]
            token  = r["token"]
            probs  = r["probs"]
            Ru, Rd, Rc, Rr = r["Ru"], r["Rd"], r["Rc"], r["Rr"]

            if pred == 0:
                st.markdown(f"""
                <div class="decision-permitir">
                    <div class="decision-text" style="color:#00C851">✅&nbsp;&nbsp;ACCESO PERMITIDO</div>
                    <div class="decision-sub">Trust Score R = {R} &nbsp;|&nbsp; {aal} &nbsp;|&nbsp; Token: {token}</div>
                </div>""", unsafe_allow_html=True)
            elif pred == 1:
                st.markdown(f"""
                <div class="decision-mfa">
                    <div class="decision-text" style="color:#FFB300">🔐&nbsp;&nbsp;ELEVAR AUTENTICACIÓN</div>
                    <div class="decision-sub">Trust Score R = {R} &nbsp;|&nbsp; {aal} requerido &nbsp;|&nbsp; Token: {token}</div>
                </div>""", unsafe_allow_html=True)
            else:
                motivo = "Regla fail-secure: intentos >= 4" if r["intentos_val"] >= 4 else f"R = {R} ≥ 0.70"
                st.markdown(f"""
                <div class="decision-bloquear">
                    <div class="decision-text" style="color:#FF4444">❌&nbsp;&nbsp;ACCESO BLOQUEADO</div>
                    <div class="decision-sub">Trust Score R = {R} &nbsp;|&nbsp; {motivo}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Desglose Trust Score
            st.markdown(f"""
            <div style="font-family:'IBM Plex Mono';font-size:0.78rem;
                        background:#1A1A1A;border:1px solid #3A3A3A;
                        border-radius:4px;padding:1rem;line-height:2;color:#888">
                <span style="color:#C8102E">R</span> =
                0.35 × Ru({Ru:.3f}) + 0.25 × Rd({Rd:.3f}) +
                0.20 × Rc({Rc:.3f}) + 0.20 × Rr({Rr:.3f})<br>
                <span style="color:#C8102E">R</span> = <b style="color:#F5F5F0">{R:.3f}</b>
                &nbsp;|&nbsp; Umbrales: PERMITIR &lt; 0.30 &nbsp;|&nbsp; BLOQUEAR ≥ 0.70<br>
                Norma: NIST SP 800-207 (Rose et al., 2020; Wang et al., 2025)
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Gráfico probabilidades
            fig_prob = go.Figure(go.Bar(
                x=["Permitir", "Elevar MFA", "Bloquear"],
                y=[p * 100 for p in probs],
                marker_color=["#00C851", "#FFB300", "#C8102E"],
                text=[f"{p*100:.1f}%" for p in probs],
                textposition="outside", width=0.5
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
            sv = r["senal_val"]; iv = r["intentos_val"]
            hv = r["hist_val"]; pv = r["postura_val"]
            sv2 = r["sens_val"]; hov = r["horario_val"]; uv = r["ubi_val"]
            if sv == 2:   factores.append(("🔴", "Señal de riesgo ALTA — Threat Intelligence activo"))
            elif sv == 1: factores.append(("🟡", "Señal de riesgo MODERADA"))
            if iv >= 4:   factores.append(("🔴", f"{iv} intentos fallidos — activó regla fail-secure (NIST SP 800-207 tenet 5)"))
            elif iv >= 2: factores.append(("🟡", f"{iv} intentos fallidos — patrón sospechoso"))
            if hv == 1:   factores.append(("🟡", "Historial anómalo detectado en las últimas 24h"))
            if pv == 0 and sv2 == 2: factores.append(("🔴", "Dispositivo no conforme accediendo a recurso de ALTA sensibilidad"))
            elif pv == 0: factores.append(("🟡", "Dispositivo sin cumplimiento de política MDM"))
            if hov == 0:  factores.append(("🟡", "Acceso fuera del horario laboral establecido"))
            if uv == 2:   factores.append(("🟡", "Acceso desde ubicación exterior al país"))
            if factores:
                for icono, desc in factores:
                    st.markdown(f"`{icono}` {desc}")
            else:
                st.markdown("`✅` Sin factores de riesgo detectados — entorno de bajo riesgo")

        else:
            st.markdown("""
            <div style="text-align:center; padding:4rem 2rem; color:#555;
                        border:1px dashed #333; border-radius:4px;">
                <div style="font-size:3rem">🔐</div>
                <div style="font-family:'IBM Plex Mono'; font-size:0.9rem; margin-top:1rem;">
                    Seleccione un escenario predefinido o<br>configure los atributos y presione EVALUAR
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
            text=cm, texttemplate="%{text}", textfont={"size":18,"color":"white"},
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
        fig_cv = go.Figure()
        fig_cv.add_trace(go.Scatter(
            x=list(range(1, 6)), y=m["cv_vals"], mode="lines+markers",
            line=dict(color="#C8102E", width=3),
            marker=dict(size=10, color="#C8102E", line=dict(color="#F5F5F0", width=2))
        ))
        fig_cv.add_hline(y=m["cv_mean"], line_dash="dash", line_color="#888",
                          annotation_text=f"Media: {m['cv_mean']:.4f} ± {m['cv_std']:.4f}",
                          annotation_font_color="#888")
        fig_cv.update_layout(
            plot_bgcolor="#1A1A1A", paper_bgcolor="#1A1A1A",
            font=dict(color="#F5F5F0", family="IBM Plex Mono", size=11),
            xaxis=dict(title="Fold", gridcolor="#2A2A2A", tickvals=[1,2,3,4,5]),
            yaxis=dict(title="Accuracy", gridcolor="#2A2A2A", range=[0.85, 1.0]),
            height=320, margin=dict(t=20, b=60, l=80, r=20), showlegend=False
        )
        st.plotly_chart(fig_cv, use_container_width=True)

    # Importancia de variables
    st.markdown('<div class="section-title">// IMPORTANCIA DE VARIABLES (GINI IMPORTANCE)</div>',
                unsafe_allow_html=True)
    nombres_display = {
        "rol": "Rol Usuario", "sensibilidad": "Sensibilidad Recurso",
        "postura_dispositivo": "Postura Dispositivo", "ubicacion": "Ubicación",
        "horario_laboral": "Horario Laboral", "historial_anomalo": "Historial Anómalo",
        "intentos_fallidos": "Intentos Fallidos", "senal_riesgo": "Señal Riesgo"
    }
    imp_sorted  = importancias.sort_values()
    labels_imp  = [nombres_display[k] for k in imp_sorted.index]
    colors_imp  = ["#C8102E" if k == importancias.index[0] else "#4A90D9"
                   for k in imp_sorted.index]
    fig_imp = go.Figure(go.Bar(
        x=imp_sorted.values, y=labels_imp, orientation="h",
        marker_color=colors_imp,
        text=[f"{v:.4f}" for v in imp_sorted.values],
        textposition="outside", width=0.6
    ))
    fig_imp.update_layout(
        plot_bgcolor="#1A1A1A", paper_bgcolor="#1A1A1A",
        font=dict(color="#F5F5F0", family="IBM Plex Mono", size=11),
        xaxis=dict(title="Importancia Gini", gridcolor="#2A2A2A",
                   range=[0, importancias.max() * 1.30]),
        yaxis=dict(gridcolor="#2A2A2A"),
        height=350, margin=dict(t=20, b=40, l=160, r=80), showlegend=False
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown(f"""
    <div style="font-family:'IBM Plex Mono';font-size:0.78rem;
                background:#1A1A1A;border:1px solid #3A3A3A;
                border-radius:4px;padding:1rem;line-height:2;color:#888">
        Variable raíz del árbol: <span style="color:#C8102E">senal_riesgo</span>
        (Gini importance = <span style="color:#C8102E">{importancias.iloc[0]:.4f}</span>)<br>
        Validación cruzada: Media = <span style="color:#00C851">{m['cv_mean']:.4f}</span>
        ± <span style="color:#00C851">{m['cv_std']:.4f}</span> &nbsp;|&nbsp;
        Scores: <span style="color:#F5F5F0">{[round(v, 4) for v in m['cv_vals']]}</span>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: ÁRBOL DE DECISIÓN
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">// ESTRUCTURA DEL ÁRBOL DE DECISIÓN</div>',
                unsafe_allow_html=True)
    prof = st.slider("Profundidad a visualizar", min_value=2, max_value=6, value=3)
    nombres_tree = ["Rol", "Sensibilidad", "Postura\nDisp.", "Ubicación",
                     "Horario", "Historial\nAnóm.", "Intentos\nFall.", "Señal\nRiesgo"]
    fig_tree, ax_tree = plt.subplots(figsize=(22, 10))
    fig_tree.patch.set_facecolor("#1A1A1A")
    ax_tree.set_facecolor("#1A1A1A")
    plot_tree(modelo, feature_names=nombres_tree,
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
    st.markdown("Campos AU-3 del NIST SP 800-53 Rev. 5: `decision` · `trust_score` · `rol` · `sensibilidad` · `ubicacion` · `nivel_mfa`")

    if st.session_state.log_eventos:
        log_html = "<br>".join(st.session_state.log_eventos)
        st.markdown(f'<div class="audit-log">{log_html}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="audit-log">
            <span style="color:#555">[SYSTEM] Log iniciado — evalúe una solicitud para generar eventos...</span>
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
            st.session_state.log_eventos    = []
            st.session_state.historial_pred = []
            st.session_state.resultado      = None
            st.rerun()

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0;border-bottom:1px solid #333">
        <div style="font-size:2.5rem">🔐</div>
        <div style="font-family:'IBM Plex Mono';font-size:0.8rem;color:#C8102E;
                    letter-spacing:0.1em;margin-top:0.5rem;">ZERO TRUST AI</div>
        <div style="font-size:0.7rem;color:#666;margin-top:0.2rem;">Decision Tree Classifier</div>
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
        random_state=42 · 2,000 registros sintéticos
    </div>""", unsafe_allow_html=True)
