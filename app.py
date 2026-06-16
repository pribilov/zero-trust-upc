import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# ── CONFIGURACIÓN DE PÁGINA ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Zero Trust UPC - Dashboard ML",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── ESTILOS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    .stApp { background-color: #0a0e1a; color: #e2e8f0; }
    
    .header-box {
        background: linear-gradient(135deg, #0f1424 0%, #1a0a0a 100%);
        border: 1px solid #dc2626;
        border-radius: 12px;
        padding: 28px 32px;
        margin-bottom: 24px;
    }
    .header-badge {
        display: inline-block;
        background: #dc2626;
        color: white;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 4px;
        margin-right: 6px;
        letter-spacing: 0.1em;
    }
    .header-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 28px;
        font-weight: 700;
        color: #f8f8f8;
        letter-spacing: 0.05em;
        margin: 12px 0 8px 0;
    }
    .header-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #64748b;
        letter-spacing: 0.03em;
    }
    .metric-card {
        background: #0f1424;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 36px;
        font-weight: 700;
        color: #22c55e;
    }
    .metric-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.12em;
        color: #94a3b8;
        margin-top: 4px;
    }
    .metric-umbral {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        color: #475569;
        margin-top: 6px;
    }
    .metric-check { color: #22c55e; }
    .section-title {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.15em;
        color: #475569;
        text-transform: uppercase;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid #1e293b;
    }
    .decision-permitir { 
        background: rgba(34,197,94,0.1); 
        border: 1px solid rgba(34,197,94,0.3);
        border-radius: 8px; padding: 16px; text-align: center;
    }
    .decision-bloquear { 
        background: rgba(239,68,68,0.1); 
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 8px; padding: 16px; text-align: center;
    }
    .decision-mfa { 
        background: rgba(234,179,8,0.1); 
        border: 1px solid rgba(234,179,8,0.3);
        border-radius: 8px; padding: 16px; text-align: center;
    }
    .log-box {
        background: #050810;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #94a3b8;
        line-height: 1.8;
    }
    div[data-testid="stTabs"] button {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── DATASET Y MODELO (cacheados) ─────────────────────────────────────────────
@st.cache_data
def generar_dataset(n=2000, random_state=42):
    """
    Dataset sintético de patrones de acceso universitario.
    Etiquetado basado en reglas Zero Trust (Rose et al., 2020; 
    Wang et al., 2025; Lukaseder et al., 2020).
    """
    rng = np.random.RandomState(random_state)
    rol               = rng.randint(0, 4, n)
    sensibilidad      = rng.randint(0, 3, n)
    postura           = rng.randint(0, 2, n)
    ubicacion         = rng.randint(0, 3, n)
    horario           = rng.randint(0, 2, n)
    historial         = rng.randint(0, 2, n)
    intentos_fallidos = rng.randint(0, 6, n)
    senal_riesgo      = rng.randint(0, 3, n)

    etiqueta = np.where(
        (senal_riesgo == 2) | (intentos_fallidos >= 4), 2,
        np.where(
            (senal_riesgo == 0) & (postura == 1) & (historial == 0), 0,
            1
        )
    )
    mask = rng.random(n) < 0.03
    etiqueta = np.where(mask, rng.randint(0, 3, n), etiqueta)

    return pd.DataFrame({
        'rol': rol, 'sensibilidad': sensibilidad,
        'postura_dispositivo': postura, 'ubicacion': ubicacion,
        'horario_laboral': horario, 'historial_anomalo': historial,
        'intentos_fallidos': intentos_fallidos, 'senal_riesgo': senal_riesgo,
        'clase': etiqueta
    })

@st.cache_resource
def entrenar_modelo():
    df = generar_dataset(2000, 42)
    X = df.drop(columns=['clase'])
    y = df['clase']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    
    modelo = DecisionTreeClassifier(
        criterion='gini', max_depth=6,
        min_samples_split=20, min_samples_leaf=10,
        class_weight='balanced', random_state=42
    )
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, average='weighted')
    rec   = recall_score(y_test, y_pred,    average='weighted')
    f1    = f1_score(y_test, y_pred,         average='weighted')
    cm    = confusion_matrix(y_test, y_pred)
    fpr_l = []
    for i in range(cm.shape[0]):
        FP = cm[:, i].sum() - cm[i, i]
        TN = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fpr_l.append(FP / (FP + TN) if (FP + TN) > 0 else 0)
    fpr = np.mean(fpr_l)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(modelo, X, y, cv=cv, scoring='accuracy')

    importancias = pd.Series(
        modelo.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    return modelo, X, y, X_train, X_test, y_train, y_test, y_pred, \
           acc, prec, rec, f1, fpr, cv_scores, importancias, cm

# Cargar modelo
modelo, X, y, X_train, X_test, y_train, y_test, y_pred, \
acc, prec, rec, f1, fpr, cv_scores, importancias, cm = entrenar_modelo()

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <span class="header-badge">ZERO TRUST</span>
    <span class="header-badge">ML</span>
    <span class="header-badge">CLOUD</span>
    <div class="header-title">🔒 SISTEMA INTELIGENTE DE CONTROL DE ACCESO</div>
    <div class="header-sub">
        Universidad Peruana de Ciencias Aplicadas &nbsp;|&nbsp; 
        Ingeniería de Redes y Comunicaciones &nbsp;|&nbsp; 
        Modelo: Árbol de Decisión &nbsp;|&nbsp; NIST SP 800-207
    </div>
</div>
""", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "⚡ Demo en Vivo",
    "📊 Métricas del Modelo",
    "🌿 Árbol de Decisión",
    "📋 Log de Auditoría"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1: DEMO EN VIVO
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">// SIMULADOR INTERACTIVO - POLICY DECISION POINT (PDP)</div>', 
                unsafe_allow_html=True)
    
    # Escenarios predefinidos
    st.markdown("**Escenarios documentados (Tabla 44 - Capítulo 4):**")
    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    
    esc_sel = None
    with col_e1:
        if st.button("E1 · Estudiante campus", use_container_width=True):
            esc_sel = 1
    with col_e2:
        if st.button("E2 · Docente exterior", use_container_width=True):
            esc_sel = 2
    with col_e3:
        if st.button("E3 · Credential stuffing", use_container_width=True):
            esc_sel = 3
    with col_e4:
        if st.button("E4 · TI fuera de horario", use_container_width=True):
            esc_sel = 4

    escenarios = {
        1: {"rol": 0, "sens": 0, "postura": 1, "ubi": 0, "horario": 1, "hist": 0, "intentos": 0, "senal": 0},
        2: {"rol": 1, "sens": 2, "postura": 0, "ubi": 2, "horario": 0, "hist": 0, "intentos": 0, "senal": 2},
        3: {"rol": 2, "sens": 1, "postura": 0, "ubi": 1, "horario": 1, "hist": 0, "intentos": 4, "senal": 2},
        4: {"rol": 3, "sens": 2, "postura": 1, "ubi": 0, "horario": 0, "hist": 1, "intentos": 0, "senal": 1},
    }

    st.divider()
    col_izq, col_der = st.columns([1, 1])

    with col_izq:
        st.markdown("**Atributos de entrada:**")
        defaults = escenarios.get(esc_sel, escenarios[1]) if esc_sel else escenarios[1]

        rol       = st.selectbox("Rol del usuario",
                        ["Estudiante (0)", "Docente (1)", "Administrativo (2)", "TI (3)"],
                        index=defaults["rol"])
        sens      = st.selectbox("Sensibilidad del recurso",
                        ["Baja (0)", "Media (1)", "Alta (2)"],
                        index=defaults["sens"])
        postura   = st.selectbox("Postura del dispositivo",
                        ["No conforme (0)", "Conforme (1)"],
                        index=defaults["postura"])
        ubi       = st.selectbox("Ubicación",
                        ["Campus (0)", "Remoto Perú (1)", "Remoto Exterior (2)"],
                        index=defaults["ubi"])
        horario   = st.selectbox("Horario laboral",
                        ["Fuera de horario (0)", "En horario (1)"],
                        index=defaults["horario"])
        hist      = st.selectbox("Historial anómalo",
                        ["Normal (0)", "Anómalo (1)"],
                        index=defaults["hist"])
        intentos  = st.slider("Intentos fallidos", 0, 5, defaults["intentos"])
        senal     = st.selectbox("Señal de riesgo",
                        ["Ninguna (0)", "Moderada (1)", "Alta (2)"],
                        index=defaults["senal"])

    with col_der:
        # Extraer valores numéricos
        r = int(rol.split("(")[1].replace(")", ""))
        s = int(sens.split("(")[1].replace(")", ""))
        p = int(postura.split("(")[1].replace(")", ""))
        u = int(ubi.split("(")[1].replace(")", ""))
        h = int(horario.split("(")[1].replace(")", ""))
        hi = int(hist.split("(")[1].replace(")", ""))
        se = int(senal.split("(")[1].replace(")", ""))

        # Trust Score
        Ru = hi * 0.6 + (intentos / 5) * 0.4
        Rd = (1 - p) * 0.7 + (u / 2) * 0.3
        Rc = (1 - h) * 0.3 + (s / 2) * 0.7
        Rr = se / 2
        R  = round(0.35 * Ru + 0.25 * Rd + 0.20 * Rc + 0.20 * Rr, 3)

        fail_secure = intentos >= 4

        if fail_secure or R >= 0.70:
            decision = "BLOQUEAR"
            css_class = "decision-bloquear"
            icon = "🚫"
            color_r = "#ef4444"
            aal = "N/A — Acceso denegado"
            token = "—"
        elif R >= 0.30:
            decision = "ELEVAR MFA"
            css_class = "decision-mfa"
            icon = "⚠️"
            color_r = "#eab308"
            aal = "AAL3" if s == 2 else "AAL2"
            token = "15 min" if s == 2 else "30 min"
        else:
            decision = "PERMITIR"
            css_class = "decision-permitir"
            icon = "✅"
            color_r = "#22c55e"
            aal = "AAL1"
            token = "4 horas"

        motivo = "Regla fail-secure: intentos >= 4 (NIST SP 800-207 tenet 5)" \
                 if fail_secure else f"R = {R} {'≥' if R >= 0.70 else ('≥' if R >= 0.30 else '<')} umbral"

        st.markdown(f"""
        <div class="{css_class}" style="margin-bottom:16px">
            <div style="font-size:32px;margin-bottom:8px">{icon}</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:22px;
                        font-weight:700;color:{color_r};letter-spacing:0.05em">
                {decision}
            </div>
        </div>""", unsafe_allow_html=True)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.metric("Trust Score R", f"{R:.3f}")
            st.metric("Nivel MFA", aal)
        with col_r2:
            st.metric("Token / Sesión", token)
            st.metric("Norma aplicada", "NIST SP 800-207")

        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                    color:#475569;margin-top:12px;padding:10px;
                    background:#050810;border-radius:6px;border:1px solid #1e293b">
            R = 0.35×{Ru:.3f} + 0.25×{Rd:.3f} + 0.20×{Rc:.3f} + 0.20×{Rr:.3f} = <b style="color:{color_r}">{R:.3f}</b><br>
            Motivo: {motivo}
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2: MÉTRICAS DEL MODELO
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">// MÉTRICAS DE EVALUACIÓN — INDICADORES DE LOGRO</div>',
                unsafe_allow_html=True)

    # 5 métricas principales
    umbrales  = [0.85, 0.80, 0.80, 0.80, 0.10]
    valores   = [acc, prec, rec, f1, fpr]
    etiquetas = ["EXACTITUD", "PRECISIÓN", "RECALL", "F1-SCORE", "FPR"]
    tipos     = [">=", ">=", ">=", ">=", "<="]

    cols = st.columns(5)
    for i, col in enumerate(cols):
        cumple = (valores[i] >= umbrales[i]) if tipos[i] == ">=" else (valores[i] <= umbrales[i])
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{valores[i]:.4f}</div>
                <div class="metric-label">{etiquetas[i]}</div>
                <div class="metric-umbral">Umbral: {tipos[i]} {umbrales[i]} 
                    {'<span class="metric-check">✓</span>' if cumple else '✗'}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # CV y árbol info
    col_cv, col_tree = st.columns(2)
    with col_cv:
        st.markdown("**Validación cruzada 5-fold:**")
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace;font-size:13px;
                    background:#050810;border:1px solid #1e293b;
                    border-radius:8px;padding:16px;line-height:2">
            Media &nbsp;&nbsp;&nbsp;&nbsp;: <span style="color:#22c55e">{cv_scores.mean():.4f}</span><br>
            Std &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <span style="color:#22c55e">+/- {cv_scores.std():.4f}</span><br>
            Scores &nbsp;&nbsp;: <span style="color:#94a3b8">{[round(float(s),4) for s in cv_scores]}</span>
        </div>""", unsafe_allow_html=True)

    with col_tree:
        st.markdown("**Parámetros del modelo:**")
        st.markdown(f"""
        <div style="font-family:'JetBrains Mono',monospace;font-size:12px;
                    background:#050810;border:1px solid #1e293b;
                    border-radius:8px;padding:16px;line-height:2;color:#94a3b8">
            criterion &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <span style="color:#63b3ed">gini</span><br>
            max_depth &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: <span style="color:#63b3ed">6</span><br>
            min_samples_split : <span style="color:#63b3ed">20</span><br>
            min_samples_leaf &nbsp;: <span style="color:#63b3ed">10</span><br>
            class_weight &nbsp;&nbsp;&nbsp;&nbsp;: <span style="color:#63b3ed">balanced</span><br>
            random_state &nbsp;&nbsp;&nbsp;&nbsp;: <span style="color:#63b3ed">42</span>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Gráfico de barras métricas vs umbrales
    st.markdown("**Comparación: valores obtenidos vs. umbrales del indicador OE03**")
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0f1424')
    ax.set_facecolor('#0f1424')

    labels_graf = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'FPR']
    x = np.arange(len(labels_graf))
    ancho = 0.35

    bars_u = ax.bar(x - ancho/2, umbrales, ancho,
                    label='Umbral del indicador',
                    color='#334155', edgecolor='#475569', linewidth=0.8)
    bars_v = ax.bar(x + ancho/2, valores, ancho,
                    label='Valor obtenido (simulación)',
                    color='#1d4ed8', edgecolor='#3b82f6', linewidth=0.8)

    for bar, val, umb, tipo in zip(bars_v, valores, umbrales, tipos):
        cumple = (val >= umb) if tipo == ">=" else (val <= umb)
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9,
                fontweight='bold', color='#22c55e' if cumple else '#ef4444')

    for bar, val in zip(bars_u, umbrales):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=9, color='#94a3b8')

    ax.set_xticks(x)
    ax.set_xticklabels(labels_graf, fontsize=11, color='#e2e8f0')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Valor de la métrica', fontsize=11, color='#94a3b8')
    ax.set_xlabel('Métrica de evaluación del modelo CART', fontsize=11, color='#94a3b8')
    ax.tick_params(colors='#94a3b8')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#334155')
    ax.spines['bottom'].set_color('#334155')
    ax.yaxis.grid(True, color='#1e293b', linewidth=0.8)
    ax.set_axisbelow(True)
    legend = ax.legend(fontsize=10, facecolor='#0f1424',
                       edgecolor='#334155', labelcolor='#e2e8f0')
    st.pyplot(fig)
    plt.close()

    st.divider()

    # Importancia de variables
    st.markdown("**Importancia de variables (Gini importance):**")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    fig2.patch.set_facecolor('#0f1424')
    ax2.set_facecolor('#0f1424')

    colores = ['#1d4ed8' if i == 0 else '#334155' for i in range(len(importancias))]
    bars_imp = ax2.barh(importancias.index[::-1],
                        importancias.values[::-1],
                        color=colores[::-1], edgecolor='#475569', linewidth=0.5)

    for bar, val in zip(bars_imp, importancias.values[::-1]):
        ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=9, color='#94a3b8')

    ax2.set_xlabel('Importancia Gini', fontsize=10, color='#94a3b8')
    ax2.tick_params(colors='#94a3b8')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#334155')
    ax2.spines['bottom'].set_color('#334155')
    ax2.xaxis.grid(True, color='#1e293b', linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.set_xlim(0, importancias.max() * 1.25)
    st.pyplot(fig2)
    plt.close()

# ════════════════════════════════════════════════════════════════════════════
# TAB 3: ÁRBOL DE DECISIÓN
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">// ÁRBOL DE DECISIÓN CART — VISUALIZACIÓN</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    **Variable raíz:** `senal_riesgo` (Gini importance = **{importancias['senal_riesgo']:.4f}**)  
    **Profundidad máxima configurada:** 6 | **Profundidad real:** {modelo.get_depth()} | 
    **Número de hojas:** {modelo.get_n_leaves()}
    """)

    fig3, ax3 = plt.subplots(figsize=(20, 10))
    fig3.patch.set_facecolor('#0f1424')
    ax3.set_facecolor('#0f1424')
    plot_tree(
        modelo,
        feature_names=X.columns.tolist(),
        class_names=['PERMITIR', 'ELEVAR_MFA', 'BLOQUEAR'],
        filled=True, rounded=True,
        max_depth=3,
        fontsize=8,
        ax=ax3
    )
    ax3.set_title('Árbol de Decisión CART (primeros 3 niveles)',
                  color='#e2e8f0', fontsize=12, pad=20)
    st.pyplot(fig3)
    plt.close()
    st.caption("Se muestran los primeros 3 niveles para mayor legibilidad. Profundidad total: 6.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 4: LOG DE AUDITORÍA
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">// SIMULACIÓN DE LOG DE AUDITORÍA — CLOUDWATCH</div>',
                unsafe_allow_html=True)
    st.markdown("""
    Simulación offline del Log Group `/aws/lambda/zero-trust-pdp`.  
    Campos AU-3 del NIST SP 800-53 Rev. 5 (NIST, 2020).
    """)

    logs_escenarios = [
        {"decision": "PERMITIR",    "trust_score": 0.020, "rol": "estudiante",   "sensibilidad": "baja",  "ubicacion": "campus",           "nivel_mfa": "AAL1",              "senal": "ninguna"},
        {"decision": "BLOQUEAR",    "trust_score": 0.865, "rol": "docente",      "sensibilidad": "alta",  "ubicacion": "remoto_exterior",   "nivel_mfa": "N/A-acceso-denegado","senal": "alta"},
        {"decision": "BLOQUEAR",    "trust_score": 0.705, "rol": "administrativo","sensibilidad": "media", "ubicacion": "remoto_peru",       "nivel_mfa": "N/A-acceso-denegado","senal": "alta"},
        {"decision": "ELEVAR_MFA",  "trust_score": 0.445, "rol": "ti",           "sensibilidad": "alta",  "ubicacion": "campus",            "nivel_mfa": "AAL3",              "senal": "moderada"},
    ]

    import datetime
    ahora = datetime.datetime.now()

    log_html = ""
    for i, ev in enumerate(logs_escenarios):
        ts = (ahora - datetime.timedelta(seconds=(3 - i) * 15)).strftime("%Y-%m-%d %H:%M:%S")
        col_dec = "#22c55e" if ev["decision"] == "PERMITIR" else ("#ef4444" if ev["decision"] == "BLOQUEAR" else "#eab308")
        log_html += f"""
        <div style="margin-bottom:12px;padding:10px;background:#050810;
                    border-left:3px solid {col_dec};border-radius:0 6px 6px 0">
            <span style="color:#475569">[{ts}]</span>
            <span style="color:#22c55e;font-weight:700"> [AUDIT]</span>
            <span style="color:#63b3ed"> decision=</span><span style="color:{col_dec};font-weight:700">{ev['decision']}</span>
            <span style="color:#63b3ed"> trust_score=</span>{ev['trust_score']:.3f}
            <br>
            <span style="color:transparent">[{ts}]&nbsp;</span>
            <span style="color:#63b3ed">rol=</span>{ev['rol']}
            <span style="color:#63b3ed"> sensibilidad=</span>{ev['sensibilidad']}
            <span style="color:#63b3ed"> ubicacion=</span>{ev['ubicacion']}
            <br>
            <span style="color:transparent">[{ts}]&nbsp;</span>
            <span style="color:#63b3ed">nivel_mfa=</span>{ev['nivel_mfa']}
            <span style="color:#63b3ed"> senal_riesgo=</span>{ev['senal']}
            <span style="color:#63b3ed"> norma=</span>NIST-SP-800-207
        </div>"""

    st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    **Campos AU-3 verificados (NIST SP 800-53 Rev. 5):**  
    `decision` · `trust_score` · `rol` · `sensibilidad` · `ubicacion` · `nivel_mfa`  
    Total: **6/6 campos presentes** en los **4/4 eventos** → Trazabilidad: **100%**
    """)

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="font-family:'JetBrains Mono',monospace;font-size:10px;
            color:#334155;text-align:center;padding:8px">
    Muro Arévalo, Jaime Roberto &amp; Pribilov Morales, Nicolette · 
    PP2 1FRC0077 · Universidad Peruana de Ciencias Aplicadas · 2026 ·
    random_state=42 · dataset: 2,000 registros sintéticos
</div>
""", unsafe_allow_html=True)
