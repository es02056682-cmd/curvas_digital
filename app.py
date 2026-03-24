import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Scaling Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@300;400&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing: antialiased;
}

/* ── Dark base ── */
.stApp { background: #000000; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem 2.5rem; max-width: 1400px; }

/* ── Sidebar dark ── */
[data-testid="stSidebar"] {
    background: #111111;
    border-right: 1px solid #2a2a2a;
    padding-top: 2rem;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #f5f5f7 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #1c1c1e !important;
    border: 1px solid #3a3a3c !important;
    border-radius: 10px;
    color: #f5f5f7 !important;
    font-size: 14px;
}
[data-testid="stSidebar"] .stMultiSelect > div > div {
    background: #1c1c1e !important;
    border: 1px solid #3a3a3c !important;
    border-radius: 10px;
}
[data-testid="stSidebar"] .stNumberInput > div > div > input {
    background: #1c1c1e !important;
    border: 1px solid #3a3a3c !important;
    border-radius: 10px;
    color: #f5f5f7 !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: #2a2a2c !important;
    border: 1px solid #3a3a3c !important;
}

/* ── Main content text ── */
.stMarkdown p, .stMarkdown div { color: #f5f5f7; }

/* ── Metric cards — glassmorphism dark ── */
.metric-card {
    background: rgba(28, 28, 30, 0.9);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    border: 1px solid #2a2a2c;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.metric-card:hover {
    box-shadow: 0 8px 32px rgba(0,0,0,0.6);
    transform: translateY(-2px);
    border-color: #3a3a3c;
}
.metric-label {
    font-size: 11px;
    font-weight: 600;
    color: #6e6e73;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-size: 2rem;
    font-weight: 600;
    color: #f5f5f7;
    letter-spacing: -0.03em;
    line-height: 1;
}
.metric-sub { font-size: 12px; color: #6e6e73; margin-top: 0.4rem; }
.metric-ok    { color: #34c759; }
.metric-warn  { color: #ff9f0a; }
.metric-alert { color: #ff3b30; }

/* ── Section titles ── */
.section-title {
    font-size: 22px;
    font-weight: 600;
    color: #f5f5f7;
    letter-spacing: -0.02em;
    margin: 2.5rem 0 0.25rem 0;
}
.section-sub {
    font-size: 14px;
    color: #6e6e73;
    margin-bottom: 1.5rem;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0a0a0a 0%, #111118 50%, #0a0a1a 100%);
    border-radius: 24px;
    padding: 3rem 3rem 2.5rem 3rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
    border: 1px solid #1c1c2a;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(0,113,227,0.2) 0%, transparent 65%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -30%; left: 20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(88,86,214,0.12) 0%, transparent 65%);
    pointer-events: none;
}
.hero-tag {
    font-size: 11px; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: #0a84ff; margin-bottom: 0.75rem;
}
.hero-title {
    font-size: 42px; font-weight: 600;
    color: #f5f5f7; letter-spacing: -0.04em;
    line-height: 1.1; margin-bottom: 0.75rem;
}
.hero-subtitle {
    font-size: 16px; color: #a1a1a6;
    font-weight: 300; max-width: 520px; line-height: 1.6;
}

.badge {
    display: inline-block;
    padding: 4px 12px; border-radius: 20px;
    font-size: 12px; font-weight: 500; letter-spacing: 0.02em;
}
.badge-green  { background: rgba(52,199,89,0.12);  color: #34c759; }
.badge-yellow { background: rgba(255,159,10,0.12); color: #ff9f0a; }
.badge-red    { background: rgba(255,59,48,0.12);  color: #ff3b30; }
.badge-grey   { background: rgba(142,142,147,0.12);color: #8e8e93; }

.plot-container {
    background: #111111;
    border-radius: 18px;
    border: 1px solid #2a2a2c;
    overflow: hidden;
    padding: 0.5rem;
}

/* ── Tables ── */
.sim-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.sim-table th {
    font-size: 11px; font-weight: 600; color: #6e6e73;
    letter-spacing: 0.08em; text-transform: uppercase;
    padding: 0.75rem 1rem; border-bottom: 1px solid #2a2a2c;
    text-align: right; background: #111111;
}
.sim-table th:first-child { text-align: left; }
.sim-table td {
    padding: 0.85rem 1rem; border-bottom: 1px solid #1c1c1e;
    color: #e5e5e7; text-align: right;
    font-family: 'DM Mono', monospace; font-size: 13px;
}
.sim-table td:first-child {
    text-align: left;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500; font-size: 13px; color: #f5f5f7;
}
.sim-table tr:last-child td { border-bottom: none; }
.sim-table tr:hover td { background: #1c1c1e; }
.sim-table-wrap {
    background: #111111; border-radius: 18px;
    border: 1px solid #2a2a2c; overflow: hidden; padding: 0.25rem 0;
}

/* ── Nav labels ── */
.nav-label {
    font-size: 11px; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: #6e6e73; padding: 1.5rem 0 0.5rem 0;
}

/* ── Tabs dark ── */
.stTabs [data-baseweb="tab-list"] {
    background: #1c1c1e; border-radius: 12px; padding: 4px; gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px; font-size: 13px; font-weight: 500;
    color: #6e6e73; padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: #2c2c2e !important;
    color: #f5f5f7 !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.4);
}

/* ── Streamlit native widgets dark override ── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: #1c1c1e !important;
    border: 1px solid #3a3a3c !important;
    color: #f5f5f7 !important;
    border-radius: 10px;
}
.stNumberInput > div > div > input {
    background: #1c1c1e !important;
    border: 1px solid #3a3a3c !important;
    color: #f5f5f7 !important;
    border-radius: 10px;
}
[data-baseweb="popover"] > div {
    background: #1c1c1e !important;
    border: 1px solid #3a3a3c !important;
}
[data-baseweb="menu"] { background: #1c1c1e !important; }
[data-baseweb="option"]:hover { background: #2c2c2e !important; }
[data-baseweb="option"] span { color: #f5f5f7 !important; }
.stSlider > div > div > div { background: #3a3a3c; }
.stSlider > div > div > div > div { background: #0a84ff; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DAYS_IN_MONTH       = 30
MIN_SPEND_THRESHOLD = 100
MIN_OBS             = 10
TARGET_CPA          = 200

# Canales sin coste controlable — excluidos del modelo, usados para target dinámico
CANALES_SIN_COSTE = ["Organico", "Whatsapp", "Manual", "Otros_Origenes", "Barrido"]

CANAL_COLORS = {
    "SEM_Marca_Pura":     "#7B2FBE",
    "SEM_Marca_Derivada": "#9B59B6",
    "SEM_Generico":       "#00B4D8",
    "SEM_Competencia":    "#FF69B4",
    "Paid_Social":        "#FF6B00",
    "Pmax":               "#3D5A80",
    "Display":            "#2ECC71",
    "Youtube":            "#CC0000",
    "Terceros":           "#2D6A4F",
    "Organico":           "#F4D03F",
    "Whatsapp":           "#A0856C",
    "Interno_MPA":        "#FFB3C6",
    "Resto":              "#B0BEC5",
    "Barrido":            "#546E7A",
    "null":               "#8E8E93",
}
DEFAULT_COLOR = "#8E8E93"

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    font=dict(family="DM Sans, -apple-system, sans-serif", size=12, color="#e5e5e7"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=16, r=16, t=48, b=16),
    xaxis=dict(
        showgrid=True, gridcolor="#2a2a2c", gridwidth=1,
        zeroline=False, linecolor="#3a3a3c",
        tickfont=dict(size=11, color="#6e6e73")
    ),
    yaxis=dict(
        showgrid=True, gridcolor="#2a2a2c", gridwidth=1,
        zeroline=False, linecolor="#3a3a3c",
        tickfont=dict(size=11, color="#6e6e73")
    ),
    legend=dict(
        bgcolor="rgba(17,17,17,0.9)", bordercolor="#2a2a2c",
        borderwidth=1, font=dict(size=11, color="#e5e5e7"), itemsizing="constant"
    ),
    hoverlabel=dict(
        bgcolor="#1c1c1e", bordercolor="#3a3a3c",
        font=dict(size=12, color="#f5f5f7", family="DM Sans")
    )
)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
def power_model(x, a, b):
    return a * (x ** b)

def monthly_ventas(spend_monthly, a, b):
    return DAYS_IN_MONTH * a * ((spend_monthly / DAYS_IN_MONTH) ** b)

def marginal_cpv(spend_daily, a, b):
    d = a * b * (spend_daily ** (b - 1))
    return 1.0 / d if d > 0 else np.inf

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_fit(filepath="Raw_data_curvas_v1.csv"):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # Build unified funnel columns from C2C + IB splits
    df["Leads_Brutos"] = (
        df.get("Leads_brutos_C2C", pd.Series(0, index=df.index)).fillna(0) +
        df.get("Leads_brutos_IB",  pd.Series(0, index=df.index)).fillna(0)
    )
    df["Leads_Utiles"] = (
        df.get("Leads_utiles_C2C", pd.Series(0, index=df.index)).fillna(0) +
        df.get("Leads_utiles_IB",  pd.Series(0, index=df.index)).fillna(0)
    )
    df["Ventas"]       = pd.to_numeric(df.get("Ventas_BI", 0), errors="coerce").fillna(0)
    df["Altas"]        = pd.to_numeric(df.get("Altas",     0), errors="coerce").fillna(0)
    df["Oportunidades"]= pd.to_numeric(df.get("Oportunidades_BI", 0), errors="coerce").fillna(0)
    df["Inversion_Fee"]= pd.to_numeric(df.get("Inversion_Fee", 0), errors="coerce").fillna(0)

    # Exclude rows with no real activity (fixed fee rows with no channel activity)
    df = df[
        df["SupraCanal"].notna() &
        (
            (df["Leads_Brutos"] > 0) |
            (df["Ventas"] > 0) |
            (df["Altas"] > 0)
        )
    ].copy()

    df_paid    = df[~df["SupraCanal"].isin(CANALES_SIN_COSTE)].copy()
    df_nocoste = df[ df["SupraCanal"].isin(CANALES_SIN_COSTE)].copy()

    # Real CR per paid channel
    cr = (
        df_paid.groupby("SupraCanal")
        .agg(V=("Ventas","sum"), A=("Altas","sum"),
             L=("Leads_Brutos","sum"), S=("Inversion_Fee","sum"))
        .reset_index()
    )
    cr["cr_venta_alta"]  = (cr["A"] / cr["V"].replace(0, np.nan)).fillna(0.8).round(4)
    cr["cr_lead_venta"]  = (cr["V"] / cr["L"].replace(0, np.nan)).fillna(0).round(4)
    cr["cpa_medio_hist"] = (cr["S"] / cr["A"].replace(0, np.nan)).round(2)

    # Dynamic organic ratio for target
    altas_paid    = df_paid["Altas"].sum()
    altas_nocoste = df_nocoste["Altas"].sum()
    ratio_org     = altas_nocoste / altas_paid if altas_paid > 0 else 0
    target_cpa_paid = round(TARGET_CPA * (1 + ratio_org), 2)

    # Monthly altas breakdown for context chart
    try:
        df["Mes"] = pd.to_datetime(df["Fecha"]).dt.to_period("M")
        altas_mes = df.copy()
        altas_mes["Tipo"] = altas_mes["SupraCanal"].apply(
            lambda c: "Sin_Coste" if c in CANALES_SIN_COSTE else "Paid"
        )
        altas_agg = (
            altas_mes.groupby(["Mes","Tipo"])["Altas"]
            .sum().unstack(fill_value=0).reset_index()
        )
        for col in ["Paid","Sin_Coste"]:
            if col not in altas_agg.columns:
                altas_agg[col] = 0
        altas_agg["Total"] = altas_agg["Paid"] + altas_agg["Sin_Coste"]
        altas_agg["CPA_max_paid"] = (
            TARGET_CPA * altas_agg["Total"] /
            altas_agg["Paid"].replace(0, np.nan)
        ).round(2)
    except Exception:
        altas_agg = pd.DataFrame()

    # Fit power model per paid channel
    results = {}
    skipped = []

    for canal, data in df_paid.groupby("SupraCanal"):
        data_fit = data[
            (data["Inversion_Fee"] > MIN_SPEND_THRESHOLD) &
            (data["Ventas"] > 0)
        ].copy()

        if len(data_fit) < MIN_OBS:
            skipped.append(canal)
            continue

        try:
            params, _ = curve_fit(
                power_model,
                data_fit["Inversion_Fee"],
                data_fit["Ventas"],
                bounds=([0, 0.01], [np.inf, 1]),
                p0=[0.01, 0.5],
                maxfev=20000
            )
            a, b = params

            y_pred = power_model(data_fit["Inversion_Fee"], a, b)
            ss_res = np.sum((data_fit["Ventas"] - y_pred) ** 2)
            ss_tot = np.sum((data_fit["Ventas"] - data_fit["Ventas"].mean()) ** 2)
            r2     = round(1 - ss_res / ss_tot if ss_tot > 0 else 0, 3)

            cr_row    = cr[cr["SupraCanal"] == canal]
            cr_va     = float(cr_row["cr_venta_alta"].values[0]) if len(cr_row) else 0.8
            avg_spend = data["Inversion_Fee"].mean() * DAYS_IN_MONTH

            ventas_act = monthly_ventas(avg_spend, a, b)
            altas_act  = ventas_act * cr_va
            cpa_med    = avg_spend / altas_act  if altas_act  > 0 else np.nan
            cpa_marg   = marginal_cpv(avg_spend / DAYS_IN_MONTH, a, b) / cr_va if cr_va > 0 else np.nan

            results[canal] = {
                "a": a, "b": b, "r2": r2, "n_obs": len(data_fit),
                "avg_monthly_spend": avg_spend,
                "cr_venta_alta": cr_va,
                "cpa_medio_actual":    round(cpa_med,  2) if not np.isnan(cpa_med)  else np.nan,
                "cpa_marginal_actual": round(cpa_marg, 2) if not np.isnan(cpa_marg) else np.nan,
            }
        except Exception:
            skipped.append(canal)

    params_df = (
        pd.DataFrame(results).T
        .reset_index().rename(columns={"index": "Canal"})
        .sort_values("avg_monthly_spend", ascending=False)
        .reset_index(drop=True)
    )

    return df, df_paid, df_nocoste, params_df, cr, target_cpa_paid, ratio_org, altas_agg, skipped

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def badge_html(label, kind):
    return f'<span class="badge badge-{kind}">{label}</span>'

def cpa_status(cpa_marg, target):
    if pd.isna(cpa_marg) or np.isinf(cpa_marg): return "grey",   "Sin datos"
    if cpa_marg > target * 1.3:                  return "red",    "🔴 Saturado"
    if cpa_marg > target:                        return "yellow", "🟡 En límite"
    return "green", "🟢 Con margen"

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner(""):
    try:
        df, df_paid, df_nocoste, params_df, cr_df, target_cpa_paid, ratio_org, altas_agg, skipped = \
            load_and_fit("Raw_data_curvas_v1.csv")
        data_ok = True
    except FileNotFoundError:
        data_ok = False

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">Performance Intelligence</div>
    <div class="hero-title">Scaling<br>Dashboard</div>
    <div class="hero-subtitle">
        Curvas de saturación por canal — CPA sobre Altas.<br>
        Optimiza cada euro antes de invertirlo.
    </div>
</div>
""", unsafe_allow_html=True)

if not data_ok:
    st.error("⚠️  No se encuentra `Raw_data_curvas_v1.csv`. Colócalo en la raíz del repositorio.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="font-size:20px;font-weight:600;color:#1d1d1f;letter-spacing:-0.02em;">◈ Scaling</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="font-size:12px;color:#6e6e73;margin-bottom:1.5rem;">Dashboard de saturación</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="nav-label">Vista</div>', unsafe_allow_html=True)
    vista = st.selectbox(
        "", ["Visión general", "Simulador por canal", "Resumen ejecutivo"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="nav-label">Targets</div>', unsafe_allow_html=True)
    target_override = st.number_input(
        "CPA objetivo agregado (€)", min_value=50, max_value=1000,
        value=TARGET_CPA, step=10
    )
    effective_target_paid = round(target_override * (1 + ratio_org), 2)

    st.markdown(f"""
    <div style="background:#1c1c1e;border-radius:12px;padding:1rem;margin-top:0.5rem;border:1px solid #2a2a2c;">
        <div style="font-size:11px;color:#6e6e73;font-weight:600;
                    letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.5rem;">
            Target dinámico paid
        </div>
        <div style="font-size:22px;font-weight:600;color:#0a84ff;letter-spacing:-0.03em;">
            {effective_target_paid:.0f} €
        </div>
        <div style="font-size:11px;color:#6e6e73;margin-top:0.25rem;">
            Orgánico subsidia {ratio_org:.0%} del volumen
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-label">Canales paid</div>', unsafe_allow_html=True)
    canales_disp = list(params_df["Canal"])
    canales_sel  = st.multiselect(
        "", canales_disp, default=canales_disp, label_visibility="collapsed"
    )

    st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)
    fecha_min = pd.to_datetime(df["Fecha"]).min().strftime("%d/%m/%Y")
    fecha_max = pd.to_datetime(df["Fecha"]).max().strftime("%d/%m/%Y")
    st.markdown(
        f'<div style="font-size:11px;color:#6e6e73;">'
        f'{len(params_df)} canales modelados · {len(df):,} registros<br>'
        f'{fecha_min} → {fecha_max}</div>',
        unsafe_allow_html=True
    )
    if skipped:
        st.markdown(
            f'<div style="font-size:11px;color:#ff9f0a;margin-top:0.5rem;">'
            f'Sin datos suficientes: {", ".join(skipped)}</div>',
            unsafe_allow_html=True
        )

params_filtered = params_df[params_df["Canal"].isin(canales_sel)]

# ═════════════════════════════════════════════════════════════════════════════
# VISTA 1 — VISIÓN GENERAL
# ═════════════════════════════════════════════════════════════════════════════
if vista == "Visión general":

    total_spend = params_filtered["avg_monthly_spend"].sum()
    avg_cpa     = params_filtered["cpa_medio_actual"].mean()
    canales_ok  = int((params_filtered["cpa_marginal_actual"] <= effective_target_paid).sum())
    canales_sat = int((params_filtered["cpa_marginal_actual"] >  effective_target_paid * 1.3).sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Inversión total / mes</div>
            <div class="metric-value">{total_spend/1000:.0f}k €</div>
            <div class="metric-sub">{len(params_filtered)} canales activos</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        cpa_cls = "metric-ok" if avg_cpa <= target_override else "metric-alert"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">CPA Medio paid</div>
            <div class="metric-value {cpa_cls}">{avg_cpa:.0f} €</div>
            <div class="metric-sub">Target: {target_override} €</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Canales con margen</div>
            <div class="metric-value metric-ok">{canales_ok}</div>
            <div class="metric-sub">CPA marginal bajo target</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        sat_cls = "metric-alert" if canales_sat > 0 else "metric-ok"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Canales saturados</div>
            <div class="metric-value {sat_cls}">{canales_sat}</div>
            <div class="metric-sub">CPA marginal &gt; {effective_target_paid:.0f} €</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Curvas de Saturación</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">CPA sobre Altas en función de la inversión mensual. Los puntos indican la posición actual de cada canal.</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["CPA Medio", "CPA Marginal"])

    def build_curves(col_y, clip_mult=5):
        fig = go.Figure()
        for _, row in params_filtered.iterrows():
            canal = row["Canal"]
            a, b  = row["a"], row["b"]
            cr_va = row["cr_venta_alta"]
            avg_s = row["avg_monthly_spend"]
            color = CANAL_COLORS.get(canal, DEFAULT_COLOR)
            spend_r  = np.linspace(avg_s * 0.3, avg_s * 1.7, 300)
            altas_c  = monthly_ventas(spend_r, a, b) * cr_va
            cpa_med  = np.where(altas_c > 0, spend_r / altas_c, np.nan)
            cpa_marg = np.array([marginal_cpv(s/DAYS_IN_MONTH,a,b)/cr_va for s in spend_r])
            cpa_marg = np.clip(cpa_marg, 0, effective_target_paid * clip_mult)
            y_vals   = cpa_med if col_y == "medio" else cpa_marg
            y_point  = row["cpa_medio_actual"] if col_y == "medio" else row["cpa_marginal_actual"]

            fig.add_trace(go.Scatter(
                x=spend_r, y=y_vals, mode="lines", name=canal,
                line=dict(color=color, width=2.5),
                customdata=np.stack([altas_c, cpa_marg if col_y=="medio" else cpa_med], axis=-1),
                hovertemplate=(
                    f"<b>{canal}</b><br>"
                    "Spend: %{x:,.0f} €/mes<br>"
                    f"CPA {'Medio' if col_y=='medio' else 'Marginal'}: %{{y:,.1f}} €<br>"
                    "Altas est.: %{customdata[0]:,.1f}<extra></extra>"
                )
            ))
            if not pd.isna(y_point):
                fig.add_trace(go.Scatter(
                    x=[avg_s], y=[y_point], mode="markers", showlegend=False,
                    marker=dict(size=10, color=color, symbol="circle",
                                line=dict(width=2.5, color="white")),
                ))
        fig.add_hline(y=effective_target_paid, line_dash="dash",
                      line_color="#ff9f0a", line_width=1.5,
                      annotation_text=f"Target paid {effective_target_paid:.0f} €",
                      annotation_font_color="#ff9f0a", annotation_font_size=11)
        fig.add_hline(y=target_override, line_dash="dot",
                      line_color="#ff3b30", line_width=1.5,
                      annotation_text=f"Target agregado {target_override} €",
                      annotation_font_color="#ff3b30", annotation_font_size=11)
        fig.update_layout(**PLOTLY_LAYOUT, height=480,
                          xaxis_title="Inversión mensual (€)",
                          yaxis_title="CPA sobre Altas (€)",
                          hovermode="closest")
        return fig

    with tab1:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.plotly_chart(build_curves("medio"), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.plotly_chart(build_curves("marginal"), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# VISTA 2 — SIMULADOR
# ═════════════════════════════════════════════════════════════════════════════
elif vista == "Simulador por canal":

    st.markdown('<div class="section-title">Simulador de inversión</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Proyecta el impacto de cambiar el presupuesto de un canal y compara múltiples escenarios a la vez.</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([2, 2, 2])
    with col_a:
        canal_sel = st.selectbox("Canal", list(params_filtered["Canal"]))
    with col_b:
        spend_manual = st.number_input(
            "Presupuesto base / mes (€) — 0 = usar histórico real",
            min_value=0, max_value=500000, value=0, step=1000
        )
    with col_c:
        delta_max = st.slider(
            "Rango de escenarios (€ ±)", 5000, 50000, 15000, step=5000
        )

    if canal_sel not in params_df["Canal"].values:
        st.warning("Canal no disponible en el modelo.")
        st.stop()

    r      = params_df[params_df["Canal"] == canal_sel].iloc[0]
    a, b   = r["a"], r["b"]
    cr_va  = r["cr_venta_alta"]
    color  = CANAL_COLORS.get(canal_sel, DEFAULT_COLOR)
    avg_s  = float(spend_manual) if spend_manual > 0 else r["avg_monthly_spend"]

    ventas_base = monthly_ventas(avg_s, a, b)
    altas_base  = ventas_base * cr_va
    cpa_base    = avg_s / altas_base  if altas_base  > 0 else np.nan
    cpa_mg_base = marginal_cpv(avg_s / DAYS_IN_MONTH, a, b) / cr_va

    # KPIs
    st.markdown('<div style="height:1rem;"></div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    kind, _ = cpa_status(cpa_mg_base, effective_target_paid)
    mg_cls  = {"green":"metric-ok","yellow":"metric-warn","red":"metric-alert","grey":""}.get(kind,"")
    cpa_cls = "metric-ok" if not np.isnan(cpa_base) and cpa_base <= effective_target_paid else "metric-alert"

    with k1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Spend base</div><div class="metric-value">{avg_s/1000:.1f}k €</div><div class="metric-sub">/ mes</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Altas estimadas</div><div class="metric-value">{altas_base:.1f}</div><div class="metric-sub">CR venta→alta: {cr_va:.1%}</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">CPA Medio</div><div class="metric-value {cpa_cls}">{cpa_base:.0f} €</div><div class="metric-sub">Target: {effective_target_paid:.0f} €</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">CPA Marginal</div><div class="metric-value {mg_cls}">{cpa_mg_base:.0f} €</div><div class="metric-sub">Coste del € extra</div></div>', unsafe_allow_html=True)
    with k5:
        st.markdown(f'<div class="metric-card"><div class="metric-label">R² del modelo</div><div class="metric-value">{r["r2"]:.3f}</div><div class="metric-sub">Bondad de ajuste</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)

    # Scenario table
    step = delta_max // 4
    deltas = [-delta_max, -delta_max + step, -step, 0, step, delta_max - step, delta_max]

    sim_rows = []
    for d in deltas:
        s      = max(avg_s + d, 1)
        v      = monthly_ventas(s, a, b)
        al     = v * cr_va
        cpa_m  = s / al if al > 0 else np.nan
        cpa_mg = marginal_cpv(s / DAYS_IN_MONTH, a, b) / cr_va
        ok     = cpa_mg <= effective_target_paid
        sim_rows.append({"delta": d, "spend": s, "altas": al,
                         "cpa_med": cpa_m, "cpa_marg": cpa_mg, "ok": ok})

    rows_html = ""
    for sr in sim_rows:
        delta_str = f"+{sr['delta']:,.0f} €" if sr["delta"] >= 0 else f"{sr['delta']:,.0f} €"
        bdg       = badge_html("✅ OK", "green") if sr["ok"] else badge_html("⚠️ Saturado", "red")
        bold      = "font-weight:600;background:#1c1c1e;" if sr["delta"] == 0 else ""
        rows_html += f"""
        <tr style="{bold}">
            <td>{delta_str}</td>
            <td>{sr['spend']:,.0f} €</td>
            <td>{sr['altas']:.1f}</td>
            <td>{sr['cpa_med']:,.1f} €</td>
            <td>{sr['cpa_marg']:,.1f} €</td>
            <td>{bdg}</td>
        </tr>"""

    st.markdown(f"""
    <div class="sim-table-wrap">
        <table class="sim-table">
            <thead><tr>
                <th>Δ Presupuesto</th><th>Spend / mes</th><th>Altas est.</th>
                <th>CPA Medio</th><th>CPA Marginal</th><th>Estado</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)

    # Charts
    spend_range = np.linspace(max(avg_s * 0.15, 500), avg_s * 2.0, 500)
    ventas_c    = monthly_ventas(spend_range, a, b)
    altas_c     = ventas_c * cr_va
    cpa_med_c   = np.where(altas_c > 0, spend_range / altas_c, np.nan)
    cpa_marg_c  = np.array([marginal_cpv(s/DAYS_IN_MONTH,a,b)/cr_va for s in spend_range])
    cpa_marg_c  = np.clip(cpa_marg_c, 0, effective_target_paid * 6)

    fig_sim = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Altas estimadas — {canal_sel}", f"CPA — {canal_sel}"),
        horizontal_spacing=0.10
    )

    for col_idx, (y_curve, y_base, y_label) in enumerate([
        (altas_c,   altas_base,  "Altas"),
        (cpa_med_c, cpa_base,    "CPA Medio")
    ], 1):
        fig_sim.add_trace(go.Scatter(
            x=spend_range, y=y_curve, mode="lines", name=y_label,
            line=dict(color=color, width=2.5),
        ), row=1, col=col_idx)
        fig_sim.add_trace(go.Scatter(
            x=[avg_s], y=[y_base], mode="markers",
            name="Base" if col_idx == 1 else None,
            showlegend=(col_idx == 1),
            marker=dict(size=12, color=color, symbol="circle",
                        line=dict(width=2.5, color="white")),
        ), row=1, col=col_idx)
        for sr in sim_rows:
            if sr["delta"] == 0: continue
            c_dot  = "#34c759" if sr["ok"] else "#ff3b30"
            y_val  = sr["altas"] if col_idx == 1 else sr["cpa_med"]
            fig_sim.add_trace(go.Scatter(
                x=[sr["spend"]], y=[y_val], mode="markers", showlegend=False,
                marker=dict(size=8, color=c_dot, symbol="diamond"),
                hovertemplate=f"Δ{sr['delta']:+,.0f}€<br>{y_label}: {y_val:.1f}<extra></extra>"
            ), row=1, col=col_idx)

    # CPA marginal overlay on right panel
    fig_sim.add_trace(go.Scatter(
        x=spend_range, y=cpa_marg_c, mode="lines", name="CPA Marginal",
        line=dict(color="#ff9f0a", width=2, dash="dot"),
    ), row=1, col=2)
    fig_sim.add_hline(y=effective_target_paid, line_dash="dash",
                      line_color="#ff9f0a", line_width=1.5, row=1, col=2,
                      annotation_text=f"Target paid {effective_target_paid:.0f}€",
                      annotation_font_color="#ff9f0a", annotation_font_size=11)
    fig_sim.add_hline(y=target_override, line_dash="dot",
                      line_color="#ff3b30", line_width=1.5, row=1, col=2,
                      annotation_text=f"Target agregado {target_override}€",
                      annotation_font_color="#ff3b30", annotation_font_size=11)

    layout2 = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis","yaxis")}
    fig_sim.update_xaxes(title_text="Inversión mensual (€)",
                         showgrid=True, gridcolor="#2a2a2c", zeroline=False,
                         tickfont=dict(color="#6e6e73"))
    fig_sim.update_yaxes(showgrid=True, gridcolor="#2a2a2c", zeroline=False,
                         tickfont=dict(color="#6e6e73"))
    fig_sim.update_layout(**layout2, height=440,
                          legend=dict(orientation="h", y=-0.18))

    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.plotly_chart(fig_sim, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# VISTA 3 — RESUMEN EJECUTIVO
# ═════════════════════════════════════════════════════════════════════════════
elif vista == "Resumen ejecutivo":

    st.markdown('<div class="section-title">Posicionamiento en curva</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Estado actual de cada canal paid respecto a su punto de saturación y el target de CPA.</div>', unsafe_allow_html=True)

    def calc_headroom(row):
        try:
            a_, b_ = row["a"], row["b"]
            cr_va  = row["cr_venta_alta"]
            s0     = row["avg_monthly_spend"] / DAYS_IN_MONTH
            s_star = (1 / (a_ * b_ * cr_va * effective_target_paid)) ** (1 / (b_ - 1))
            return round((s_star - s0) / s0 * 100, 1)
        except Exception:
            return np.nan

    rows_exec = []
    for _, row in params_filtered.iterrows():
        hw           = calc_headroom(row)
        kind, label  = cpa_status(row["cpa_marginal_actual"], effective_target_paid)
        rows_exec.append({
            "canal": row["Canal"], "spend": row["avg_monthly_spend"],
            "cpa_med": row["cpa_medio_actual"], "cpa_marg": row["cpa_marginal_actual"],
            "headroom": hw, "b": row["b"], "r2": row["r2"],
            "kind": kind, "label": label
        })
    rows_exec.sort(key=lambda x: x["spend"], reverse=True)

    table_html = ""
    for rx in rows_exec:
        hw_str   = f"+{rx['headroom']:.1f}%" if (not pd.isna(rx["headroom"]) and rx["headroom"] > 0) \
                   else (f"{rx['headroom']:.1f}%" if not pd.isna(rx["headroom"]) else "—")
        hw_color = "#34c759" if (not pd.isna(rx["headroom"]) and rx["headroom"] > 0) else "#ff3b30"
        dot      = {"green":"#34c759","yellow":"#ff9f0a","red":"#ff3b30","grey":"#8e8e93"}.get(rx["kind"],"#8e8e93")
        cpa_med_str  = f"{rx['cpa_med']:,.1f} €"  if not pd.isna(rx["cpa_med"])  else "—"
        cpa_marg_str = f"{rx['cpa_marg']:,.1f} €" if not pd.isna(rx["cpa_marg"]) else "—"
        table_html += f"""
        <tr>
            <td>
                <span style="display:inline-block;width:8px;height:8px;border-radius:50%;
                             background:{dot};margin-right:8px;vertical-align:middle;"></span>
                <span style="color:#f5f5f7;">{rx['canal']}</span>
            </td>
            <td>{rx['spend']:,.0f} €</td>
            <td>{cpa_med_str}</td>
            <td>{cpa_marg_str}</td>
            <td style="color:{hw_color};font-weight:500;">{hw_str}</td>
            <td>{rx['b']:.3f}</td>
            <td>{rx['r2']:.3f}</td>
            <td>{badge_html(rx['label'], rx['kind'])}</td>
        </tr>"""

    st.markdown(f"""
    <div class="sim-table-wrap">
        <table class="sim-table">
            <thead><tr>
                <th style="text-align:left;">Canal</th>
                <th>Inversión / mes</th><th>CPA Medio</th><th>CPA Marginal</th>
                <th>Headroom</th><th>b</th><th>R²</th><th>Estado</th>
            </tr></thead>
            <tbody>{table_html}</tbody>
        </table>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)

    # Bubble chart
    st.markdown('<div class="section-title">Mapa de eficiencia</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Cada burbuja es un canal. Tamaño = inversión mensual. Zona verde = por debajo del target paid.</div>', unsafe_allow_html=True)

    fig_b = go.Figure()
    fig_b.add_hrect(
        y0=0, y1=effective_target_paid,
        fillcolor="rgba(52,199,89,0.05)", line_width=0,
        annotation_text="Zona eficiente",
        annotation_position="top left",
        annotation_font_color="#34c759", annotation_font_size=11
    )
    for rx in rows_exec:
        if pd.isna(rx["cpa_marg"]): continue
        c = CANAL_COLORS.get(rx["canal"], DEFAULT_COLOR)
        fig_b.add_trace(go.Scatter(
            x=[rx["spend"]], y=[rx["cpa_marg"]],
            mode="markers+text", name=rx["canal"],
            text=[rx["canal"]], textposition="top center",
            textfont=dict(size=11, color="#1d1d1f"),
            marker=dict(
                size=max(12, min(50, rx["spend"] / 2000)),
                color=c, opacity=0.85,
                line=dict(width=2, color="white")
            ),
            hovertemplate=(
                f"<b>{rx['canal']}</b><br>"
                f"Spend: {rx['spend']:,.0f} €/mes<br>"
                f"CPA Marginal: {rx['cpa_marg']:,.1f} €<br>"
                f"CPA Medio: {rx['cpa_med']:,.1f} €<extra></extra>"
            )
        ))
    fig_b.add_hline(
        y=effective_target_paid, line_dash="dash",
        line_color="#ff9f0a", line_width=1.5,
        annotation_text=f"Target paid {effective_target_paid:.0f} €",
        annotation_font_color="#ff9f0a", annotation_font_size=11
    )
    fig_b.update_layout(
        **PLOTLY_LAYOUT, height=480,
        xaxis_title="Inversión mensual (€)",
        yaxis_title="CPA Marginal (€)",
        showlegend=False, hovermode="closest"
    )
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.plotly_chart(fig_b, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)
