# ◈ Scaling Intelligence Dashboard

Dashboard de curvas de saturación por canal — **CPA sobre Altas**.  
Optimiza la inversión paid identificando dónde escala cada canal y dónde satura.

---

## Estructura del repositorio

```
scaling-app/
├── app.py                    # Aplicación Streamlit
├── requirements.txt          # Dependencias Python
├── Raw_data_curvas_v1.csv    # Dataset (añadir manualmente, no subir a Git)
└── README.md
```

---

## Setup local

### 1. Clona el repositorio
```bash
git clone https://github.com/TU_USUARIO/scaling-app.git
cd scaling-app
```

### 2. Crea un entorno virtual (recomendado)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Instala dependencias
```bash
pip install -r requirements.txt
```

### 4. Añade el CSV
Coloca `Raw_data_curvas_v1.csv` en la raíz del proyecto.  
El CSV debe contener las columnas: `Fecha`, `SupraCanal`, `Inversion_Fee`, `Leads_Brutos` (o `Leads Brutos`), `Ventas`, `Altas`.

### 5. Arranca la app
```bash
streamlit run app.py
```

Abre el navegador en `http://localhost:8501`

---

## Deploy en Streamlit Cloud

1. Sube el repo a GitHub (el CSV **no** hace falta subirlo si usas BigQuery en el futuro)
2. Ve a [share.streamlit.io](https://share.streamlit.io) → New app
3. Selecciona tu repo y rama, apunta a `app.py`
4. Deploy

---

## Lógica del modelo

| Elemento | Detalle |
|---|---|
| Modelo | Power law: `Ventas = a · Spend^b` |
| Variable objetivo | `Altas` (fin de funnel) |
| CR venta→alta | Calculado por canal desde el histórico |
| Canales sin coste | Excluidos del modelo, usados para target dinámico |
| Target CPA paid | `CPA_agregado × (1 + ratio_orgánico)` |

### Funnel contemplado
```
Lead Bruto → Lead Útil → Oportunidad → Venta → Alta ← métrica objetivo
```

### Canales excluidos del modelo paid
`Organico`, `Whatsapp`, `Manual`, `Otros_Origenes`  
→ Aportan Altas a coste cero y se usan para calcular el **target CPA dinámico**.

---

## Vistas de la app

- **Visión general** — Curvas de saturación comparativas (CPA Medio y Marginal) con posición actual de cada canal
- **Simulador por canal** — Proyecta escenarios de inversión con tabla de impacto y gráficos
- **Resumen ejecutivo** — Semáforo de saturación + mapa de eficiencia (bubble chart)
