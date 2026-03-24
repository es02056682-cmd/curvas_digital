# ◈ Scaling Intelligence Dashboard

Dashboard de curvas de saturación por canal — **CPA sobre Altas**.

## Archivos del repositorio

```
scaling-app/
├── app.py                      # Aplicación Streamlit
├── requirements.txt            # Dependencias
├── Raw_data_curvas_v1.csv      # Dataset histórico
└── README.md
```

## Deploy en Streamlit Cloud

1. Sube este repositorio a GitHub (puede ser privado)
2. Ve a [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Selecciona tu repo, rama `main`, archivo `app.py`
4. **Deploy** — listo en ~2 minutos

## Uso local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Estructura del CSV

El archivo `Raw_data_curvas_v1.csv` debe tener estas columnas:

| Columna | Tipo | Descripción |
|---|---|---|
| `Fecha` | date | Fecha del registro |
| `SupraCanal` | string | Canal de marketing |
| `Inversion_Fee` | float | Inversión con fee |
| `Leads_brutos_C2C` | int | Leads brutos C2C |
| `Leads_brutos_IB` | int | Leads brutos IB |
| `Ventas_BI` | int | Ventas |
| `Altas` | int | Altas — métrica objetivo |

## Canales excluidos del modelo paid
`Organico`, `Whatsapp`, `Manual`, `Otros_Origenes`, `Barrido`
Se usan para calcular el **target CPA dinámico paid**.

## Modelo
`Ventas = a · Spend^b` con `0 < b < 1` (rendimientos decrecientes)
