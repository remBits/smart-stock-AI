# main.py — Smartstock IA: Intelligence Engine
# --- Instalación librerías para Backend y cálculos ---
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import pandas as pd
import numpy as np
import io
import os
import logging
from typing import List, Tuple
from scipy.stats import norm

# --- Instalación librería ML: Prophet ---
from prophet import Prophet

# --- Configuración de Registro (Logging) ---
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("inventory_app")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Inicialización de FastAPI ---
app = FastAPI(title="Smartstock IA - Command Center")

# Montar archivos estáticos para CSS y JS
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# HELPERS: Lectura de CSV y Procesamiento de Metadatos
# -----------------------------------------------------------------------------

def try_read_csv_bytes(b: bytes) -> pd.DataFrame:
    """Intenta leer bytes de un archivo subido en un DataFrame con sniff de separadores."""
    encodings = ["utf-8", "latin1", "cp1252"]
    separators = [None, ",", ";", "\t"]
    last_err = None

    for enc in encodings:
        try:
            text = b.decode(enc)
            for sep in separators:
                try:
                    if sep is None:
                        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
                    else:
                        df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")

                    if isinstance(df, pd.DataFrame) and not df.empty:
                        return df
                except Exception as e:
                    last_err = e
                    continue
        except Exception as e:
            last_err = e
            continue

    raise HTTPException(status_code=400, detail="No se pudo parsear el CSV.")


def normalize_and_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres y mapea alias comunes."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    rename_map = {
        "inventario": "stock", "cantidad": "stock", "existencias": "stock", "qty": "stock",
        "demanda": "demand", "consumo": "demand", "ventas": "demand", "sales": "demand",
        "leadtime": "lead_time", "entrega": "lead_time", "plazo_entrega": "lead_time",
        "precio": "unit_cost", "costo": "unit_cost", "unit_price": "unit_cost",
        "sku_code": "sku", "id": "sku", "producto": "sku", "item": "sku", "product_id": "sku",
        "lead_time_days": "lead_time", "cost": "unit_cost"
    }
    df.rename(columns=rename_map, inplace=True)
    return df


def require_columns_or_error(df: pd.DataFrame, required: List[str]) -> Tuple[bool, any]:
    """Verifica columnas. Si faltan, intenta asignar por posición."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        available_cols = list(df.columns)
        for i, req_col in enumerate(required):
            if req_col not in df.columns and i < len(available_cols):
                df.rename(columns={available_cols[i]: req_col}, inplace=True)

        still_missing = [c for c in required if c not in df.columns]
        if still_missing:
            return False, {"message": f"Faltan columnas: {', '.join(still_missing)}"}

    return True, ""


# -----------------------------------------------------------------------------
# PIPELINE DE PREDICCIÓN (MODO AVANZADO)
# -----------------------------------------------------------------------------

def detect_time_series(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Detecta si el CSV tiene una columna de fecha y demanda para construir serie temporal.
    Busca columnas tipo 'fecha', 'date' o 'ds'.
    """
    df = df.copy()
    date_cols = [c for c in df.columns if "fecha" in c or "date" in c or c == "ds"]
    if not date_cols:
        return None

    col = date_cols[0]
    df["ds"] = pd.to_datetime(df[col], errors="coerce")
    if df["ds"].isnull().all():
        return None

    # Para escalabilidad, de forma futura la función sería
    #if "demand" not in df.columns:
    #    return None

    # Para demo, se usa una heurística simple como fallback, 
    # lo que asegura su funcionamiento.
    if "demand" not in df.columns:
        df["demand"] = df["stock"] * 0.5  # heurística simple demo

    ts = df[["ds", "demand"]].dropna()
    if ts.shape[0] < 30:
        # muy pocos datos para una predicción razonable
        return None

    return ts


def preprocess_ts(df_ts: pd.DataFrame) -> pd.DataFrame:
    """Preprocesamiento básico de la serie temporal."""
    df_ts = df_ts.copy()
    df_ts = df_ts.sort_values("ds")
    df_ts = df_ts.set_index("ds").asfreq("D")
    df_ts["demand"] = df_ts["demand"].interpolate(limit_direction="both")

    q_low, q_high = df_ts["demand"].quantile([0.01, 0.99])
    df_ts["demand"] = df_ts["demand"].clip(q_low, q_high)

    return df_ts.reset_index()


def build_prophet_model() -> Prophet:
    """Modelo Prophet estándar para demanda de inventario."""
    m = Prophet(
        weekly_seasonality=True,
        daily_seasonality=False,
        yearly_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.2,
        interval_width=0.8,
    )
    m.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    return m


def forecast_with_prophet(df_ts: pd.DataFrame, horizon: int = 14) -> pd.DataFrame:
    """Entrena Prophet y genera pronóstico."""
    df_ts = df_ts.rename(columns={"demand": "y"})
    df_ts = df_ts.sort_values("ds")

    m = build_prophet_model()
    m.fit(df_ts)

    future = m.make_future_dataframe(periods=horizon)
    fcst = m.predict(future)

    return fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon)


def compute_risk_from_forecast(fcst: pd.DataFrame, stock: float, lead_time: float) -> float:
    """Calcula riesgo de quiebre usando demanda futura pronosticada."""
    ventana = int(max(1, round(lead_time)))
    ventana_df = fcst.head(ventana)

    demanda_esperada = ventana_df["yhat"].sum()
    demanda_max = ventana_df["yhat_upper"].sum()

    if stock >= demanda_max:
        riesgo = 0.05
    elif stock <= demanda_esperada:
        riesgo = 0.9
    else:
        ratio = (demanda_max - stock) / (demanda_max - demanda_esperada + 1e-6)
        riesgo = 0.1 + 0.8 * ratio

    return float(min(1.0, max(0.0, riesgo)))


# -----------------------------------------------------------------------------
# RUTAS
# -----------------------------------------------------------------------------

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Archivo sin nombre.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Archivo vacío.")

    # 1. Lectura Robusta
    df = try_read_csv_bytes(content)

    # 2. Normalización y Mapeo
    df = normalize_and_map_columns(df)

    # 3. Verificación de columnas obligatorias (modo simple)
    required = ["sku", "stock", "demand", "lead_time"]
    ok, msg = require_columns_or_error(df, required)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    # 4. Conversión Numérica
    for col in ["stock", "demand", "lead_time", "unit_cost"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Valores por defecto para cálculos seguros
    df["stock"] = df["stock"].fillna(0.0)
    df["demand"] = df["demand"].fillna(0.0)
    df["lead_time"] = df["lead_time"].fillna(7.0)
    if "unit_cost" not in df.columns:
        df["unit_cost"] = 1000.0
    df["unit_cost"] = df["unit_cost"].fillna(1000.0)

    # 5. Intentar construir serie temporal (modo avanzado)
    df_ts_global = detect_time_series(df)
    use_advanced = df_ts_global is not None

    if use_advanced:
        try:
            df_ts_global = preprocess_ts(df_ts_global)
            fcst_global = forecast_with_prophet(df_ts_global, horizon=14)
        except Exception as e:
            log.error(f"Error en modo avanzado Prophet: {e}")
            use_advanced = False
            fcst_global = None
    else:
        fcst_global = None

    # 6. Lógica de Ingeniería de Inventario por SKU
    z = norm.ppf(0.95)
    results = []

    for idx, row in df.iterrows():
        try:
            sku = str(row.get("sku", f"ITEM-{idx}"))
            demand = float(row["demand"])
            stock = float(row["stock"])
            lead_time = float(row["lead_time"])
            unit_cost = float(row["unit_cost"])

            # --- Cálculos base (modo simple) ---
            daily_demand = demand / 30 if demand > 0 else 0.0
            expected_consumption = daily_demand * lead_time
            demand_std = max(0.1, daily_demand * 0.20)
            safety_stock = z * demand_std * np.sqrt(max(1.0, lead_time))
            rop = expected_consumption + safety_stock

            denom = demand_std * np.sqrt(max(1.0, lead_time))
            if denom <= 0:
                risk_simple = 100.0 if stock < expected_consumption else 0.0
            else:
                z_score = (stock - expected_consumption) / denom
                risk_simple = (1.0 - float(norm.cdf(z_score))) * 100.0

            risk_simple_pct = float(max(0.0, min(100.0, round(risk_simple, 1))))
            suggested_order = float(max(0.0, round(rop - stock, 1)))

            # --- MODO AVANZADO: usar forecast real si está disponible ---
            if use_advanced and fcst_global is not None:
                risk_adv = compute_risk_from_forecast(fcst_global, stock, lead_time)
                risk_pct = round(risk_adv * 100.0, 1)
                chart_data = fcst_global["yhat"].round(1).tolist()
                mode = "advanced"
            else:
                risk_pct = risk_simple_pct
                chart_data = [round(max(0, np.random.normal(daily_demand, demand_std)), 1) for _ in range(7)]
                mode = "simple"
            
            # Impact level
            if risk_pct > 80:
                impact = "Alto"
            elif risk_pct > 50:
                impact = "Medio"
            else:
                impact = "Bajo"
            
            # Urgencia
            if risk_pct > 80:
                urgency = "Acción inmediata"
            elif stock < rop:
                urgency = "Reponer pronto"
            else:
                urgency = "Sin urgencia"


            # Narrativa de IA
            if risk_pct > 80:
                days_cover = round(stock / daily_demand, 1) if daily_demand > 0 else 0
                interpretation = f"CRÍTICO: Stock podría agotarse en ~{days_cover} días. Revisa reposición urgente."
            elif stock < rop:
                interpretation = f"REABASTECER: Nivel debajo del punto de reorden (ROP ≈ {int(rop)} unidades)."
            else:
                interpretation = "ÓPTIMO: Inventario bajo control según la demanda estimada."

            results.append({
                "sku": sku,
                "category": row.get("category", "General"),
                "stock": stock,
                "risk": risk_pct,
                "rop": round(rop, 1),
                "suggested_order": int(suggested_order),
                "savings": int(suggested_order * unit_cost * 0.20),
                "ai_interpretation": interpretation,
                "chart_data": chart_data,
                "mode": mode,
                "impact_level": impact,
                "reorder_urgency": urgency,
            })
        except Exception as e_row:
            log.error(f"Error en fila {idx}: {e_row}")

    return JSONResponse(content=results)


if __name__ == "__main__":
    import uvicorn
    # Puerto antiguo
    #uvicorn.run(app, host="127.0.0.1", port=8000)
    uvicorn.run(app, host="0.0.0.0", port=8000)
