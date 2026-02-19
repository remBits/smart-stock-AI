# main.py — Smartstock IA: Intelligence Engine
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
        "sku_code": "sku", "id": "sku", "producto": "sku", "item": "sku"
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

    # 3. Verificación de columnas obligatorias
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
    if "unit_cost" not in df.columns: df["unit_cost"] = 1000.0
    df["unit_cost"] = df["unit_cost"].fillna(1000.0)

    # 5. Lógica de Ingeniería de Inventario
    z = norm.ppf(0.95)
    results = []

    for idx, row in df.iterrows():
        try:
            sku = str(row.get("sku", f"ITEM-{idx}"))
            demand = float(row["demand"])
            stock = float(row["stock"])
            lead_time = float(row["lead_time"])
            unit_cost = float(row["unit_cost"])

            # Cálculos de Reorden y Riesgo
            daily_demand = demand / 30
            expected_consumption = daily_demand * lead_time
            demand_std = max(0.1, daily_demand * 0.20)
            safety_stock = z * demand_std * np.sqrt(max(1.0, lead_time))
            rop = expected_consumption + safety_stock
            
            denom = demand_std * np.sqrt(max(1.0, lead_time))
            if denom <= 0:
                risk = 100.0 if stock < expected_consumption else 0.0
            else:
                z_score = (stock - expected_consumption) / denom
                risk = (1.0 - float(norm.cdf(z_score))) * 100.0
            
            risk_pct = float(max(0.0, min(100.0, round(risk, 1))))
            suggested_order = float(max(0.0, round(rop - stock, 1)))
            
            # Narrativa de IA
            if risk_pct > 80:
                interpretation = f"CRÍTICO: Stock agotable en {round(stock/daily_demand, 1) if daily_demand > 0 else 0} días."
            elif stock < rop:
                interpretation = f"REABASTECER: Nivel debajo del ROP ({int(rop)})."
            else:
                interpretation = "ÓPTIMO: Inventario bajo control."

            # Datos para el gráfico basados en la demanda real
            forecast = [round(max(0, np.random.normal(daily_demand, demand_std)), 1) for _ in range(7)]

            results.append({
                "sku": sku,
                "category": row.get("category", "General"),
                "stock": stock,
                "risk": risk_pct,
                "rop": round(rop, 1),
                "suggested_order": int(suggested_order),
                "savings": int(suggested_order * unit_cost * 0.20),
                "ai_interpretation": interpretation,
                "chart_data": forecast
            })
        except Exception as e_row:
            log.error(f"Error en fila {idx}: {e_row}")

    return JSONResponse(content=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)