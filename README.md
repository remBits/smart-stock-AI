# SmartStock AI

## Descripción
Modelo de predicción de demanda diaria para PYMEs usando Prophet.

## Estructura
- data_loader.py
- preprocessing.py
- model.py
- evaluation.py
- reporting.py
- app.py
- main.py

## Instalación
pip install -r requirements.txt

## Ejecutar API
uvicorn app:app --reload

## Endpoint
POST /predict
Subir archivo CSV con columnas Date y Units Sold.

_Creado para Hackathon NODO 2026 con 💜_