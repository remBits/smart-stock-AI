# SmartStock IA

SmartStock IA es una aplicación web orientada al análisis predictivo de inventarios, diseñada para apoyar la toma de decisiones en gestión de stock mediante visualizaciones interpretables y asistencia contextual.

El sistema procesa datos de inventario (SKU, stock, demanda, punto de reposición) y genera:

* Resumen Ejecutivo interpretativo
* Resumen Técnico detallado por producto
* Reporte global de SKUs con KPIs agregados
* Visualización de pronóstico de los productos más críticos
* Asistencia contextual mediante chatbot

La aplicación se encuentra disponible en [smart-stock-ai.onrender.com](smart-stock-ai.onrender.com).

A continuación, encontramos las secciones de:

* Estructura del repositorio
* Transparencia
* Cómo usar la aplicación
* Funcionalidades principales
* Estado del Proyecto y Mejoras Futuras

Esto para ofrecer una visión completa sobre el producto.

---

## 📁 Estructura del repositorio

El proyecto está organizado de la siguiente manera:

```
data/
├── *.CSV                     # Archivos de datos de ejemplo

external_backend/
├── index.html                # Interfaz principal de la aplicación
├── main.py                   # Lógica de procesamiento y API
└── static/
    ├── script.js             # Lógica frontend (visualización y filtros)
    └── style.css             # Estilos y modo accesible

notebooks/
└── EDA_pipeline_MVP_inicial.ipynb   # Exploración de datos y prototipo inicial

.gitignore
README.md
requirements.txt
```

---

## 🤝 Transparencia

SmartStock IA utiliza modelos predictivos y componentes de IA generativa como herramientas de apoyo para la interpretación de resultados para el cliente. La IA no toma decisiones por el usuario ni reemplaza el criterio humano; su función es potenciar el análisis, facilitar la comprensión de los datos y acelerar la identificación de riesgos en inventario.

El diseño, las reglas de negocio, los umbrales de riesgo y la lógica de interpretación son definidos por el equipo de desarrollo; solamente el código fue potenciado con IA generativa. Además, la totalidad del código de la aplicación se encuentra disponible en este repositorio, promoviendo transparencia técnica y buenas prácticas de desarrollo. Creemos en una IA responsable: explicable, con resultados auditables y siempre bajo supervisión humana.

---

## 🎛 Cómo usar la aplicación

Para utilizar SmartStock IA, el usuario debe ingresar a la plataforma y cargar su archivo de inventario en formato CSV o Excel, siguiendo la estructura esperada (SKU, stock, demanda, punto de reposición, entre otros campos). Una vez cargado el archivo, el sistema valida la información. Si el archivo presenta inconsistencias o columnas faltantes, se activa el asistente de ayuda contextual (error 422), que orienta al usuario sobre cómo corregir el problema.

Cuando el archivo cumple con los requisitos, el sistema procesa los datos y genera automáticamente el análisis predictivo. El usuario visualizará primero un Resumen Ejecutivo con interpretación en lenguaje natural y un gráfico con los productos más críticos. Luego podrá explorar el Resumen Técnico, que muestra métricas detalladas por producto, incluyendo nivel de riesgo, compra sugerida y capital asociado.

En la pestaña de SKUs, el usuario puede revisar el inventario completo, aplicar filtros por nivel de riesgo y consultar definiciones de conceptos clave como SKU o ROP mediante el centro de definiciones interactivo. Si requiere mayor profundidad en la interpretación, puede activar el asistente contextual desde el flujo exitoso del análisis (la pestaña Panel).

---

## 🚀 Funcionalidades principales

### 📊 Resumen Ejecutivo

* Interpretación en lenguaje natural
* Visualización de los 3 productos con mayor riesgo
* Gráfico dinámico de pronóstico

### 🔎 Resumen Técnico

* Análisis individual por producto
* Nivel de riesgo
* Compra sugerida
* Impacto en capital

### 📦 Reporte de SKUs

* KPIs globales:
  * Riesgo promedio
  * Capital potencialmente optimizable
  * SKU críticos
  * Total de productos analizados
* Centro de definiciones accesible (SKU, ROP, Riesgo)
* Filtro dinámico por nivel de riesgo (Alto, Medio, Bajo)

### 🤖 Asistente SmartStock IA

* Manejo de errores de carga (código 422)
* Activación contextual tras análisis exitoso (código 200)
* Prototipo de módulo avanzado con LLM (versión Pro futura)

## 🏗 Arquitectura

* Frontend: HTML, CSS y JavaScript puro
* Backend: API externa para procesamiento predictivo basada en Python
* Visualización: Chart.js
* Diseño accesible con soporte para modo estándar y modo accesible
* Mantención de la funcionalidad de la página mediante automatización (UptimeRobot: funcionalidad de Monitor de PING)

## 🔐 Privacidad

La aplicación no almacena información personal del usuario.
El módulo avanzado con LLM (versión futura) utilizará servicios externos bajo consentimiento informado.

---

### 🔄 Estado del Proyecto y Mejoras Futuras

### 📌 Estado del Proyecto

MVP funcional con:

* Procesamiento de inventario
* Interpretación automatizada
* Visualización dinámica
* Prototipo de asistente contextual

### 📈 Próximas mejoras

* Integración completa con módulo LLM avanzado para el asistente
* Explicabilidad del backend actual mediante notebook
* 
* Mejora de filtros tipo Excel
* Implementación de capa de persistencia
* Exportación de reportes
* Dashboard comparativo histórico
---

_SmartStock IA_
_Creado para Hackathon NODO 2026 con 💜_