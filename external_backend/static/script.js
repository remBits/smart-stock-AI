/**
 * Smartstock IA - Command Center Logic
 * Gestión completa de Dashboard, KPIs y Visualización
 */

/** 
 * DOM E INICIALIZACIÓN
*/

// Inicialización al cargar el DOM
document.addEventListener('DOMContentLoaded', () => {
    console.log("Smartstock IA: Command Center Operativo");
});

/**
 * Gestiona el cambio de pestañas en la interfaz
 * @param {string} tabId - El ID del contenido a mostrar
 * @param {HTMLElement} element - El item del menú clickeado
 */
function switchTab(tabId, element) {
    // Ocultar todos los contenidos
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.add('hidden');
    });
    
    // Mostrar el seleccionado
    const targetTab = document.getElementById(tabId);
    if (targetTab) {
        targetTab.classList.remove('hidden');
    }

    // Actualizar estado visual del menú
    document.querySelectorAll('.nav-item').forEach(nav => {
        nav.classList.remove('active');
    });
    element.classList.add('active');

    // Reseteo de barra navegadora
    window.scrollTo({ top: 0, behavior: "smooth" });
}

// Preparación para renderizado de Chart.js
let chartInstance = null; // resumen técnico
let executiveChartInstance = null; // resumen ejecutivo

/**
 * PREPARACIÓN FUNCIONES
 */

// Función para mostrar chat en casos de código 422
function showChat() {
    const chat = document.getElementById("chat-container");
    chat.classList.remove("hidden");
}

/**
 * Envía el archivo al servidor y procesa la respuesta
 */
async function processData() {
    const fileInput = document.getElementById('csvFile');
    const msgBox = document.getElementById('system-msg');
    
    if (!fileInput.files[0]) {
        alert("SISTEMA: Por favor cargue un archivo CSV o Excel válido.");
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        // Estado de carga visual
        msgBox.style.display = "block";
        msgBox.innerText = "IA: ANALIZANDO CADENA DE SUMINISTRO EN TIEMPO REAL...";

        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        // Manejo inteligente de respuestas

        if (response.status === 422) {
            const errorData = await response.json();
            handleColumnMappingError(errorData);
            return;
        }
        
        if (!response.ok) {
            throw new Error("Fallo en la respuesta del servidor.");
        }
        
        const data = await response.json();

        const sorted = [...data].sort((a, b) => b.risk - a.risk);
        const top3 = sorted.slice(0, Math.min(3, sorted.length));

        // Revelar secciones del Dashboard y ocultar mensaje de bienvenida
        document.getElementById('welcome-msg').classList.add('hidden');
        document.getElementById('macro-dashboard').classList.remove('hidden');
        document.getElementById('results-area').classList.remove('hidden');

        // 1. Actualizar KPIs Globales
        updateGlobalMetrics(data);

        // 2. Renderizar Detalle del primer producto detectado
        if (data.length > 0) {
            updateProductDetail(data[0]);
        }

        // 3. Poblar Tabla de SKUs
        populateInventoryTable(data);

        msgBox.innerText = "SISTEMA: ANÁLISIS DE RIESGO COMPLETADO EXITOSAMENTE.";

        // 4. Ocultar chat en los casos de éxito
        document.getElementById("chat-container").classList.add("hidden");
        document.getElementById("chat-messages").innerHTML = "";

        // 5. Gráfico para Resumen Ejecutivo (primera iteración, revisar si funcionó)
        generateExecutiveSummary(data);
        renderExecutiveChart(top3);

    } catch (error) {
        console.error("Error:", error);
        msgBox.innerText = "ERROR CRÍTICO: " + error.message;
    }
}

/**
 * MODO ASISTENTE
 */

// Chatbot para manejo de errores
// Conversación
function addChatMessage(text, sender = "bot") {
    const container = document.getElementById("chat-messages");
    if (!container) return;

    const message = document.createElement("div");
    message.classList.add("chat-message", sender);

    message.innerText = text;
    container.appendChild(message);

    // Scroll automático
    container.scrollTop = container.scrollHeight;
}

// Input usuario
function handleUserMessage() {
    const input = document.getElementById("chat-input");
    if (!input || !input.value.trim()) return;

    const userText = input.value.trim();

    addChatMessage(userText, "user");
    input.value = "";

    simulateBotResponse(userText);
}

// Simulación inteligente básica

let awaitingAdvancedConfirmation = false;
let fallbackCount = 0;

function simulateBotResponse(userText) {

    const text = userText
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .trim();

    const isDefinition =
    text.includes("que es") || text.includes("que son") ||
    text.includes("que significa") || text.includes("que significan") ||
    text.includes("definicion de") || text.includes("significado de") ||
    text.includes("define") || text.includes("defineme") ||
    text.includes("no entiendo que") || text.includes("no entiendo lo") ||
    text.includes("no comprendo") || text.includes("concepto");

    const isPossession = 
    text.includes("tengo") || text.includes("poseo") ||
    text.includes("tiene") || text.includes("posee") ||
    text.includes("contiene") || text.includes("incluye") ||
    text.includes("inclui") || text.includes("adjunte") ||
    text.includes("agregue");

    const isHelp =
    text.includes("ayuda") || text.includes("ayudame") ||
    text.includes("asistencia") || text.includes("asisteme") ||
    text.includes("auxilio") || text.includes("auxiliame") ||
    text.includes("me ayudes") || text.includes("me auxilies") ||
    text.includes("me asistas") || text.includes("ayudeme") || 
    text.includes("auxilieme") || text.includes("asistame");

    const isReal = 
    text.includes("humana") || text.includes("real") ||
    text.includes("humano") || text.includes("personal") ||
    text.includes("persona") || text.includes("personas") ||
    text.includes("verdadera") || text.includes("verdad") ||
    text.includes("personalizada") || text.includes("calidad");

    const isAdvanced =
    text.includes("activar") || text.includes("activa") ||
    text.includes("inicia") || text.includes("inicializa") ||
    text.includes("ia") || text.includes("llm") || 
    text.includes("chatgpt") || text.includes("gemini") || 
    text.includes("opeanai") || text.includes("anthropic") || 
    text.includes("deepseek") || text.includes("asistencia avanzada") ||
    text.includes("asistencia mas avanzada") || text.includes("modulo avanzado");

    const isLack = 
    text.includes("falta") || text.includes("faltaba") ||
    text.includes("faltaria");

    const columns = ["sku", "stock", "demand", "lead_time"];
    const mentioned = columns.filter(col => text.includes(col));

    let isFallback = true;
    let response = "No entendí tu solicitud. Puedes revisar el formato del CSV (o Excel) o pedirme un ejemplo.";

    if (text.includes("ejemplo") || text.includes("formato") || text.includes("ejemplos")) {
        response = "Un CSV válido debe contener las columnas: sku, stock, demand y lead_time. Ejemplo:\nsku,stock,demand,lead_time\nA123,50,30,7";
        awaitingAdvancedConfirmation = false;
        isFallback = false;
        fallbackCount = 0;
    }

    else if (
        text.includes("gracias") || text.includes("agradezco") || 
        text.includes("agradecida") || text.includes("agradecido")
        ) {
        response = "¡Con gusto! 😊 Puedes intentar subir nuevamente tu archivo cuando lo tengas listo.";
        awaitingAdvancedConfirmation = false;
        isFallback = false;
        fallbackCount = 0;
    }

    else if (
        (isPossession) &&
        (mentioned.length > 0 && mentioned.length < 4)
        ) {
        response = `Detecté que mencionas: ${mentioned.join(", ")}. Recuerda que tu CSV o Excel debe incluir las cuatro columnas obligatorias: sku, stock, demand y lead_time.`;
        awaitingAdvancedConfirmation = false;
        isFallback = false;
        fallbackCount+=0.8;
    }

    else if (
        isLack
    ) {
        response = 'Recuerda que tu CSV o Excel debe incluir las cuatro columnas obligatorias: sku, stock, demand y lead_time.'
        awaitingAdvancedConfirmation = false;
        isFallback = false;
        fallbackCount+= 0.8;
    }

    else if (awaitingAdvancedConfirmation && (text === "si" || text === "sí")) {
        response = "🔮 El módulo avanzado con LLM estará disponible próximamente en la versión Pro.";
        awaitingAdvancedConfirmation = false;
        isFallback = false;
        fallbackCount = 0;
    }
    
    else if (awaitingAdvancedConfirmation && text === "no") {
        response = "Entendido. Puedes revisar la sección de Preguntas Frecuentes para más información.";
        awaitingAdvancedConfirmation = false;
        isFallback = false;
        fallbackCount = 0;
    }
    

    else if (
        (
            text.includes("map") ||
            text.includes("mapeo") ||
            text.includes("mapear")
        )
        &&
        (
            text.includes("columna") ||
            text.includes("columnas")
        )
        ) {
        response = "Parece que el sistema no pudo mapear automáticamente tus columnas. ¿Quieres activar el módulo avanzado con IA?";
        awaitingAdvancedConfirmation = true;
        isFallback = false;
        fallbackCount = 0;
    }

    else if (
        isDefinition &&
        (isAdvanced)
    ) {
        response = "El modo de Asistencia avanzada es una conexión con una LLM externa para ayuda más personalizada.";
        awaitingAdvancedConfirmation = false;
        isFallback = false;
        fallbackCount = 0;
    }

    else if (isAdvanced) {
        response = "Recuerda que la conexión con una IA externa implica conversaciones con un servicio externo, la información que proporcione es de su responsabilidad. \nSmartStock IA se compromete a enviar solo la información estrictamente necesaria (lista de columnas y esquema requerido).\n🔮 El módulo avanzado con LLM estará disponible próximamente en la versión Pro.";
        awaitingAdvancedConfirmation = false;
        isFallback = false;
        fallbackCount = 0;
    }

    else if (
        isDefinition &&
        (
            text.includes("mapeo") ||
            text.includes("mapear") ||
            text.includes("map")
        )
    ) {
        response = "El mapeo de columnas es la forma en que el sistema reconoce los datos que usará para predecir. Si no puede reconocerlas, no puede predecir.";
        awaitingAdvancedConfirmation = false;
        isFallback = false;
        fallbackCount = 0;
    }

    else if (
        (isDefinition) &&
        (text.includes("columna") || text.includes("columnas"))
    ) {
        response = "Una columna es conjunto de datos ordenados de arriba hacia abajo y representan un mismo tipo de información. Por ejemplo, la columna SKU representa un conjunto de productos identificados por un código único, similar al RUT o DNI de las personas.";
        awaitingAdvancedConfirmation = false;
        isFallback = false;
        fallbackCount = 0;
    }

    else if (
        (isDefinition) &&
        (text.includes("sku") || text.includes("skus"))
    ) {
        response = "SKU (Stock Keeping Unit) es un identificador único para cada producto. Permite distinguir artículos en el inventario.";
        awaitingAdvancedConfirmation = false;
        isFallback = false;
        fallbackCount = 0;
    }
    
    else if (
        (isDefinition) &&
        text.includes("stock")
    ) {
        response = "Stock es la cantidad actual disponible de un producto en inventario.";
        awaitingAdvancedConfirmation = false;
        isFallback = false;
        fallbackCount = 0;
    }
    
    else if (
        (isDefinition) &&
        text.includes("demand")
    ) {
        response = "Demand representa la demanda estimada o promedio de ventas de un producto en un periodo determinado.";
        awaitingAdvancedConfirmation = false;
        isFallback = false;
        fallbackCount = 0;
    }
    
    else if (
        (isDefinition) &&
        text.includes("lead_time")
    ) {
        response = "Lead_time es el tiempo que tarda un proveedor en reabastecer un producto desde que se realiza el pedido.";
        awaitingAdvancedConfirmation = false;
        isFallback = false;
        fallbackCount = 0;
    }
    
    else if (
        (isDefinition) &&
        text.includes("csv")
    ) {
        response = "Un archivo CSV (Comma Separated Values) es un archivo de texto donde los datos están separados por comas. Se usa comúnmente para importar y exportar datos.\nSmartStock IA también acepta Excel.";
        awaitingAdvancedConfirmation = false;
        isFallback = false;
        fallbackCount = 0;
    }
    
    else if (
        (isDefinition) &&
        text.includes("excel")
    ) {
        response = "Excel es un programa de hojas de cálculo que permite organizar datos en tablas. Puedes exportar tus hojas como archivo CSV desde Excel, aunque SmartStockIA también acepta Excel.";
        awaitingAdvancedConfirmation = false;
        isFallback = false;
        fallbackCount = 0;
    }

    else if (
        (isDefinition) &&
        text.includes("rop")
    ) {
        response = "ROP es el Punto de Reposición e indica el nivel de stock al que un producto debería reponerse. Permite evitar quiebres de stock, buscando el equilibrio para evitar el sobrestock, y es una de las métricas que se podrán calcular cuando tu archivo pueda ser leído por el sistema.";
        awaitingAdvancedConfirmation = false;
        isFallback = false;
        fallbackCount = 0;
    }

    else if (
        (isPossession) &&
        (text.includes("sku") || text.includes("skus")) &&
        text.includes("stock") &&
        text.includes("demand")
    ) {
        response = "Parece que ya tienes parte del esquema correcto. Verifica que también incluyas la columna lead_time y que los nombres coincidan exactamente.";
        awaitingAdvancedConfirmation = false;
        isFallback = false;
        fallbackCount = 0;
    }

    else if (
        text.includes("no funciona") ||
        text.includes("no funciono") ||
        text.includes("no entiendo") ||
        text.includes("no entendi") ||
        text.includes("no comprendo") ||
        text.includes("no comprendi") ||
        text.includes("error")
    ) {
        response = "Lo siento mucho. ¿Deseas activar la asistencia más avanzada potenciada con IA?";
        awaitingAdvancedConfirmation = true;
        isFallback = false;
        fallbackCount = 0;
    }

    else if(
        text.includes("preguntas frecuentes") ||
        text.includes("faq")
    ) {
        response = "Si lo deseas, puedes dirigirte a la pestaña de Preguntas Frecuentes en la barra lateral."
        awaitingAdvancedConfirmation = false;
        isFallback = false;
        fallbackCount = 0;
    }

    else if(
        (
            text.includes("sinonimo") || text.includes("sinonimos"))
        ||
        (
            (text.includes("palabra") || text.includes("palabras")) 
            && 
            (text.includes("igual") || text.includes("iguales")))
    ) {
        response = "Puedes ver una lista de sinónimos en la pestaña de Preguntas Frecuentes. El sistema ya maneja un listado, sin embargo, el vocabulario humano es diverso y variado como la humanidad misma. Es posible que, a pesar del trabajo exhaustivo ya realizado, existan diferencias."
        awaitingAdvancedConfirmation = false;
        isFallback = false;
        fallbackCount+= 0.8;
    }

    else if(
        (isHelp)
        &&
        (isReal)
        ) {
        response = "Por ahora, puedes dirigirte a la página de Preguntas Frecuentes en la barra lateral. La asistencia humana no es una funcionalidad contemplada por el momento. Pero nuestra sección de Preguntas Frecuentes fue escrita por humanos y para humanos. 😊"
        awaitingAdvancedConfirmation = false;
        isFallback = false;
        fallbackCount+= 0.8;
    }

    else if(isHelp) {
        response = "¡Por supuesto! Puedes pedirme un ejemplo, preguntarme qué significan algunos conceptos o ir a la sección de Preguntas Frecuentes, escrita por humanos y para humanos. 😄";
        awaitingAdvancedConfirmation = false;
        isFallback = false;
        fallbackCount = 0;
    }

    else if (text.includes("hola")) {
        response = "¡Hola! Siempre es un gusto ayudarte. Puedes pedirme un ejemplo, preguntarme por algunos conceptos o ir a la sección de Preguntas Frecuentes."
        awaitingAdvancedConfirmation = false;
        isFallback = false;
        fallbackCount = 0;
    }

    else if (text.includes("adios") || text.includes("chao")) {
        response = "¡Adiós! Si necesitas más ayuda, aquí estaré. ¡Que tengas un excelente día!"
        awaitingAdvancedConfirmation = false;
        isFallback = false;
        fallbackCount = 0;
    }

    else if (
        text.includes("gracias a ti") || text.includes("no, a ti") || 
        text.includes("a ud") || text.includes("a usted")
        ) {
        response = "Siempre es un gusto ayudarte. Si necesitas más ayuda, aquí estaré. ¡Que tengas un excelente día!"
        awaitingAdvancedConfirmation = false;
        isFallback = false;
        fallbackCount = 0;
    }

    if (isFallback) {
        fallbackCount++;
    }
    
    if (fallbackCount >= 3) {
        response = "Parece que estamos teniendo dificultades para resolver tu caso. ¿Deseas activar la asistencia avanzada con IA?";
        awaitingAdvancedConfirmation = true;
        fallbackCount = 0;
    }

    setTimeout(() => {
        addChatMessage("🤖 " + response, "bot");
    }, 500);
}

// Guía de manejo de errores para el chatbot
function generateErrorGuidance(errorMessage) {

    if (errorMessage.includes("column_mapping_required")) {
        return "Tu archivo no contiene todas las columnas obligatorias. Asegúrate de incluir: sku, stock, demand y lead time.";
    }

    if (errorMessage.includes("Invalid file format")) {
        return "El archivo parece tener un formato inválido. Verifica que esté delimitado por comas y guardado como CSV UTF-8 o Excel.";
    }

    if (errorMessage.includes("Empty file")) {
        return "El archivo está vacío. Asegúrate de que contenga datos antes de subirlo.";
    }

    return "El sistema detectó un problema con el archivo. Revisa que cumpla con el formato esperado.";
}

// Reseteo de Dashboard para optimización de experiencia
function resetDashboardState() {
    // Limpiar tabla SKUs
    const tbody = document.getElementById('table-body');
    if (tbody) tbody.innerHTML = '';

    // Limpiar gráfico si existe
    if (chartInstance) {
        chartInstance.destroy();
        chartInstance = null;
    }

    // Limpiar detalle producto
    const productTitle = document.getElementById('product-title');
    if (productTitle) productTitle.innerText = '';

    const insightText = document.getElementById('insight-text');
    if (insightText) insightText.innerText = '';

    // Reset KPIs visibles
    const kpis = ['kpi-risk', 'kpi-capital', 'kpi-critical', 'kpi-demand'];
    kpis.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.innerText = '-';
    });

    // Reset chat después de cada error
    const chat = document.getElementById("chat-messages");
    if (chat) chat.innerHTML = "";
}

// Manejo de error 422
function handleColumnMappingError(errorData) {

    // Activar chat
    showChat();

    // Ocultar dashboards previos
    document.getElementById('macro-dashboard')?.classList.add('hidden');
    document.getElementById('results-area')?.classList.add('hidden');

    // Reset completo del estado visual
    resetDashboardState();

    // Volver a mostrar mensaje de bienvenida
    document.getElementById('welcome-msg')?.classList.remove('hidden');

    // FUNCIÓN FUTURA: 🧠 Preparación futura para integración LLM real
    if (errorData.details?.type === "llm_suggestion") {
        addChatMessage("🤖 " + errorData.details.message, "bot");
        return;
    }

    const missing = errorData.details?.missing || [];

    let message = "🤖 ¡Hola! SmartStock IA detectó un problema en tu archivo.\n\n";

    if (missing.length > 0) {
        message += "Faltan las siguientes columnas obligatorias:\n";
        missing.forEach(col => {
            message += "• " + col + "\n";
        });
    } else {
        message += "No se pudieron detectar columnas válidas.";
    }

    message += "\nPor favor verifica el formato del CSV o Excel y vuelve a intentarlo.";

    addChatMessage(message, "bot");

    // Guidance inteligente adicional
    const guidance = generateErrorGuidance(errorData.error || "");
    addChatMessage(guidance, "bot");
}


/**
 * Calcula y muestra métricas de toda la flota de SKUs
 */
function updateGlobalMetrics(data) {
    const totalRisk = data.reduce((acc, curr) => acc + curr.risk, 0);
    const avgRisk = Math.round(totalRisk / data.length);
    const criticalCount = data.filter(item => item.risk > 75).length;
    
    // Suponemos un costo operativo basado en los ahorros proyectados
    const totalSavings = data.reduce((acc, curr) => acc + curr.savings, 0);
    const totalDemand = data.length;

    document.getElementById('kpi-risk').innerText = avgRisk + "%";
    document.getElementById('kpi-capital').innerText = "$" + totalSavings.toLocaleString('es-CL');
    document.getElementById('kpi-critical').innerText = criticalCount;
    document.getElementById('kpi-demand').innerText = totalDemand + " SKUs";
    document.getElementById('executive-summary').innerHTML = 
    generateExecutiveSummary(data);
}

/**
 * Genera el texto del Resumen Ejecutivo (versión explicativa y accionable)
 */
function generateExecutiveSummary(data) {

    const totalRisk = data.reduce((acc, curr) => acc + curr.risk, 0);
    const avgRisk = Math.round(totalRisk / data.length);

    const criticalItems = data
        .filter(item => item.risk > 75)
        .sort((a, b) => b.risk - a.risk);

    const mediumItems = data
        .filter(item => item.risk > 40 && item.risk <= 75)
        .sort((a, b) => b.risk - a.risk);

    const totalSavings = data.reduce((acc, curr) => acc + curr.savings, 0);

    const topRiskItem = criticalItems.length > 0 
        ? criticalItems[0] 
        : data.reduce((max, item) => item.risk > max.risk ? item : max, data[0]);

    // Tomamos máximo 3 de cada categoría
    const top3Critical = criticalItems.slice(0, 3);
    const top3Medium = mediumItems.slice(0, 3);

    return `
    <p><strong>Resumen Ejecutivo</strong></p>

    <p>Este resumen presenta las principales conclusiones generadas a partir del análisis de su inventario.</p>

    <p>El riesgo promedio del inventario es de <strong>${avgRisk}%</strong>. 
    Este indicador refleja qué tan expuestos están sus productos a posibles quiebres de stock o reposiciones tardías si no se toman acciones preventivas.</p>

    <p>Se detectaron <strong>${criticalItems.length}</strong> productos en estado crítico y 
    <strong>${mediumItems.length}</strong> en riesgo medio.</p>

    ${top3Critical.length > 0 ? `
    <p><strong>Productos más urgentes (riesgo crítico):</strong></p>
    <ul>
        ${top3Critical.map(item => `
            <li><strong>${item.sku}</strong> — ${item.risk}% de riesgo</li>
        `).join("")}
    </ul>
    ` : ""}

    ${top3Medium.length > 0 ? `
    <p><strong>Productos a monitorear (riesgo medio):</strong></p>
    <ul>
        ${top3Medium.map(item => `
            <li><strong>${item.sku}</strong> — ${item.risk}% de riesgo</li>
        `).join("")}
    </ul>
    ` : ""}

    <p>El producto con mayor nivel de riesgo actualmente es 
    <strong>${topRiskItem.sku}</strong>. 
    Se recomienda evaluar reposición de aproximadamente 
    <strong>${Math.round(topRiskItem.suggested_order || 0)} unidades</strong> 
    dentro de los próximos 
    <strong>${Math.round(topRiskItem.days_cover || 0)} días</strong>, 
    para evitar quiebres de stock.</p>

    <p>Para otros productos, se recomienda ver los detalles en la sección de Resumen Técnico.</p>

    <p>Una gestión ineficiente del inventario puede impactar directamente en la liquidez del negocio. 
    En este caso, el capital potencialmente optimizable asciende a 
    <strong>$${totalSavings.toLocaleString('es-CL')}</strong>.</p>

    <p>A continuación, puede observar el pronóstico semanal de demanda de los tres productos más relevantes, lo que permite anticipar tendencias y planificar reposiciones con mayor precisión.</p>
    `;
}


/**
 * Actualiza la sección de resultados individuales (Gráfico y KPIs)
 */
function updateProductDetail(item) {
    // Producto seleccionado
    document.getElementById('product-title').innerText =
    `${item.sku} - ${item.category}`;

    // IDs basados en el HTML del Command Center
    document.getElementById('v-risk').innerText = item.risk + "%";
    document.getElementById('v-order').innerText = item.suggested_order;
    document.getElementById('v-save').innerText = "$" + item.savings.toLocaleString('es-CL');
    document.getElementById('insight-text').innerText = item.ai_interpretation;

    // Actualizar el gráfico lineal
    renderMainChart(item.chart_data, item.sku);
}

/**
 * Llena la tabla de la segunda pestaña
 */
function populateInventoryTable(data) {
    const tbody = document.getElementById('table-body');
    if (!tbody) return;
    
    tbody.innerHTML = '';

    data.forEach(item => {
        const row = document.createElement('tr');
        
        // Estilo según riesgo
        let riskColor = '#00f2fe'; // Cian (Estable)
        if (item.risk > 75) riskColor = '#ff4d4d'; // Rojo (Crítico)
        else if (item.risk > 40) riskColor = '#feca57'; // Amarillo (Preventivo)

        row.innerHTML = `
            <td><strong>${item.sku}</strong></td>
            <td>${item.category}</td>
            <td>${item.rop}</td>
            <td>${item.stock}</td>
            <td style="color: ${riskColor}; font-weight: bold;">${item.risk}%</td>
        `;

        // Al hacer clic, vuelve al panel y muestra ese producto
        row.style.cursor = "pointer";
        row.onclick = () => {
            updateProductDetail(item);
            switchTab('dashboard-view', document.querySelector('.nav-item'));
            window.scrollTo({ top: 0, behavior: "smooth" });
        };

        tbody.appendChild(row);
    });
}

/**
 * Renderizado de Chart.js principal
 */
function renderMainChart(points, sku) {
    const ctx = document.getElementById('mainChart').getContext('2d');
    
    if (chartInstance) {
        chartInstance.destroy();
    }

    const labels = points.map((_, i) => `Día ${i + 1}`);

    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: `Demanda Predicha: ${sku}`,
                data: points,
                borderColor: '#00f2fe',
                backgroundColor: 'rgba(0, 242, 254, 0.1)',
                borderWidth: 3,
                pointBackgroundColor: '#00f2fe',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: { 
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#888' }
                },
                x: { 
                    grid: { display: false },
                    ticks: { color: '#888' }
                }
            }
        }
    });
}

/**
 * Renderizado de Chart.js del Resumen Ejecutivo
 */
function renderExecutiveChart(top3) {

    if (!top3 || top3.length === 0) return;

    const canvas = document.getElementById('executiveChart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');

    if (executiveChartInstance) {
        executiveChartInstance.destroy();
    }

    const accessibleColors = [
        { border: '#D62828', dash: [] },
        { border: '#1D3557', dash: [6,6] },
        { border: '#2A9D8F', dash: [2,2] }
    ];

    const datasets = top3.map((item, index) => ({
        label: `${item.sku} (${item.risk}%)`,
        data: item.chart_data,
        borderColor: accessibleColors[index].border,
        borderDash: accessibleColors[index].dash,
        borderWidth: 3,
        fill: false,
        tension: 0.3
    }));

    executiveChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: top3[0].chart_data.map((_, i) => `Periodo ${i + 1}`),
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top' }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Demanda proyectada'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Periodo'
                    }
                }
            }
        }
    });
}

/**
 * MODO ACCESIBLE
 */
function toggleAccessibility() {
    document.body.classList.toggle('accessible-mode');
    
    // Guardar preferencia
    const isActive = document.body.classList.contains('accessible-mode');
    console.log("Modo accesible:", isActive);
}

/**
 * LISTENER DEL CHATBOT
 */
document.addEventListener("DOMContentLoaded", () => {
    const input = document.getElementById("chat-input");
    if (!input) return;

    input.addEventListener("keydown", function (event) {
        if (event.key === "Enter") {
            event.preventDefault();
            handleUserMessage();
        }
    });
});


/**
 * PESTAÑA DE USUARIOS
 */
// --- AUTH TABS ---
function switchAuthTab(tab) {
    const loginForm = document.getElementById("login-form");
    const registerForm = document.getElementById("register-form");
    const buttons = document.querySelectorAll(".tab-btn");
  
    buttons.forEach(btn => btn.classList.remove("active"));
  
    if (tab === "login") {
      loginForm.classList.remove("hidden");
      registerForm.classList.add("hidden");
      buttons[0].classList.add("active");
    } else {
      registerForm.classList.remove("hidden");
      loginForm.classList.add("hidden");
      buttons[1].classList.add("active");
    }
  }
  
  // --- LOGIN FALSO ---
  function fakeLogin() {
    document.getElementById("login-message").innerText =
      "Funcionalidad no disponible aún. A futuro aquí dirá: ¡Inicio de sesión exitoso!";
  
    document.getElementById("history-section").classList.remove("hidden");
  }
  
  // --- REGISTRO FALSO ---
  function fakeRegister() {
    document.getElementById("register-message").innerText =
      "Funcionalidad no disponible aún. A futuro aquí dirá: ¡Registro exitoso! Revisa tu correo para confirmar tu cuenta.";
  }
  
  // --- PLACEHOLDER DE FUNCIONALIDAD FUTURA ---
  function showFutureFeature() {
    alert("Funcionalidad futura en desarrollo.");
  }

  // --- BOTÓN DE DESCARGA FALSO ---
  function fakeDownload(button) {
    const row = button.closest("tr");
  
    let message = document.createElement("div");
    message.className = "small-message";
    message.innerText = "Funcionalidad futura: aquí se descargará un resumen ejecutivo en PDF.";
  
    row.appendChild(message);
  
    setTimeout(() => {
      message.remove();
    }, 3000);
  }