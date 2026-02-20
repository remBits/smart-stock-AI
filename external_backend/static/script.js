/**
 * Smartstock IA - Command Center Logic
 * Gestión completa de Dashboard, KPIs y Visualización
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
}

/**
 * Envía el archivo al servidor y procesa la respuesta
 */
async function processData() {
    const fileInput = document.getElementById('csvFile');
    const msgBox = document.getElementById('system-msg');
    
    if (!fileInput.files[0]) {
        alert("SISTEMA: Por favor cargue un archivo CSV válido.");
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

        if (!response.ok) throw new Error("Fallo en la respuesta del servidor.");

        const data = await response.json();

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

    } catch (error) {
        console.error("Error:", error);
        msgBox.innerText = "ERROR CRÍTICO: " + error.message;
    }
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
 * Genera el texto del Resumen Ejecutivo
 */
function generateExecutiveSummary(data) {

    const totalRisk = data.reduce((acc, curr) => acc + curr.risk, 0);
    const avgRisk = Math.round(totalRisk / data.length);

    const critical = data.filter(item => item.risk > 75).length;
    const medium = data.filter(item => item.risk > 40 && item.risk <= 75).length;

    const totalSavings = data.reduce((acc, curr) => acc + curr.savings, 0);

    // Producto más crítico
    const topRiskItem = data.reduce((max, item) => item.risk > max.risk ? item : max, data[0]);

    return `
    El riesgo promedio del inventario es de ${avgRisk}%, con ${critical} productos en estado crítico y ${medium} en riesgo medio.
    El sistema proyecta la demanda semanal y calcula puntos de reposición para recomendar cuándo y cuánto reordenar cada producto.
    El capital potencialmente optimizable asciende a $${totalSavings.toLocaleString('es-CL')}.
    🔴 Producto más crítico actual: <strong>${topRiskItem.sku}</strong> - ${topRiskItem.risk}% de riesgo.
    `;
}


/**
 * Actualiza la sección de resultados individuales (Gráfico y KPIs)
 */
function updateProductDetail(item) {
    // Producto seleccionado
    document.getElementById('product-title').innerText =
    `${item.sku} - ${item.category}`;

    // IDs basados en tu HTML de Command Center
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
        };

        tbody.appendChild(row);
    });
}

/**
 * Renderizado de Chart.js
 */
let chartInstance = null;

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
