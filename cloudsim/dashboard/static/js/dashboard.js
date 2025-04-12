// Dashboard JavaScript

// Initialize state for the dashboard
const state = {
    running: false,
    latencyData: {
        labels: [],
        values: []
    },
    settings: {
        road: {
            weather: 'Clear',
            traffic_density: 0.5
        },
        network: {
            latency: 50,
            packet_loss: 2.0
        },
        computation: {
            processing_power: 0.6
        },
        algorithms: {
            perception: 'Algorithm A',
            planning: 'Algorithm B'
        }
    }
};

// DOM Elements
const elements = {
    // Settings inputs
    weather: document.getElementById('weather'),
    trafficDensity: document.getElementById('trafficDensity'),
    latency: document.getElementById('latency'),
    packetLoss: document.getElementById('packetLoss'),
    processingPower: document.getElementById('processingPower'),
    perception: document.getElementById('perception'),
    planning: document.getElementById('planning'),
    
    // Control buttons
    startBtn: document.getElementById('startBtn'),
    stopBtn: document.getElementById('stopBtn'),
    resetBtn: document.getElementById('resetBtn'),
    applySettingsBtn: document.getElementById('applySettingsBtn'),
    
    // Display elements
    simulationFrame: document.getElementById('simulationFrame'),
    riskValue: document.getElementById('riskValue'),
    currentRoundTripLatency: document.getElementById('currentRoundTripLatency'),
    avgObservationLatency: document.getElementById('avgObservationLatency'),
    avgActionLatency: document.getElementById('avgActionLatency')
};

// Initialize latency chart
const latencyChart = new Chart(
    document.getElementById('latencyChart'),
    {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Round-trip Latency (ms)',
                data: [],
                borderColor: '#0d6efd',
                backgroundColor: 'rgba(13, 110, 253, 0.1)',
                borderWidth: 2,
                tension: 0.2,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Time Steps',
                        color: '#f8f9fa'
                    },
                    ticks: {
                        color: '#f8f9fa'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Latency (ms)',
                        color: '#f8f9fa'
                    },
                    ticks: {
                        color: '#f8f9fa'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    beginAtZero: true
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#f8f9fa'
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.7)',
                    titleColor: '#f8f9fa',
                    bodyColor: '#f8f9fa',
                    borderColor: '#6c757d',
                    borderWidth: 1
                }
            }
        }
    }
);

// Event Listeners
elements.startBtn.addEventListener('click', startSimulation);
elements.stopBtn.addEventListener('click', stopSimulation);
elements.resetBtn.addEventListener('click', resetSimulation);
elements.applySettingsBtn.addEventListener('click', applySettings);

// Update input values from state
function updateInputsFromState() {
    elements.weather.value = state.settings.road.weather;
    elements.trafficDensity.value = state.settings.road.traffic_density;
    elements.latency.value = state.settings.network.latency;
    elements.packetLoss.value = state.settings.network.packet_loss;
    elements.processingPower.value = state.settings.computation.processing_power;
    elements.perception.value = state.settings.algorithms.perception;
    elements.planning.value = state.settings.algorithms.planning;
}

// Update state from input values
function updateStateFromInputs() {
    state.settings.road.weather = elements.weather.value;
    state.settings.road.traffic_density = parseFloat(elements.trafficDensity.value);
    state.settings.network.latency = parseInt(elements.latency.value);
    state.settings.network.packet_loss = parseFloat(elements.packetLoss.value);
    state.settings.computation.processing_power = parseFloat(elements.processingPower.value);
    state.settings.algorithms.perception = elements.perception.value;
    state.settings.algorithms.planning = elements.planning.value;
}

// Apply settings to the simulator
function applySettings() {
    updateStateFromInputs();
    
    fetch('/api/settings', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(state.settings)
    })
    .then(response => response.json())
    .then(data => {
        console.log('Settings applied successfully:', data);
    })
    .catch(error => {
        console.error('Error applying settings:', error);
    });
}

// Start the simulation
function startSimulation() {
    fetch('/api/control', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ command: 'start' })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            state.running = true;
            elements.startBtn.disabled = true;
            elements.stopBtn.disabled = false;
            elements.resetBtn.disabled = true;
            
            // Start polling for updates
            startPolling();
        }
    })
    .catch(error => {
        console.error('Error starting simulation:', error);
    });
}

// Stop the simulation
function stopSimulation() {
    fetch('/api/control', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ command: 'stop' })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            state.running = false;
            elements.startBtn.disabled = false;
            elements.stopBtn.disabled = true;
            elements.resetBtn.disabled = false;
            
            // Stop polling for updates
            stopPolling();
        }
    })
    .catch(error => {
        console.error('Error stopping simulation:', error);
    });
}

// Reset the simulation
function resetSimulation() {
    fetch('/api/control', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ command: 'reset' })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            // Reset chart data
            state.latencyData.labels = [];
            state.latencyData.values = [];
            latencyChart.data.labels = [];
            latencyChart.data.datasets[0].data = [];
            latencyChart.update();
            
            // Reset displayed values
            elements.riskValue.textContent = '0.15';
            elements.currentRoundTripLatency.textContent = '0 ms';
            elements.avgObservationLatency.textContent = '0 ms';
            elements.avgActionLatency.textContent = '0 ms';
            
            // Clear simulation frame
            elements.simulationFrame.src = '';
        }
    })
    .catch(error => {
        console.error('Error resetting simulation:', error);
    });
}

// Polling setup
let pollingInterval = null;
const POLLING_RATE = 100; // ms

function startPolling() {
    if (!pollingInterval) {
        pollingInterval = setInterval(updateDashboard, POLLING_RATE);
    }
}

function stopPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
}

// Update dashboard with latest simulation state
function updateDashboard() {
    fetch('/api/state')
        .then(response => response.json())
        .then(data => {
            // Update simulation frame if available
            if (data.simulation_frame) {
                elements.simulationFrame.src = `data:image/jpeg;base64,${data.simulation_frame}`;
            }
            
            // Update risk value
            elements.riskValue.textContent = data.current_risk.toFixed(2);
            
            // Update latency information
            if (data.round_trip_latency) {
                const latencyMs = (data.round_trip_latency * 1000).toFixed(2);
                elements.currentRoundTripLatency.textContent = `${latencyMs} ms`;
                
                // Add to chart
                state.latencyData.labels.push(state.latencyData.labels.length);
                state.latencyData.values.push(parseFloat(latencyMs));
                
                // Keep only the last 50 data points for better visualization
                if (state.latencyData.labels.length > 50) {
                    state.latencyData.labels.shift();
                    state.latencyData.values.shift();
                }
                
                // Update chart
                latencyChart.data.labels = state.latencyData.labels;
                latencyChart.data.datasets[0].data = state.latencyData.values;
                latencyChart.update();
            }
            
            // Calculate and display average observation latency
            if (data.observation_latencies && data.observation_latencies.length > 0) {
                const avgObsLatency = data.observation_latencies.reduce((a, b) => a + b, 0) / data.observation_latencies.length;
                elements.avgObservationLatency.textContent = `${(avgObsLatency * 1000).toFixed(2)} ms`;
            }
            
            // Calculate and display average action latency
            if (data.action_latencies && data.action_latencies.length > 0) {
                const avgActLatency = data.action_latencies.reduce((a, b) => a + b, 0) / data.action_latencies.length;
                elements.avgActionLatency.textContent = `${(avgActLatency * 1000).toFixed(2)} ms`;
            }
        })
        .catch(error => {
            console.error('Error updating dashboard:', error);
            // If there's an error, stop polling
            stopPolling();
        });
}

// Initialize the dashboard
function initDashboard() {
    updateInputsFromState();
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', initDashboard); 