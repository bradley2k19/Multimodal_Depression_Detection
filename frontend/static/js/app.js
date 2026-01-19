// ================================
// SOCKET.IO CONNECTION
// ================================
const socket = io({
    transports: ['polling'],
    upgrade: false,
    reconnection: true,
    reconnectionDelay: 1000,
    reconnectionDelayMax: 5000,
    reconnectionAttempts: 5,
    timeout: 60000
});

// ================================
// DOM ELEMENTS
// ================================
const webcam = document.getElementById("webcam");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const startBtn = document.getElementById("start-btn");
const analyzeBtn = document.getElementById("analyze-btn");
const downloadBtn = document.getElementById("download-btn");

const bufferText = document.getElementById("buffer-text");
const bufferProgress = document.getElementById("buffer-progress");

const riskPercentage = document.getElementById("risk-percentage");
const riskLevel = document.getElementById("risk-level");

const analysisCountEl = document.getElementById("analysis-count");
const avgRiskEl = document.getElementById("avg-risk");
const lastUpdateEl = document.getElementById("last-update");

const auList = document.getElementById("au-list");
const explanationText = document.getElementById("explanation-text");

const loadingOverlay = document.getElementById("loading-overlay");
const loadingText = loadingOverlay.querySelector("p");

// ================================
// STATE
// ================================
let stream = null;
let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;
let analysisCount = 0;
let riskHistory = [];
let uploadInProgress = false;
let currentAnalysis = null;

// ================================
// CAMERA
// ================================
startBtn.addEventListener("click", async () => {
    try {
        // Request BOTH video AND audio
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: true,
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });
        webcam.srcObject = stream;
        analyzeBtn.disabled = false;
        startBtn.textContent = "📹 Camera Started";
        startBtn.disabled = true;
    } catch (error) {
        alert("Camera/Mic access denied: " + error.message);
    }
});

// ================================
// START RECORDING
// ================================
analyzeBtn.addEventListener("click", () => {
    if (isRecording) {
        stopRecordingAndAnalyze();
    } else {
        startRecording();
    }
});

function startRecording() {
    recordedChunks = [];
    
    const options = { 
        mimeType: "video/webm;codecs=vp9",
        audioBitsPerSecond: 128000
    };
    
    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        options.mimeType = "video/webm;codecs=vp8";
    }
    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        options.mimeType = "video/webm";
    }

    mediaRecorder = new MediaRecorder(stream, options);

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            recordedChunks.push(event.data);
        }
    };

    mediaRecorder.onstop = () => {
        console.log("Recording stopped");
    };

    mediaRecorder.start();
    isRecording = true;
    analyzeBtn.textContent = "⏹️ Stop & Analyze";
    bufferText.textContent = "🔴 Recording...";
    bufferProgress.style.width = "100%";
}

function stopRecordingAndAnalyze() {
    if (!mediaRecorder) return;

    mediaRecorder.stop();
    isRecording = false;
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "🔍 Analyze Now";

    loadingOverlay.style.display = "flex";
    loadingText.textContent = "Preparing video for analysis...";

    setTimeout(() => {
        sendVideoToBackend();
    }, 1000);
}

// ================================
// SEND VIDEO TO BACKEND
// ================================
async function sendVideoToBackend() {
    if (uploadInProgress) {
        console.log("⚠️ Upload already in progress, ignoring");
        return;
    }
    
    uploadInProgress = true;
    
    try {
        loadingText.textContent = "Uploading video...";

        const blob = new Blob(recordedChunks, { type: "video/webm" });
        console.log(`📹 Video blob size: ${blob.size} bytes`);

        if (blob.size === 0) {
            throw new Error("Recorded video is empty. Please try recording again.");
        }

        const CHUNK_SIZE = 1024 * 1024;
        const totalChunks = Math.ceil(blob.size / CHUNK_SIZE);
        
        console.log(`📦 Splitting into ${totalChunks} chunks`);

        const receivedChunks = new Set();
        
        const chunkConfirmHandler = (data) => {
            receivedChunks.add(data.chunk);
            console.log(`✅ Server confirmed chunk ${data.chunk}/${data.total_chunks - 1}`);
        };
        socket.on("chunk_received", chunkConfirmHandler);

        for (let i = 0; i < totalChunks; i++) {
            const start = i * CHUNK_SIZE;
            const end = Math.min(start + CHUNK_SIZE, blob.size);
            const chunk = blob.slice(start, end);
            const chunkData = await chunk.arrayBuffer();
            
            console.log(`📤 Sending chunk ${i}/${totalChunks - 1}, size: ${chunkData.byteLength} bytes`);
            
            socket.emit("upload_chunk", {
                chunk: i,
                data: chunkData,
                total_chunks: totalChunks
            });

            const progress = Math.round(((i + 1) / totalChunks) * 100);
            bufferText.textContent = `Upload: ${progress}%`;
            bufferProgress.style.width = `${progress}%`;
            
            await new Promise(resolve => setTimeout(resolve, 150));
        }

        console.log("✅ All chunks sent!");
        loadingText.textContent = "Verifying upload...";

        const maxWaitTime = 10000;
        const startTime = Date.now();
        
        while (receivedChunks.size < totalChunks && (Date.now() - startTime) < maxWaitTime) {
            await new Promise(resolve => setTimeout(resolve, 200));
            bufferText.textContent = `Verified: ${receivedChunks.size}/${totalChunks} chunks`;
        }
        
        socket.off("chunk_received", chunkConfirmHandler);

        console.log(`🎉 Starting multimodal analysis...`);
        loadingText.textContent = "Analyzing with AI models...";
        bufferText.textContent = "Processing...";

    } catch (error) {
        console.error("❌ Error sending video:", error);
        loadingOverlay.style.display = "none";
        alert("Error uploading video: " + error.message);
        analyzeBtn.disabled = false;
        bufferText.textContent = "Upload failed";
        bufferProgress.style.width = "0%";
    } finally {
        uploadInProgress = false;
    }
}

// ================================
// SOCKET EVENTS
// ================================
socket.on("connect", () => {
    console.log("✅ Connected to server");
});

socket.on("disconnect", () => {
    console.log("⚠️ Disconnected from server");
});

socket.on("connect_error", (error) => {
    console.error("❌ Connection error:", error);
});

socket.on("processing_update", data => {
    loadingText.textContent = data.status;
    console.log(data.status);
});

// ================================
// ANALYSIS RESULT - PROFESSIONAL DISPLAY
// ================================
socket.on("analysis_complete", data => {
    console.log("Analysis complete:", data);

    if (data.status !== "success") {
        alert("Analysis failed: " + data.message);
        loadingOverlay.style.display = "none";
        analyzeBtn.disabled = false;
        return;
    }

    // Store current analysis for tooltip
    currentAnalysis = data;

    // Get final probability
    const finalProbability = data.final_probability;
    const confidencePercent = Math.round(finalProbability * 100);

    analysisCount++;
    riskHistory.push(confidencePercent);

    // Update main display
    riskPercentage.textContent = `${confidencePercent}%`;
    riskLevel.textContent = getRiskLevel(confidencePercent);

    analysisCountEl.textContent = analysisCount;
    avgRiskEl.textContent =
        Math.round(riskHistory.reduce((a, b) => a + b, 0) / riskHistory.length) + "%";
    lastUpdateEl.textContent = new Date().toLocaleTimeString();

    // Display explanation
    explanationText.textContent = data.explanation;

    // Display modality breakdown with professional styling
    displayModalityBreakdown(data);

    // Display feature importance
    displayFeatureImportance(data);

    // Update chart with hover info
    updateChart(confidencePercent, data);

    // Cleanup
    loadingOverlay.style.display = "none";
    analyzeBtn.disabled = false;
    recordedChunks = [];
    bufferText.textContent = "Ready for next analysis";
    bufferProgress.style.width = "0%";
});

// ================================
// MODALITY BREAKDOWN - PROFESSIONAL
// ================================
function displayModalityBreakdown(data) {
    const container = document.getElementById("modality-breakdown") || createModalityContainer();
    
    const facialPct = data.facial_probability ? Math.round(data.facial_probability * 100) : null;
    const voicePct = data.voice_probability ? Math.round(data.voice_probability * 100) : null;
    const finalPct = Math.round(data.final_probability * 100);
    
    let html = `
        <div class="modality-section">
            <h3 class="section-title">📊 Multimodal Analysis Breakdown</h3>
            <div class="modality-cards">
    `;
    
    // Facial card
    if (facialPct !== null) {
        const facialLevel = getRiskLevel(facialPct);
        const facialColor = getRiskColor(facialPct);
        html += `
            <div class="modality-card facial-card" data-tooltip="Facial Expression Analysis">
                <div class="card-header">
                    <span class="card-icon">👁️</span>
                    <span class="card-title">Facial</span>
                </div>
                <div class="probability-display" style="color: ${facialColor};">
                    ${facialPct}%
                </div>
                <div class="risk-badge" style="background: ${facialColor}20; color: ${facialColor};">
                    ${facialLevel}
                </div>
                <div class="card-weight">Weight: ${Math.round(data.facial_weight * 100)}%</div>
            </div>
        `;
    }
    
    // Voice card
    if (voicePct !== null) {
        const voiceLevel = getRiskLevel(voicePct);
        const voiceColor = getRiskColor(voicePct);
        html += `
            <div class="modality-card voice-card" data-tooltip="Voice Characteristics Analysis">
                <div class="card-header">
                    <span class="card-icon">🎤</span>
                    <span class="card-title">Voice</span>
                </div>
                <div class="probability-display" style="color: ${voiceColor};">
                    ${voicePct}%
                </div>
                <div class="risk-badge" style="background: ${voiceColor}20; color: ${voiceColor};">
                    ${voiceLevel}
                </div>
                <div class="card-weight">Weight: ${Math.round(data.voice_weight * 100)}%</div>
            </div>
        `;
    }
    
    html += `</div>`;
    
    // Fusion visualization
    if (facialPct !== null && voicePct !== null) {
        const facialW = Math.round(data.facial_weight * 100);
        const voiceW = Math.round(data.voice_weight * 100);
        html += `
            <div class="fusion-section">
                <h4 class="fusion-title">🔗 Fusion Weights</h4>
                <div class="fusion-bar-container">
                    <div class="fusion-component facial-component" style="width: ${facialW}%;">
                        <span class="fusion-label">${facialW}%</span>
                    </div>
                    <div class="fusion-component voice-component" style="width: ${voiceW}%;">
                        <span class="fusion-label">${voiceW}%</span>
                    </div>
                </div>
                <div class="fusion-legend">
                    <div class="legend-item"><span class="legend-dot facial"></span>Facial</div>
                    <div class="legend-item"><span class="legend-dot voice"></span>Voice</div>
                </div>
            </div>
        `;
    }
    
    html += `</div>`;
    container.innerHTML = html;
}

// ================================
// FEATURE IMPORTANCE
// ================================
function displayFeatureImportance(data) {
    const container = document.getElementById("feature-importance") || createFeatureContainer();
    
    let html = `
        <div class="feature-section">
            <h3 class="section-title">🔍 Key Indicators</h3>
            <div class="feature-columns">
    `;
    
    // Facial features
    if (data.feature_stats && data.feature_stats.top_action_units) {
        html += `
            <div class="feature-column facial-features">
                <div class="feature-header">
                    <span class="feature-icon">👁️</span>
                    <span>Facial Expressions</span>
                </div>
                <div class="feature-list">
        `;
        
        const topAUs = data.feature_stats.top_action_units.slice(0, 4);
        if (topAUs.length > 0) {
            topAUs.forEach((au, idx) => {
                const intensity = (au.intensity * 100).toFixed(0);
                const barWidth = Math.min(intensity, 100);
                html += `
                    <div class="feature-item">
                        <div class="feature-name">${au.name.split('-')[0].trim()}</div>
                        <div class="feature-bar">
                            <div class="feature-fill" style="width: ${barWidth}%"></div>
                        </div>
                        <div class="feature-value">${intensity}%</div>
                    </div>
                `;
            });
        }
        
        html += `
                </div>
            </div>
        `;
    }
    
    // Voice features
    if (data.modality_insights && data.modality_insights.voice && data.modality_insights.voice.stats) {
        const voiceStats = data.modality_insights.voice.stats;
        html += `
            <div class="feature-column voice-features">
                <div class="feature-header">
                    <span class="feature-icon">🎤</span>
                    <span>Voice Characteristics</span>
                </div>
                <div class="feature-list">
        `;
        
        const features = [
            { name: 'Pitch Variation', value: voiceStats.pitch_std, max: 50 },
            { name: 'Energy Level', value: voiceStats.mfcc_mean, max: 1 },
            { name: 'Spectral Range', value: voiceStats.spectral_centroid, max: 4000 }
        ];
        
        features.forEach(feat => {
            const percent = Math.min((feat.value / feat.max) * 100, 100);
            html += `
                <div class="feature-item">
                    <div class="feature-name">${feat.name}</div>
                    <div class="feature-bar">
                        <div class="feature-fill" style="width: ${percent}%"></div>
                    </div>
                    <div class="feature-value">${feat.value.toFixed(1)}</div>
                </div>
            `;
        });
        
        html += `
                </div>
            </div>
        `;
    }
    
    html += `
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

// ================================
// CREATE CONTAINERS IF NEEDED
// ================================
function createModalityContainer() {
    const resultCard = document.querySelector(".results-section");
    if (resultCard) {
        const container = document.createElement("div");
        container.id = "modality-breakdown";
        container.className = "card";
        resultCard.insertBefore(container, resultCard.firstChild);
        return container;
    }
    return null;
}

function createFeatureContainer() {
    const resultCard = document.querySelector(".card.au-card");
    if (resultCard) {
        const container = document.createElement("div");
        container.id = "feature-importance";
        container.className = "card";
        resultCard.parentNode.insertBefore(container, resultCard);
        return container;
    }
    return null;
}

// ================================
// PROFESSIONAL CHART WITH HOVER INFO
// ================================
const historyCtx = document.getElementById("history-chart").getContext("2d");

const historyChart = new Chart(historyCtx, {
    type: "line",
    data: {
        labels: [],
        datasets: [{
            label: "Depression Risk Score (%)",
            data: [],
            borderColor: "#8b5cf6",
            backgroundColor: "rgba(139, 92, 246, 0.1)",
            borderWidth: 3,
            pointRadius: 6,
            pointBackgroundColor: "#8b5cf6",
            pointBorderColor: "#fff",
            pointBorderWidth: 2,
            pointHoverRadius: 8,
            tension: 0.4,
            fill: true
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: true,
        interaction: {
            mode: 'index',
            intersect: false
        },
        plugins: {
            legend: {
                display: true,
                labels: {
                    color: "#e5e7eb",
                    font: { size: 13, weight: 'bold' }
                }
            },
            tooltip: {
                backgroundColor: "rgba(15, 23, 42, 0.95)",
                titleColor: "#8b5cf6",
                bodyColor: "#e5e7eb",
                borderColor: "#8b5cf6",
                borderWidth: 2,
                padding: 12,
                titleFont: { size: 14, weight: 'bold' },
                bodyFont: { size: 12 },
                displayColors: false,
                callbacks: {
                    title: (context) => `Analysis ${context[0].dataIndex + 1}`,
                    afterLabel: (context) => {
                        const idx = context.dataIndex;
                        if (currentAnalysis) {
                            const details = [
                                `Facial: ${(currentAnalysis.facial_probability * 100).toFixed(1)}%`,
                                `Voice: ${(currentAnalysis.voice_probability * 100).toFixed(1)}%`,
                                `Final: ${(currentAnalysis.final_probability * 100).toFixed(1)}%`
                            ];
                            return details.join('\n');
                        }
                        return '';
                    }
                }
            }
        },
        scales: {
            y: {
                min: 0,
                max: 100,
                grid: {
                    color: "rgba(148, 163, 184, 0.1)",
                    drawBorder: false
                },
                ticks: {
                    color: "#94a3b8",
                    callback: (value) => value + "%"
                },
                title: {
                    display: true,
                    text: "Risk Score (%)",
                    color: "#e5e7eb"
                }
            },
            x: {
                grid: {
                    display: false
                },
                ticks: {
                    color: "#94a3b8"
                }
            }
        }
    }
});

function updateChart(value, data) {
    historyChart.data.labels.push(`Analysis ${analysisCount}`);
    historyChart.data.datasets[0].data.push(value);
    historyChart.update();
}

// ================================
// UTILITY FUNCTIONS
// ================================
function getRiskLevel(percentage) {
    if (percentage >= 70) return "🔴 High Risk";
    if (percentage >= 50) return "🟡 Moderate Risk";
    if (percentage >= 30) return "🟠 Low Risk";
    return "🟢 Minimal Risk";
}

function getRiskColor(percentage) {
    if (percentage >= 70) return "#ef4444";
    if (percentage >= 50) return "#f59e0b";
    if (percentage >= 30) return "#f97316";
    return "#22c55e";
}

downloadBtn.addEventListener("click", () => {
    alert("Download feature coming soon");
});