// frontend/app.js
const API_KEY = "REPLACE_ME"; // Overwritten dynamically by deployment script variables

const N = 28;         // Native model resolution footprint
const SCALE = 10;     // Magnification scalar multiplier for display

const pad = document.getElementById("pad");
const view = pad.getContext("2d");
view.imageSmoothingEnabled = false; // Ensures crisp, pixelated rendering blocks

// Hidden low-res background matrix buffer
const grid = document.createElement("canvas");
grid.width = N;
grid.height = N;
const gctx = grid.getContext("2d");
gctx.lineWidth = 2.5;
gctx.lineCap = "round";
gctx.lineJoin = "round";

let drawing = false;

// OOD Toast notification
const oodToast = document.getElementById("oodToast");
let toastTimerReference = null;

function showOODToast() {
    const oodToastElement = document.getElementById("oodToast");
    
    // Clear out any pre-existing active timer structures to reset the clock
    if (toastTimerReference) {
        clearTimeout(toastTimerReference);
    }
    
    // Activate visibility states
    oodToastElement.classList.add("show");
    
    // Schedule clean removal exactly 3 seconds out
    toastTimerReference = setTimeout(() => {
        oodToastElement.classList.remove("show");
        toastTimerReference = null;
    }, 3000);
}
// Project the low-res grid up onto the high-res viewport display
function render() {
    view.drawImage(grid, 0, 0, pad.width, pad.height);
}

function clearPad() {
    gctx.fillStyle = "#fff";
    gctx.fillRect(0, 0, N, N);
    render();
    document.getElementById("result").textContent = "";
}

// Map screen tracking movements onto internal low-res coordinate maps
pad.onmousedown = e => {
    drawing = true;
    gctx.beginPath();
    gctx.moveTo(e.offsetX / SCALE, e.offsetY / SCALE);
};

pad.onmousemove = e => {
    if (!drawing) return;
    gctx.lineTo(e.offsetX / SCALE, e.offsetY / SCALE);
    gctx.stroke();
    render();
};

pad.onmouseup = () => { drawing = false; };
pad.onmouseleave = () => { drawing = false; };

// Package pixels up into a structured 2D matrix array and invert color spaces
function getPixels() {
    const data = gctx.getImageData(0, 0, N, N).data;
    const pixels = [];

    for (let y = 0; y < N; y++) {
        const row = [];
        for (let x = 0; x < N; x++) {
            // Read red channel array value (offset by index footprint blocks)
            const rChannel = data[(y * N + x) * 4];
            // Invert polarity: 255 (white background) maps to 0 (empty ink track space)
            row.push(255 - rChannel);
        }
        pixels.push(row);
    }
    return pixels;
}

// Issue an asynchronous network fetch transaction payload sequence
async function classify() {
    const out = document.getElementById("result");
    out.textContent = "Analyzing matrix pattern...";

    try {
        const r = await fetch("/api/classify", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-API-Key": API_KEY
            },
            body: JSON.stringify({ pixels: getPixels() })
        });

        if (!r.ok) {
            out.textContent = "Error communicating with Gateway: " + r.status;
            return;
        }

        const d = await r.json();
        out.textContent = `Prediction: ${d.prediction} (${(d.confidence * 100).toFixed(1)}%)`;

        if (d.is_ood) {
            showOODToast();
        }

        refreshHistory();
    } catch (err) {
        out.textContent = "Network error occurred.";
    }
}

// Populate the bottom recent-predictions list view with live database history metrics
async function refreshHistory() {
    try {
        const r = await fetch("/api/results");
        if (!r.ok) return;

        const ul = document.getElementById("history");
        ul.innerHTML = "";

        const data = await r.json();
        for (const row of data.results) {
            const li = document.createElement("li");
            const timestamp = new Date(row.created_at).toLocaleTimeString();
            const oodIndicator = row.is_ood ? " 🚨 OOD" : "";
            li.textContent = `[${timestamp}] Digit: ${row.prediction} — Conf: ${(row.confidence * 100).toFixed(1)}% (Model: ${row.model_version})${oodIndicator}`;
            ul.appendChild(li);
        }
    } catch (err) {
        console.error("Failed to sync structural log listings:", err);
    }
}

// Setup core application button triggers
document.getElementById("classify").onclick = classify;
document.getElementById("clear").onclick = clearPad;

// Run initial system initialization parameters
clearPad();
refreshHistory();