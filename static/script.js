document.addEventListener('DOMContentLoaded', () => {
    // --- Chart.js Setup ---
    const ctx = document.getElementById('anomalyChart').getContext('2d');

    // Gradient styling for the chart
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(6, 182, 212, 0.5)');
    gradient.addColorStop(1, 'rgba(6, 182, 212, 0.0)');

    const maxDataPoints = 40; // How many dots to show on the graph

    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [], // Time / Tx IDs
            datasets: [{
                label: 'Anomaly Score (Lower = Worse)',
                data: [],
                borderColor: '#06b6d4',
                backgroundColor: gradient,
                borderWidth: 2,
                pointBackgroundColor: '#0f172a',
                pointBorderColor: '#06b6d4',
                pointBorderWidth: 2,
                pointRadius: 3,
                fill: true,
                tension: 0.4 // Smooth curves
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    display: false // Hide X axis to look cleaner
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#94a3b8'
                    },
                    // We know scores drift around 0; anomalies fall into negative numbers (e.g., -0.2)
                    min: -0.3,
                    max: 0.3
                }
            },
            animation: {
                duration: 400, // Smooth slide
                easing: 'linear'
            }
        }
    });

    // --- DOM Elements ---
    const valProcessed = document.getElementById('val-processed');
    const valAnomalies = document.getElementById('val-anomalies');
    const valHitrate = document.getElementById('val-hitrate');
    const valLatency = document.getElementById('val-latency');
    const feedList = document.getElementById('transaction-feed');
    const txBadge = document.getElementById('tx-badge');

    // Stats trackers
    let totalLatency = 0;

    // --- Server-Sent Events (SSE) Listener ---
    // Connects to our FastAPI /stream endpoint
    const evtSource = new EventSource('/stream');

    evtSource.onopen = function () {
        txBadge.textContent = "Live TGN Feed";
        txBadge.style.backgroundColor = "rgba(16, 185, 129, 0.2)";
        txBadge.style.color = "#10b981";
    };

    evtSource.onmessage = function (event) {
        const tx = JSON.parse(event.data);

        // 1. Update Metrics
        valProcessed.textContent = tx.total_processed;
        valAnomalies.textContent = tx.total_anomalies;

        const hitrate = (tx.total_anomalies / tx.total_processed) * 100;
        valHitrate.textContent = hitrate.toFixed(2) + "%";

        totalLatency += tx.latency_ms;
        const avgLatency = totalLatency / tx.total_processed;
        valLatency.textContent = avgLatency.toFixed(2) + " ms";

        // 2. Update Feed
        const item = document.createElement('div');
        item.className = `tx-item ${tx.is_anomaly ? 'anomaly' : ''}`;

        const statusText = tx.is_anomaly ? '⚠️ ZERO-DAY' : '✓ OK';

        item.innerHTML = `
            <div class="tx-header">
                <span class="tx-path">#${tx.src} ➔ #${tx.dst}</span>
                <span>${statusText}</span>
            </div>
            <div class="tx-amount">$${tx.amount.toFixed(2)}</div>
            <div class="tx-desc">${tx.desc} (Score: ${tx.score.toFixed(3)})</div>
        `;

        feedList.insertBefore(item, feedList.firstChild);

        // Keep feed from growing infinitely
        if (feedList.children.length > 50) {
            feedList.removeChild(feedList.lastChild);
        }

        // 3. Update Chart
        chart.data.labels.push(tx.id);
        chart.data.datasets[0].data.push(tx.score);

        // If it's an anomaly, make the chart dot red
        const dotColors = chart.data.datasets[0].data.map(
            score => score < 0 ? '#ef4444' : '#06b6d4'
        );
        chart.data.datasets[0].pointBorderColor = dotColors;
        chart.data.datasets[0].pointBackgroundColor = dotColors;

        if (chart.data.labels.length > maxDataPoints) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }

        chart.update();
    };

    evtSource.onerror = function () {
        txBadge.textContent = "Disconnected - Retrying...";
        txBadge.style.backgroundColor = "rgba(239, 68, 68, 0.2)";
        txBadge.style.color = "#ef4444";
    };
});
