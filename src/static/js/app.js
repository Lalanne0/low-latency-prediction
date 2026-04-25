/**
 * app.js — Main application logic.
 * Manages WebSocket connections, tab switching, inference mode selection, and UI updates.
 */

(() => {
    'use strict';

    // ── State ────────────────────────────────
    let liveWs = null;
    let demoWs = null;
    let liveTickCount = 0;
    let demoTickCount = 0;
    let activeTab = 'live';
    let initialPrice = { live: null, demo: null };
    let lastAlertTime = 0;
    const ALERT_COOLDOWN_MS = 30000; // 30s cooldown between alerts

    // ── DOM refs ─────────────────────────────
    const $ = (id) => document.getElementById(id);

    // ── Initialisation ───────────────────────
    document.addEventListener('DOMContentLoaded', () => {
        // Init charts
        ChartManager.create('chart-live', 'live');
        ChartManager.create('chart-demo', 'demo');

        // Init demo controller
        DemoController.init();

        // Tab switching
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => switchTab(tab.dataset.tab));
        });

        // Disclaimer dismiss
        $('dismiss-disclaimer')?.addEventListener('click', () => {
            $('disclaimer-banner')?.classList.add('hidden');
        });

        // Alert dismiss
        $('dismiss-alert')?.addEventListener('click', () => {
            $('sell-alert')?.classList.add('hidden');
        });

        // Ticker change
        $('ticker-change-btn')?.addEventListener('click', changeTicker);
        $('ticker-input')?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') changeTicker();
        });

        // Inference mode buttons
        initModeSelector();

        // Connect WebSockets
        connectLive();
        connectDemo();

        // Load config
        loadConfig();
    });

    // ── Tab Switching ────────────────────────
    function switchTab(tab) {
        activeTab = tab;
        document.querySelectorAll('.tab').forEach(t => {
            t.classList.toggle('active', t.dataset.tab === tab);
            t.setAttribute('aria-selected', t.dataset.tab === tab);
        });
        document.querySelectorAll('.panel').forEach(p => {
            p.classList.toggle('active', p.id === `panel-${tab}`);
        });

        // Show/hide ticker selector (only in live mode)
        const ts = $('ticker-selector-live');
        if (ts) ts.style.display = tab === 'live' ? 'flex' : 'none';
    }

    // ── Inference Mode Selector ──────────────
    function initModeSelector() {
        // Bind both live and demo mode selectors
        ['mode-selector', 'mode-selector-demo'].forEach(id => {
            const selector = $(id);
            if (!selector) return;
            selector.addEventListener('click', (e) => {
                const btn = e.target.closest('.mode-btn');
                if (!btn || btn.classList.contains('unavailable')) return;
                changeMode(btn.dataset.mode);
            });
        });
    }

    async function changeMode(mode) {
        try {
            const resp = await fetch('/api/mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode }),
            });
            const data = await resp.json();
            if (data.status === 'ok') {
                updateModeUI(data.mode, data.modes);
            }
        } catch (e) {
            console.error('Mode change failed:', e);
        }
    }

    function updateModeUI(activeMode, modes) {
        // Update both selectors (live and demo)
        ['mode-selector', 'mode-selector-demo'].forEach(selectorId => {
            const selector = $(selectorId);
            if (!selector) return;
            selector.querySelectorAll('.mode-btn').forEach(btn => {
                const m = btn.dataset.mode;
                btn.classList.toggle('active', m === activeMode);
                if (modes) {
                    const modeInfo = modes.find(x => x.value === m);
                    if (modeInfo && !modeInfo.available) {
                        btn.classList.add('unavailable');
                    } else {
                        btn.classList.remove('unavailable');
                    }
                }
            });
        });

        // Update both labels
        const modeLabels = {
            'sklearn': 'scikit-learn (baseline)',
            'numpy': 'Raw NumPy',
            'numpy_fused': 'NumPy fused (float32)',
            'numba': 'Numba JIT (float32)',
        };
        const labelText = `Active: ${modeLabels[activeMode] || activeMode}`;
        ['mode-active-label', 'mode-active-label-demo'].forEach(id => {
            const el = $(id);
            if (el) el.textContent = labelText;
        });
    }

    // ── WebSocket — Live ─────────────────────
    function connectLive() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        liveWs = new WebSocket(`${protocol}//${location.host}/ws/live`);

        liveWs.onopen = () => setStatus('live', true);
        liveWs.onclose = () => {
            setStatus('live', false);
            setTimeout(connectLive, 3000); // auto-reconnect
        };
        liveWs.onerror = () => liveWs.close();
        liveWs.onmessage = (e) => handleMessage('live', JSON.parse(e.data));
    }

    // ── WebSocket — Demo ─────────────────────
    function connectDemo() {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
        demoWs = new WebSocket(`${protocol}//${location.host}/ws/demo`);

        demoWs.onopen = () => setStatus('demo', true);
        demoWs.onclose = () => {
            setStatus('demo', false);
            setTimeout(connectDemo, 3000);
        };
        demoWs.onerror = () => demoWs.close();
        demoWs.onmessage = (e) => handleMessage('demo', JSON.parse(e.data));
    }

    // ── Message Handler ──────────────────────
    function handleMessage(mode, msg) {
        if (msg.type !== 'update') return;

        const { tick, prediction, alert } = msg;

        // Update chart
        ChartManager.addTick(mode, tick);

        // Track initial price for % change
        if (initialPrice[mode] === null) {
            initialPrice[mode] = tick.close;
        }

        // Update price display
        updatePrice(mode, tick.close);

        // Update prediction display
        updatePrediction(mode, prediction);

        // Update tick count
        if (mode === 'live') {
            liveTickCount++;
            const tc = $('tick-count-live');
            if (tc) tc.textContent = `Ticks: ${liveTickCount.toLocaleString()}`;
        } else {
            demoTickCount++;
            const tc = $('tick-count-demo');
            if (tc) tc.textContent = `Ticks: ${demoTickCount.toLocaleString()}`;
        }

        // Handle signal change alert — only for active tab, with cooldown
        if (alert && mode === activeTab) {
            const now = Date.now();
            if (now - lastAlertTime > ALERT_COOLDOWN_MS) {
                lastAlertTime = now;
                showSellAlert(prediction.confidence);
            }
        }
    }

    // ── UI Updates ───────────────────────────
    function updatePrice(mode, price) {
        const suffix = mode === 'live' ? '' : '-demo';
        const priceEl = $(`chart-price${suffix}`);
        const changeEl = $(`chart-change${suffix}`);

        if (priceEl) priceEl.textContent = `$${price.toFixed(2)}`;

        if (changeEl && initialPrice[mode]) {
            const pctChange = ((price - initialPrice[mode]) / initialPrice[mode]) * 100;
            changeEl.textContent = `${pctChange >= 0 ? '+' : ''}${pctChange.toFixed(2)}%`;
            changeEl.className = `change ${pctChange > 0 ? 'positive' : pctChange < 0 ? 'negative' : 'neutral'}`;
        }
    }

    function updatePrediction(mode, pred) {
        const suffix = mode === 'live' ? '-live' : '-demo';

        // Signal
        const signalEl = $(`signal${suffix}`);
        if (signalEl) {
            signalEl.textContent = pred.signal;
            signalEl.className = `signal ${pred.signal.toLowerCase()}`;
        }

        // Chart color
        ChartManager.setSignalColor(mode, pred.signal);

        // Confidence
        const confFill = $(`confidence-fill${suffix}`);
        const confValue = $(`confidence-value${suffix}`);
        const confPct = (pred.confidence * 100).toFixed(1);
        if (confFill) {
            confFill.style.width = `${confPct}%`;
            confFill.className = `confidence-fill ${pred.signal === 'SELL' ? 'sell' : ''}`;
        }
        if (confValue) confValue.textContent = `${confPct}%`;

        // Latency
        const latencyEl = $(`latency${suffix}`);
        const latencyBar = $(`latency-bar-fill${suffix}`);
        if (latencyEl) {
            latencyEl.textContent = pred.inference_us.toFixed(1);
            latencyEl.className = 'latency-value';
            if (pred.inference_us > 1000) latencyEl.classList.add('very-slow');
            else if (pred.inference_us > 500) latencyEl.classList.add('slow');
        }
        if (latencyBar) {
            // Scale: 0–1000μs → 0–100%
            const pct = Math.min(100, (pred.inference_us / 1000) * 100);
            latencyBar.style.width = `${pct}%`;
            if (pred.inference_us > 1000) {
                latencyBar.style.background = 'linear-gradient(90deg, #ff4757, #ff6b81)';
            } else if (pred.inference_us > 500) {
                latencyBar.style.background = 'linear-gradient(90deg, #f59e0b, #fbbf24)';
            } else {
                latencyBar.style.background = 'linear-gradient(90deg, #00d4aa, #00b894)';
            }
        }

        // Update mode UI from prediction data (reflects server-side active mode)
        if (pred.mode) {
            updateModeUI(pred.mode, null);
        }
    }

    function showSellAlert(confidence) {
        const alertEl = $('sell-alert');
        const confEl = $('alert-confidence');
        if (alertEl) {
            alertEl.classList.remove('hidden');
            if (confEl) confEl.textContent = `${(confidence * 100).toFixed(1)}%`;
        }
    }

    function setStatus(mode, connected) {
        const suffix = mode === 'live' ? '-live' : '-demo';
        const dot = $(`ws-status${suffix}`);
        const text = $(`ws-status-text${suffix}`);
        if (dot) dot.className = `status-indicator ${connected ? 'connected' : 'disconnected'}`;
        if (text) text.textContent = connected ? 'Connected' : 'Disconnected';
    }

    // ── Ticker Change ────────────────────────
    async function changeTicker() {
        const input = $('ticker-input');
        const btn = $('ticker-change-btn');
        if (!input || !btn) return;

        const ticker = input.value.trim().toUpperCase();
        if (!ticker) return;

        btn.disabled = true;
        btn.textContent = '…';

        try {
            const resp = await fetch('/api/ticker', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ticker }),
            });
            const data = await resp.json();
            if (data.status === 'ok') {
                const label = $('chart-ticker-label');
                if (label) label.textContent = data.ticker;
                ChartManager.clear('live');
                liveTickCount = 0;
                initialPrice.live = null;
            }
        } catch (e) {
            console.error('Ticker change failed:', e);
        }

        btn.disabled = false;
        btn.textContent = 'Apply';
    }

    // ── Load Config ──────────────────────────
    async function loadConfig() {
        try {
            const resp = await fetch('/api/config');
            const config = await resp.json();
            const input = $('ticker-input');
            const label = $('chart-ticker-label');
            if (input) input.value = config.ticker;
            if (label) label.textContent = config.ticker;

            // Set active inference mode from server config
            if (config.inference_mode && config.inference_modes) {
                updateModeUI(config.inference_mode, config.inference_modes);
            }
        } catch (e) {
            console.log('Config load failed (server may still be starting):', e);
        }
    }
})();
