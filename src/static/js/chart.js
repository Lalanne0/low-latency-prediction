/**
 * chart.js — TradingView Lightweight Charts integration
 * Manages chart instances for live and demo modes.
 */

const ChartManager = (() => {
    const charts = {};
    const series = {};
    const MAX_POINTS = 300;

    /**
     * Initialise a Lightweight Chart inside the given container.
     */
    function create(containerId, mode) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const chart = LightweightCharts.createChart(container, {
            layout: {
                background: { type: 'solid', color: 'transparent' },
                textColor: '#94a3b8',
                fontFamily: "'Inter', sans-serif",
                fontSize: 12,
            },
            grid: {
                vertLines: { color: 'rgba(55, 65, 81, 0.25)' },
                horzLines: { color: 'rgba(55, 65, 81, 0.25)' },
            },
            crosshair: {
                mode: LightweightCharts.CrosshairMode.Normal,
                vertLine: { color: 'rgba(99, 102, 241, 0.4)', width: 1, style: 2 },
                horzLine: { color: 'rgba(99, 102, 241, 0.4)', width: 1, style: 2 },
            },
            rightPriceScale: {
                borderColor: 'rgba(55, 65, 81, 0.5)',
                scaleMargins: { top: 0.1, bottom: 0.1 },
            },
            timeScale: {
                borderColor: 'rgba(55, 65, 81, 0.5)',
                timeVisible: true,
                secondsVisible: true,
                rightOffset: 5,
            },
            handleScroll: { vertTouchDrag: false },
        });

        // Area series for a sleek look
        const areaSeries = chart.addAreaSeries({
            topColor: mode === 'demo'
                ? 'rgba(245, 158, 11, 0.35)'
                : 'rgba(99, 102, 241, 0.35)',
            bottomColor: mode === 'demo'
                ? 'rgba(245, 158, 11, 0.02)'
                : 'rgba(99, 102, 241, 0.02)',
            lineColor: mode === 'demo' ? '#f59e0b' : '#6366f1',
            lineWidth: 2,
            crosshairMarkerVisible: true,
            crosshairMarkerRadius: 4,
        });

        charts[mode] = chart;
        series[mode] = { area: areaSeries, data: [] };

        // Responsive resize
        const ro = new ResizeObserver(() => {
            chart.applyOptions({
                width: container.clientWidth,
                height: container.clientHeight,
            });
        });
        ro.observe(container);
    }

    /**
     * Add a single tick to the chart.
     */
    function addTick(mode, tick) {
        if (!series[mode]) return;

        const s = series[mode];
        const point = { time: tick.time, value: tick.close };

        // Deduplicate — Lightweight Charts requires strictly increasing time
        if (s.data.length > 0 && s.data[s.data.length - 1].time >= point.time) {
            // Update last point if same timestamp
            s.data[s.data.length - 1] = point;
        } else {
            s.data.push(point);
        }

        // Rolling window
        if (s.data.length > MAX_POINTS) {
            s.data.shift();
        }

        s.area.setData(s.data);
        charts[mode].timeScale().scrollToRealTime();
    }

    /**
     * Update the area series colour based on signal.
     */
    function setSignalColor(mode, signal) {
        if (!series[mode]) return;

        if (signal === 'SELL') {
            series[mode].area.applyOptions({
                topColor: 'rgba(255, 71, 87, 0.35)',
                bottomColor: 'rgba(255, 71, 87, 0.02)',
                lineColor: '#ff4757',
            });
        } else {
            const isDemo = mode === 'demo';
            series[mode].area.applyOptions({
                topColor: isDemo
                    ? 'rgba(245, 158, 11, 0.35)'
                    : 'rgba(99, 102, 241, 0.35)',
                bottomColor: isDemo
                    ? 'rgba(245, 158, 11, 0.02)'
                    : 'rgba(99, 102, 241, 0.02)',
                lineColor: isDemo ? '#f59e0b' : '#6366f1',
            });
        }
    }

    /**
     * Clear all data from a chart.
     */
    function clear(mode) {
        if (!series[mode]) return;
        series[mode].data = [];
        series[mode].area.setData([]);
    }

    return { create, addTick, setSignalColor, clear };
})();
