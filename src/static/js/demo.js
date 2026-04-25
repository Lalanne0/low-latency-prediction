/**
 * demo.js — Demo mode controller.
 * Handles the crash button and demo reset.
 */

const DemoController = (() => {
    const crashBtn = () => document.getElementById('crash-btn');
    const resetBtn = () => document.getElementById('reset-demo-btn');

    function init() {
        crashBtn()?.addEventListener('click', triggerCrash);
        resetBtn()?.addEventListener('click', resetDemo);
    }

    async function triggerCrash() {
        const btn = crashBtn();
        if (!btn) return;

        btn.disabled = true;
        btn.textContent = '💥 Crashing…';

        try {
            await fetch('/api/demo/crash', { method: 'POST' });
        } catch (e) {
            console.error('Crash trigger failed:', e);
        }

        // Re-enable after 3 seconds (crash animation time)
        setTimeout(() => {
            btn.disabled = false;
            btn.textContent = '💥 Simulate Stock Crash';
        }, 3000);
    }

    async function resetDemo() {
        try {
            await fetch('/api/demo/reset', { method: 'POST' });
            ChartManager.clear('demo');
            // Reset signal display
            const signal = document.getElementById('signal-demo');
            if (signal) {
                signal.textContent = 'KEEP';
                signal.className = 'signal keep';
            }
        } catch (e) {
            console.error('Demo reset failed:', e);
        }
    }

    return { init };
})();
