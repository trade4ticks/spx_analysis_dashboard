/**
 * ai_explorer.js — Alpine component + chart logic for AI Surface Explorer.
 *
 * Chart auto-detection rules (result shape → chart type):
 *   date/time col + numeric col(s)          → line
 *   single category col + numeric col(s)    → bar
 *   exactly two numeric cols, nothing else  → scatter
 *   anything else                           → table only (chartType = null)
 */

'use strict';

let _explorerChart = null;

const _CHART_COLORS = [
    '#3498db', '#2ecc71', '#e74c3c', '#f39c12',
    '#9b59b6', '#1abc9c', '#e67e22', '#ecf0f1',
];

// ── Column type classifiers ───────────────────────────────────────────────────

function _isDatelike(val) {
    return typeof val === 'string' && /^\d{4}-\d{2}-\d{2}/.test(val);
}
function _isTimelike(val) {
    return typeof val === 'string' && /^\d{2}:\d{2}(:\d{2})?$/.test(val);
}
function _isNumeric(val) {
    return typeof val === 'number';
}
function _isCategory(val) {
    return typeof val === 'string' && !_isDatelike(val) && !_isTimelike(val);
}

// ── Chart type detection ──────────────────────────────────────────────────────

function detectChartType(columns, rows) {
    if (!rows.length || columns.length < 2) return null;

    const s = rows[0];   // sample row

    const dateCols    = columns.filter(c => _isDatelike(s[c]));
    const timeCols    = columns.filter(c => _isTimelike(s[c]));
    const numericCols = columns.filter(c => _isNumeric(s[c]));
    const catCols     = columns.filter(c => _isCategory(s[c]));

    const hasDateTime = dateCols.length > 0 || timeCols.length > 0;

    if (hasDateTime && numericCols.length > 0) return 'line';
    if (catCols.length === 1 && numericCols.length >= 1) return 'bar';
    if (numericCols.length === 2 && !hasDateTime && catCols.length === 0) return 'scatter';
    return null;
}

// ── Chart rendering ───────────────────────────────────────────────────────────

function renderExplorerChart(chartType, columns, rows) {
    const canvas = document.getElementById('explorer-chart');
    if (!canvas) return;

    if (_explorerChart) { _explorerChart.destroy(); _explorerChart = null; }
    if (!chartType || !rows.length) return;

    const s = rows[0];

    const dateCols    = columns.filter(c => _isDatelike(s[c]));
    const timeCols    = columns.filter(c => _isTimelike(s[c]));
    const numericCols = columns.filter(c => _isNumeric(s[c]));
    const catCols     = columns.filter(c => _isCategory(s[c]));

    const baseScales = {
        x: {
            ticks: { color: '#888', font: { size: 9 }, maxTicksLimit: 12, maxRotation: 45 },
            grid:  { color: 'rgba(255,255,255,0.05)' },
            border: { color: 'transparent' },
        },
        y: {
            ticks: { color: '#888', font: { size: 9 } },
            grid:  { color: 'rgba(255,255,255,0.05)' },
            border: { color: 'transparent' },
        },
    };

    const baseOpts = {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: {
            legend: {
                display: numericCols.length > 1,
                labels: { color: '#aaa', font: { size: 10 }, boxWidth: 14, padding: 10 },
            },
            tooltip: {
                backgroundColor: 'rgba(20,20,20,0.92)',
                titleColor: '#999', bodyColor: '#ddd',
                borderColor: '#444', borderWidth: 1,
            },
        },
    };

    // ── Line ──────────────────────────────────────────────────────────────────
    if (chartType === 'line') {
        let labels;
        if (dateCols.length > 0 && timeCols.length > 0) {
            // Combine date + time into a single label
            labels = rows.map(r => `${r[dateCols[0]]} ${String(r[timeCols[0]]).substring(0, 5)}`);
        } else if (dateCols.length > 0) {
            labels = rows.map(r => r[dateCols[0]]);
        } else {
            labels = rows.map(r => String(r[timeCols[0]]).substring(0, 5));
        }

        const usePoints = rows.length <= 120;

        const datasets = numericCols.map((col, i) => ({
            label: col,
            data: rows.map(r => r[col]),
            borderColor: _CHART_COLORS[i % _CHART_COLORS.length],
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            pointRadius: usePoints ? 2 : 0,
            pointHoverRadius: 4,
            tension: 0,
        }));

        _explorerChart = new Chart(canvas, {
            type: 'line',
            data: { labels, datasets },
            options: { ...baseOpts, scales: baseScales },
        });

    // ── Bar ───────────────────────────────────────────────────────────────────
    } else if (chartType === 'bar') {
        const xCol   = catCols[0];
        const labels = rows.map(r => String(r[xCol]));

        const datasets = numericCols.map((col, i) => ({
            label: col,
            data: rows.map(r => r[col]),
            backgroundColor: _CHART_COLORS[i % _CHART_COLORS.length] + 'cc',
            borderColor: _CHART_COLORS[i % _CHART_COLORS.length],
            borderWidth: 1,
        }));

        _explorerChart = new Chart(canvas, {
            type: 'bar',
            data: { labels, datasets },
            options: { ...baseOpts, scales: baseScales },
        });

    // ── Scatter ───────────────────────────────────────────────────────────────
    } else if (chartType === 'scatter') {
        const xCol = numericCols[0];
        const yCol = numericCols[1];

        _explorerChart = new Chart(canvas, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: `${xCol} vs ${yCol}`,
                    data: rows.map(r => ({ x: r[xCol], y: r[yCol] })),
                    backgroundColor: '#3498dbaa',
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    borderWidth: 0,
                }],
            },
            options: {
                ...baseOpts,
                scales: {
                    x: {
                        ...baseScales.x,
                        title: { display: true, text: xCol, color: '#aaa', font: { size: 10 } },
                    },
                    y: {
                        ...baseScales.y,
                        title: { display: true, text: yCol, color: '#aaa', font: { size: 10 } },
                    },
                },
            },
        });
    }
}

// ── Alpine component ──────────────────────────────────────────────────────────

document.addEventListener('alpine:init', () => {
    Alpine.data('aiExplorer', () => ({

        question:  '',
        loading:   false,
        error:     null,
        sql:       null,
        columns:   [],
        rows:      [],
        summary:   null,
        chartType: null,
        sqlOpen:   false,

        // ── Format a table cell value ───────────────────────────────────────
        fmtCell(val) {
            if (val === null || val === undefined) return '—';
            if (typeof val === 'number') {
                // Integer check: no fractional part
                return Number.isInteger(val) ? val.toString() : val.toFixed(4);
            }
            return String(val);
        },

        // ── Submit a question ────────────────────────────────────────────────
        async submit() {
            const q = this.question.trim();
            if (!q || this.loading) return;

            this.loading   = true;
            this.error     = null;
            this.sql       = null;
            this.columns   = [];
            this.rows      = [];
            this.summary   = null;
            this.chartType = null;
            this.sqlOpen   = false;

            // Destroy any existing chart now so the canvas is clean
            if (_explorerChart) { _explorerChart.destroy(); _explorerChart = null; }

            try {
                const res  = await fetch('/api/ai-explorer/query', {
                    method:  'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body:    JSON.stringify({ question: q }),
                });
                const data = await res.json();

                if (!res.ok) {
                    this.error = data.detail ?? `HTTP ${res.status}`;
                    return;
                }

                this.sql     = data.sql     ?? null;
                this.columns = data.columns ?? [];
                this.rows    = data.rows    ?? [];
                this.summary = data.summary ?? null;
                this.error   = data.error   ?? null;

                if (!this.error && this.rows.length) {
                    this.chartType = detectChartType(this.columns, this.rows);
                    if (this.chartType) {
                        // Canvas becomes visible after Alpine reacts
                        this.$nextTick(() => {
                            renderExplorerChart(this.chartType, this.columns, this.rows);
                        });
                    }
                }
            } catch (e) {
                this.error = e.message ?? 'Request failed';
            } finally {
                this.loading = false;
            }
        },

        // Ctrl/Cmd + Enter submits; plain Enter inserts a newline
        onKeydown(e) {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                this.submit();
            }
        },

        // Expose to template
        get hasResult() { return this.sql !== null; },
        get rowCount()  { return this.rows.length; },
        get chartLabel() {
            if (!this.chartType) return '';
            return this.chartType.charAt(0).toUpperCase() + this.chartType.slice(1) + ' Chart';
        },

    }));
});
