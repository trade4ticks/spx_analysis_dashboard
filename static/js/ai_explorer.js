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

// One Chart.js instance per conversation turn, keyed by turn index
const _explorerCharts = {};

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

    const s = rows[0];
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

// ── Per-turn chart rendering ──────────────────────────────────────────────────

function renderTurnChart(idx, chartType, columns, rows) {
    const canvasId = `explorer-chart-${idx}`;
    const canvas   = document.getElementById(canvasId);
    if (!canvas) return;

    if (_explorerCharts[idx]) { _explorerCharts[idx].destroy(); delete _explorerCharts[idx]; }
    if (!chartType || !rows.length) return;

    const s = rows[0];
    const dateCols    = columns.filter(c => _isDatelike(s[c]));
    const timeCols    = columns.filter(c => _isTimelike(s[c]));
    const numericCols = columns.filter(c => _isNumeric(s[c]));
    const catCols     = columns.filter(c => _isCategory(s[c]));

    const baseScales = {
        x: {
            ticks:  { color: '#888', font: { size: 9 }, maxTicksLimit: 12, maxRotation: 45 },
            grid:   { color: 'rgba(255,255,255,0.05)' },
            border: { color: 'transparent' },
        },
        y: {
            ticks:  { color: '#888', font: { size: 9 } },
            grid:   { color: 'rgba(255,255,255,0.05)' },
            border: { color: 'transparent' },
        },
    };
    const baseOpts = {
        responsive:          true,
        maintainAspectRatio: false,
        animation:           false,
        plugins: {
            legend: {
                display: numericCols.length > 1,
                labels:  { color: '#aaa', font: { size: 10 }, boxWidth: 14, padding: 10 },
            },
            tooltip: {
                backgroundColor: 'rgba(20,20,20,0.92)',
                titleColor: '#999', bodyColor: '#ddd',
                borderColor: '#444', borderWidth: 1,
            },
        },
    };

    if (chartType === 'line') {
        let labels;
        if (dateCols.length > 0 && timeCols.length > 0) {
            labels = rows.map(r => `${r[dateCols[0]]} ${String(r[timeCols[0]]).substring(0, 5)}`);
        } else if (dateCols.length > 0) {
            labels = rows.map(r => r[dateCols[0]]);
        } else {
            labels = rows.map(r => String(r[timeCols[0]]).substring(0, 5));
        }
        const datasets = numericCols.map((col, i) => ({
            label:           col,
            data:            rows.map(r => r[col]),
            borderColor:     _CHART_COLORS[i % _CHART_COLORS.length],
            backgroundColor: 'transparent',
            borderWidth:     1.5,
            pointRadius:     rows.length <= 120 ? 2 : 0,
            pointHoverRadius: 4,
            tension:         0,
        }));
        _explorerCharts[idx] = new Chart(canvas, {
            type: 'line',
            data: { labels, datasets },
            options: { ...baseOpts, scales: baseScales },
        });

    } else if (chartType === 'bar') {
        const xCol   = catCols[0];
        const labels = rows.map(r => String(r[xCol]));
        const datasets = numericCols.map((col, i) => ({
            label:           col,
            data:            rows.map(r => r[col]),
            backgroundColor: _CHART_COLORS[i % _CHART_COLORS.length] + 'cc',
            borderColor:     _CHART_COLORS[i % _CHART_COLORS.length],
            borderWidth:     1,
        }));
        _explorerCharts[idx] = new Chart(canvas, {
            type: 'bar',
            data: { labels, datasets },
            options: { ...baseOpts, scales: baseScales },
        });

    } else if (chartType === 'scatter') {
        const xCol = numericCols[0];
        const yCol = numericCols[1];
        _explorerCharts[idx] = new Chart(canvas, {
            type: 'scatter',
            data: {
                datasets: [{
                    label:           `${xCol} vs ${yCol}`,
                    data:            rows.map(r => ({ x: r[xCol], y: r[yCol] })),
                    backgroundColor: '#3498dbaa',
                    pointRadius:     4,
                    pointHoverRadius: 6,
                    borderWidth:     0,
                }],
            },
            options: {
                ...baseOpts,
                scales: {
                    x: { ...baseScales.x, title: { display: true, text: xCol, color: '#aaa', font: { size: 10 } } },
                    y: { ...baseScales.y, title: { display: true, text: yCol, color: '#aaa', font: { size: 10 } } },
                },
            },
        });
    }
}

// ── Alpine component ──────────────────────────────────────────────────────────

document.addEventListener('alpine:init', () => {
    Alpine.data('aiExplorer', () => ({

        question: '',
        loading:  false,
        history:  [],   // [{question, sql, columns, rows, summary, error, chartType, done}]

        fmtCell(val) {
            if (val === null || val === undefined) return '—';
            if (typeof val === 'number') {
                return Number.isInteger(val) ? val.toString() : val.toFixed(4);
            }
            return String(val);
        },

        chartLabel(chartType) {
            if (!chartType) return '';
            return chartType.charAt(0).toUpperCase() + chartType.slice(1) + ' Chart';
        },

        clearHistory() {
            Object.keys(_explorerCharts).forEach(k => {
                _explorerCharts[k].destroy();
                delete _explorerCharts[k];
            });
            this.history = [];
        },

        _scrollToBottom() {
            const el = document.querySelector('.ai-response');
            if (el) el.scrollTop = el.scrollHeight;
        },

        async submit() {
            const q = this.question.trim();
            if (!q || this.loading) return;

            this.loading  = true;
            this.question = '';

            // Build history payload for the backend (no row data — just metadata)
            const historyPayload = this.history.map(t => ({
                question: t.question,
                sql:      t.sql     || null,
                summary:  t.summary || null,
                error:    t.error   || null,
            }));

            // Push a pending turn immediately so the loading card renders
            this.history.push({
                question: q,
                sql: null, columns: [], rows: [],
                summary: null, error: null, chartType: null,
                done: false,
            });
            const idx = this.history.length - 1;

            await this.$nextTick();
            this._scrollToBottom();

            try {
                const res = await fetch('/api/ai-explorer/query', {
                    method:  'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body:    JSON.stringify({ question: q, history: historyPayload }),
                });

                let data;
                try {
                    data = await res.json();
                } catch {
                    const text = await res.text().catch(() => '');
                    this.history[idx].error = `Server error (HTTP ${res.status}): ${text.slice(0, 300)}`;
                    this.history[idx].done  = true;
                    return;
                }

                if (!res.ok) {
                    this.history[idx].error = data.detail ?? `HTTP ${res.status}`;
                    this.history[idx].done  = true;
                    return;
                }

                // Mutate the turn in-place — Alpine's proxy picks up the changes
                this.history[idx].sql     = data.sql     ?? null;
                this.history[idx].columns = data.columns ?? [];
                this.history[idx].rows    = data.rows    ?? [];
                this.history[idx].summary = data.summary ?? null;
                this.history[idx].error   = data.error   ?? null;
                this.history[idx].done    = true;

                if (!data.error && data.rows?.length) {
                    const ct = detectChartType(data.columns, data.rows);
                    this.history[idx].chartType = ct;
                    if (ct) {
                        await this.$nextTick();
                        renderTurnChart(idx, ct, data.columns, data.rows);
                    }
                }

            } catch (e) {
                this.history[idx].error = e.message ?? 'Request failed';
                this.history[idx].done  = true;
            } finally {
                this.loading = false;
                await this.$nextTick();
                this._scrollToBottom();
            }
        },

        onKeydown(e) {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                this.submit();
            }
        },

    }));
});
