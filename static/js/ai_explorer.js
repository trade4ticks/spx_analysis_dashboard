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
            zoom: {
                pan:  { enabled: true, mode: 'xy', modifierKey: 'shift' },
                zoom: {
                    wheel: { enabled: true },
                    pinch: { enabled: true },
                    drag:  { enabled: true, backgroundColor: 'rgba(52,152,219,0.15)',
                             borderColor: '#3498db', borderWidth: 1 },
                    mode:  'xy',
                },
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
        const colorCol = numericCols.length >= 3 ? numericCols[2] : null;

        let bgColors;
        if (colorCol) {
            // Gradient from red (negative) → gray (zero) → green (positive)
            const cVals = rows.map(r => r[colorCol]).filter(v => v != null);
            const cMin  = Math.min(...cVals), cMax = Math.max(...cVals);
            const cRng  = Math.max(Math.abs(cMin), Math.abs(cMax)) || 1;
            bgColors = rows.map(r => {
                const v = r[colorCol];
                if (v == null) return 'rgba(128,128,128,0.5)';
                const t = v / cRng;  // -1 to +1
                if (t >= 0) return `rgba(${Math.round(46 + (1-t)*80)},${Math.round(204 - (1-t)*80)},${Math.round(113 - (1-t)*40)},0.75)`;
                const a = -t;
                return `rgba(${Math.round(231 - (1-a)*80)},${Math.round(76 + (1-a)*80)},${Math.round(60 + (1-a)*40)},0.75)`;
            });
        } else {
            bgColors = '#3498dbaa';
        }

        _explorerCharts[idx] = new Chart(canvas, {
            type: 'scatter',
            data: {
                datasets: [{
                    label:           `${xCol} vs ${yCol}`,
                    data:            rows.map(r => ({ x: r[xCol], y: r[yCol] })),
                    backgroundColor: bgColors,
                    pointRadius:     4,
                    pointHoverRadius: 6,
                    borderWidth:     0,
                }],
            },
            options: {
                ...baseOpts,
                plugins: {
                    ...baseOpts.plugins,
                    tooltip: {
                        ...baseOpts.plugins.tooltip,
                        callbacks: {
                            label: ctx => {
                                const r = rows[ctx.dataIndex];
                                return columns.map(col => {
                                    const v = r[col];
                                    if (v == null) return `${col}: —`;
                                    if (typeof v === 'number') return `${col}: ${Number.isInteger(v) ? v : v.toFixed(4)}`;
                                    return `${col}: ${v}`;
                                });
                            },
                        },
                    },
                },
                scales: {
                    x: { ...baseScales.x, title: { display: true, text: xCol, color: '#aaa', font: { size: 10 } } },
                    y: { ...baseScales.y, title: { display: true, text: yCol, color: '#aaa', font: { size: 10 } } },
                },
            },
        });
    }
}

// ── Config-driven chart renderer (Claude provides Chart.js spec) ─────────────

function renderFromConfig(idx, config, columns, rows) {
    const canvasId = `explorer-chart-${idx}`;
    const canvas   = document.getElementById(canvasId);
    if (!canvas) return;

    if (_explorerCharts[idx]) { _explorerCharts[idx].destroy(); delete _explorerCharts[idx]; }
    if (!rows.length || !config.datasets?.length) return;

    const chartType  = config.type || 'line';
    const configDs   = config.datasets;
    const datasets   = [];
    let labels       = null;

    for (const ds of configDs) {
        const out = {};

        if (ds.regression) {
            // ── Compute least-squares regression ──────────────────────
            const pairs = rows
                .map(r => [r[ds.xSource], r[ds.ySource]])
                .filter(([a, b]) => a != null && b != null);
            if (pairs.length >= 2) {
                let sx=0,sy=0,sxx=0,sxy=0;
                for (const [a,b] of pairs) { sx+=a; sy+=b; sxx+=a*a; sxy+=a*b; }
                const n=pairs.length, slope=(n*sxy-sx*sy)/(n*sxx-sx*sx), inter=(sy-slope*sx)/n;
                const xs=pairs.map(p=>p[0]), xMin=Math.min(...xs), xMax=Math.max(...xs);
                out.data = [{x:xMin,y:inter+slope*xMin},{x:xMax,y:inter+slope*xMax}];
            } else {
                out.data = [];
            }
            out.type = ds.type || 'line';

        } else if (ds.xColumn && ds.yColumn) {
            // ── Scatter x/y ──────────────────────────────────────────
            out.data = rows.map(r => ({ x: r[ds.xColumn], y: r[ds.yColumn] }));

            if (ds.colorColumn) {
                const cVals = rows.map(r => r[ds.colorColumn]).filter(v => v != null);
                const cMax  = Math.max(Math.abs(Math.min(...cVals)), Math.abs(Math.max(...cVals))) || 1;
                out.backgroundColor = rows.map(r => {
                    const v = r[ds.colorColumn];
                    if (v == null) return 'rgba(128,128,128,0.5)';
                    const t = v / cMax;
                    if (t >= 0) return `rgba(${Math.round(46+(1-t)*80)},${Math.round(204-(1-t)*80)},${Math.round(113-(1-t)*40)},0.75)`;
                    const a = -t;
                    return `rgba(${Math.round(231-(1-a)*80)},${Math.round(76+(1-a)*80)},${Math.round(60+(1-a)*40)},0.75)`;
                });
            }

        } else if (ds.labelsColumn && ds.yColumn) {
            // ── Line/bar with label column ───────────────────────────
            if (!labels) labels = rows.map(r => String(r[ds.labelsColumn] ?? ''));
            out.data = rows.map(r => r[ds.yColumn]);

        } else if (ds.yColumn) {
            out.data = rows.map(r => r[ds.yColumn]);
        }

        // Copy all standard Chart.js props from the spec (skip our custom keys)
        const skip = new Set(['xColumn','yColumn','labelsColumn','colorColumn',
                              'regression','xSource','ySource']);
        for (const [k, v] of Object.entries(ds)) {
            if (!skip.has(k) && !(k in out)) out[k] = v;
        }
        datasets.push(out);
    }

    // Auto-detect labels for line/bar if none came from labelsColumn
    if (!labels && chartType !== 'scatter') {
        const s = rows[0];
        const dc = columns.filter(c => _isDatelike(s[c]));
        const tc = columns.filter(c => _isTimelike(s[c]));
        if (dc.length && tc.length) labels = rows.map(r => `${r[dc[0]]} ${String(r[tc[0]]).substring(0,5)}`);
        else if (dc.length)         labels = rows.map(r => r[dc[0]]);
        else if (tc.length)         labels = rows.map(r => String(r[tc[0]]).substring(0,5));
    }

    // Merge user options with dark-theme defaults
    const uo = config.options || {};
    const baseS = {
        x: { ticks:{color:'#888',font:{size:9},maxTicksLimit:12,maxRotation:45},
             grid:{color:'rgba(255,255,255,0.05)'}, border:{color:'transparent'} },
        y: { ticks:{color:'#888',font:{size:9}},
             grid:{color:'rgba(255,255,255,0.05)'}, border:{color:'transparent'} },
    };
    const scales = {};
    for (const ax of ['x','y']) {
        scales[ax] = { ...baseS[ax], ...(uo.scales?.[ax]||{}) };
        scales[ax].ticks = { ...baseS[ax].ticks, ...(uo.scales?.[ax]?.ticks||{}) };
        scales[ax].grid  = { ...baseS[ax].grid,  ...(uo.scales?.[ax]?.grid||{}) };
        if (uo.scales?.[ax]?.title) scales[ax].title = uo.scales[ax].title;
    }

    const opts = {
        responsive: true, maintainAspectRatio: false, animation: false,
        plugins: {
            legend: { display: datasets.length > 1,
                      labels: {color:'#aaa',font:{size:10},boxWidth:14,padding:10} },
            tooltip: {
                backgroundColor: 'rgba(20,20,20,0.92)',
                titleColor: '#999', bodyColor: '#ddd',
                borderColor: '#444', borderWidth: 1,
                filter: item => !configDs[item.datasetIndex]?.regression,
                callbacks: {
                    label: ctx => {
                        const r = rows[ctx.dataIndex];
                        if (!r) return `${ctx.dataset.label ?? ''}: ${ctx.parsed.y}`;
                        return columns.map(col => {
                            const v = r[col];
                            if (v == null) return `${col}: —`;
                            if (typeof v === 'number') return `${col}: ${Number.isInteger(v)?v:v.toFixed(4)}`;
                            return `${col}: ${v}`;
                        });
                    },
                },
            },
            zoom: {
                pan:  { enabled: true, mode: 'xy', modifierKey: 'shift' },
                zoom: { wheel:{enabled:true}, pinch:{enabled:true},
                        drag:{enabled:true,backgroundColor:'rgba(52,152,219,0.15)',
                              borderColor:'#3498db',borderWidth:1}, mode:'xy' },
            },
        },
        scales,
    };

    const chartData = { datasets };
    if (labels) chartData.labels = labels;

    _explorerCharts[idx] = new Chart(canvas, { type: chartType, data: chartData, options: opts });
}

// ── Alpine component ──────────────────────────────────────────────────────────

document.addEventListener('alpine:init', () => {
    Alpine.data('aiExplorer', () => ({

        question: '',
        loading:  false,
        history:  [],   // [{question, sql, columns, rows, summary, error, chartType, done}]
        expandedChartIdx: null,

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

        resetChartZoom(key) {
            const chart = _explorerCharts[key];
            if (chart && chart.resetZoom) chart.resetZoom();
        },

        expandChart(idx) {
            const turn = this.history[idx];
            if (!turn) return;
            this.expandedChartIdx = idx;
            this.$nextTick(() => {
                if (turn.chartConfig) {
                    renderFromConfig('fs', turn.chartConfig, turn.columns, turn.rows);
                } else {
                    renderTurnChart('fs', turn.chartType, turn.columns, turn.rows);
                }
            });
        },

        collapseChart() {
            if (_explorerCharts['fs']) {
                _explorerCharts['fs'].destroy();
                delete _explorerCharts['fs'];
            }
            this.expandedChartIdx = null;
        },

        _scrollToBottom() {
            const el = document.querySelector('.ai-response');
            if (el) el.scrollTop = el.scrollHeight;
        },

        _updateTurn(idx, patch) {
            // splice triggers Alpine's array Proxy reliably; direct index mutation does not
            this.history.splice(idx, 1, { ...this.history[idx], ...patch });
        },

        async submit() {
            const q = this.question.trim();
            if (!q || this.loading) return;

            this.loading  = true;
            this.question = '';

            const historyPayload = this.history.map(t => ({
                question: t.question,
                sql:      t.sql     || null,
                summary:  t.summary || null,
                error:    t.error   || null,
            }));

            // Array reassignment (not push) guarantees Alpine detects the change
            this.history = [...this.history, {
                question: q,
                sql: null, columns: [], rows: [],
                summary: null, error: null, chartType: null, chartConfig: null,
                done: false,
            }];
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
                    this._updateTurn(idx, { error: `Server error (HTTP ${res.status}): ${text.slice(0, 300)}`, done: true });
                    return;
                }

                if (!res.ok) {
                    this._updateTurn(idx, { error: data.detail ?? `HTTP ${res.status}`, done: true });
                    return;
                }

                let ct = null;
                const cc = data.chart_config ?? null;
                if (!data.error && data.rows?.length) {
                    if (cc && cc.type) {
                        ct = cc.type;
                    } else if (data.chart_hint && ['line','bar','scatter'].includes(data.chart_hint)) {
                        ct = data.chart_hint;
                    } else {
                        ct = detectChartType(data.columns ?? [], data.rows);
                    }
                }

                this._updateTurn(idx, {
                    sql:         data.sql      ?? null,
                    columns:     data.columns  ?? [],
                    rows:        data.rows     ?? [],
                    summary:     data.summary  ?? null,
                    error:       data.error    ?? null,
                    chartType:   ct,
                    chartConfig: cc,
                    done:        true,
                });

                if (ct && data.rows?.length) {
                    await this.$nextTick();
                    if (cc) {
                        renderFromConfig(idx, cc, data.columns, data.rows);
                    } else {
                        renderTurnChart(idx, ct, data.columns, data.rows);
                    }
                }

            } catch (e) {
                this._updateTurn(idx, { error: e.message ?? 'Request failed', done: true });
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
