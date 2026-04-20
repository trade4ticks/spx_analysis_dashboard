/**
 * today.js — Alpine component for the intraday strike IV heatmap.
 *
 * Data source: raw parquet files via /api/raw/expirations + /api/today/iv_grid
 *
 * Grid layout
 * -----------
 * Rows    = time slices 09:35 → 16:00 in 5-min steps (78 slots)
 * Columns = 15 strikes anchored to opening spot (10 below, 5 above, ×100)
 *
 * Modes
 * -----
 * IV      — raw IV, displayed as percentage (e.g. 18.25)
 *           neutral dark background
 *
 * 5min Chg — IV change vs prior 5-min slice (09:35 uses prev-day 16:00)
 *            displayed in percentage points (e.g. +0.22)
 *            pink  #ff1a8c → positive  (IV rose)
 *            dark  #2a2a2a → zero
 *            blue  #1a8cff → negative  (IV fell)
 *            range ±1.0 pp
 */

// ── Color helpers ─────────────────────────────────────────────────────────────
const _C_LOW  = [255, 26, 140];  // #ff1a8c  pink
const _C_MID  = [42,  42,  42];  // #2a2a2a  dark gray
const _C_HIGH = [26, 140, 255];  // #1a8cff  blue

function _lerp(a, b, t) {
    t = Math.max(0, Math.min(1, t));
    return [
        Math.round(a[0] + (b[0] - a[0]) * t),
        Math.round(a[1] + (b[1] - a[1]) * t),
        Math.round(a[2] + (b[2] - a[2]) * t),
    ];
}
function _hex([r, g, b]) {
    return '#' + [r, g, b].map(v => v.toString(16).padStart(2, '0')).join('');
}
function _three(t) {
    t = Math.max(0, Math.min(1, t));
    return t <= 0.5 ? _lerp(_C_LOW, _C_MID, t * 2) : _lerp(_C_MID, _C_HIGH, (t - 0.5) * 2);
}
/** positive → pink, zero → dark, negative → blue */
function chgColor(v_pp, range = 1.0) {
    return _hex(_three(0.5 - 0.5 * Math.max(-1, Math.min(1, v_pp / range))));
}
function contrast(hex) {
    const r = parseInt(hex.slice(1,3), 16);
    const g = parseInt(hex.slice(3,5), 16);
    const b = parseInt(hex.slice(5,7), 16);
    return (0.299*r + 0.587*g + 0.114*b) / 255 > 0.45 ? '#111' : '#ddd';
}

// ── Time grid ─────────────────────────────────────────────────────────────────
function allTimeSlots() {
    const slots = [];
    for (let m = 35; m <= 55; m += 5)
        slots.push(`09:${String(m).padStart(2, '0')}`);
    for (let h = 10; h <= 15; h++)
        for (let m = 0; m <= 55; m += 5)
            slots.push(`${h}:${String(m).padStart(2, '0')}`);
    slots.push('16:00');
    return slots;   // 78 entries
}
function prevSlot(t) {
    const [h, m] = t.split(':').map(Number);
    const tot = h * 60 + m - 5;
    if (tot < 0) return null;
    return `${String(Math.floor(tot/60)).padStart(2,'0')}:${String(tot%60).padStart(2,'0')}`;
}

// ── Scatter chart instance (outside Alpine to avoid reactive proxy issues) ────
let _scatterChart = null;

// ── Alpine component ──────────────────────────────────────────────────────────
document.addEventListener('alpine:init', () => {
    Alpine.data('today', () => ({
        // Controls
        mode: 'iv',           // iv | change
        dates: [],
        date:  null,
        expirations: [],      // [{expiration, dte, label, settlements}]
        selectedExp: null,    // "YYYY-MM-DD"
        selectedSettlement: 'PM',

        // Data
        loading:    false,
        error:      null,
        strikes:    [],       // [5900, 6000, …]
        rows:       [],       // [{time, data:{strike_str: iv}}, …]
        prev:       {},       // {strike_str: iv}  prev-day 16:00
        spotSeries: [],       // [{time, price}, …]

        // Scatter
        scatterDays:     30,
        scatterY:        'vix',   // vix | vix9d | vix3m
        scatterPoints:   [],
        scatterExpanded: false,

        ALL_TIMES: allTimeSlots(),

        // ── Init ─────────────────────────────────────────────────────────────
        async init() {
            await Promise.all([this.loadDates(), this.loadScatter()]);
        },

        async loadDates() {
            try {
                const res  = await fetch('/api/meta/dates');
                this.dates = await res.json();
                if (this.dates.length) {
                    this.date = this.dates[0];
                    await this.loadExpirations();
                }
            } catch(e) { this.error = 'Failed to load dates'; }
        },

        async onDateChange() {
            await this.loadExpirations();
        },

        async loadExpirations() {
            if (!this.date) return;
            try {
                // Use raw parquet expirations — actual expiration folders on disk
                const res  = await fetch(`/api/raw/expirations?date=${this.date}`);
                const data = await res.json();
                // Raw API returns {expiration, dte, settlements} — add label here
                this.expirations = (data.expirations ?? []).map(e => ({
                    ...e,
                    label: `${e.expiration}  (${e.dte} DTE)`,
                }));

                if (!this.expirations.length) {
                    this.error = `No raw parquet data found for ${this.date}`;
                    return;
                }
                this.error = null;

                // Default: expiration closest to 30 DTE
                const best = this.expirations.reduce((b, e) =>
                    Math.abs(e.dte - 30) < Math.abs(b.dte - 30) ? e : b
                );
                this.selectedExp        = best.expiration;
                this.selectedSettlement = best.settlements.includes('PM') ? 'PM' : best.settlements[0];
                await this.loadGrid();
            } catch(e) {
                this.error = 'Failed to load expirations';
                console.error(e);
            }
        },

        async onExpirationChange() {
            // Update settlement to match the newly selected expiration
            const exp = this.expirations.find(e => e.expiration === this.selectedExp);
            if (exp) {
                this.selectedSettlement = exp.settlements.includes('PM') ? 'PM' : exp.settlements[0];
            }
            await this.loadGrid();
        },

        async loadGrid() {
            if (!this.date || !this.selectedExp) return;
            this.loading = true;
            this.error   = null;
            try {
                const url = `/api/today/iv_grid?date=${this.date}&expiration=${this.selectedExp}&settlement=${this.selectedSettlement}`;
                const res  = await fetch(url);
                if (!res.ok) {
                    const err = await res.json().catch(() => ({}));
                    throw new Error(err.detail ?? `HTTP ${res.status}`);
                }
                const data      = await res.json();
                this.strikes    = data.strikes;
                this.rows       = data.rows;
                this.prev       = data.prev;
                this.spotSeries = data.spot_series ?? [];
                this.renderGrid();
            } catch(e) {
                this.error = `Failed to load grid: ${e.message}`;
                console.error(e);
            } finally {
                this.loading = false;
            }
        },

        onModeChange(m) {
            this.mode = m;
            this.renderGrid();
        },

        // ── Render ────────────────────────────────────────────────────────────
        renderGrid() {
            const container = document.getElementById('today-container');
            if (!container) return;
            if (!this.strikes.length) {
                container.innerHTML = '<div class="hm-empty">No data for this selection.</div>';
                return;
            }

            const dataMap = {};
            for (const row of this.rows) dataMap[row.time] = row.data;

            // For vs-Open mode: use 09:35 slice (or earliest available)
            const openData = dataMap['09:35']
                          ?? (this.rows.length ? this.rows[0].data : {});

            const mode    = this.mode;
            const prev    = this.prev;
            const strikes = this.strikes;

            let html = '<table class="today-table"><thead><tr>';
            html += '<th class="today-time-hdr">Time</th>';
            for (const s of strikes) html += `<th>${s}</th>`;
            html += '</tr></thead><tbody>';

            for (const t of this.ALL_TIMES) {
                html += `<tr><td class="today-time-cell">${t}</td>`;

                const rowData  = dataMap[t] || {};
                const prevData = t === '09:35' ? prev
                               : (dataMap[prevSlot(t)] || {});

                for (const s of strikes) {
                    const sk = String(s);
                    const iv = rowData[sk];

                    let bg = '#1e1e1e', fg = '#555', display = '', title = '';

                    if (iv != null) {
                        if (mode === 'iv') {
                            bg      = '#252525';
                            fg      = '#ccc';
                            display = (iv * 100).toFixed(2);
                            title   = `${t}  K=${s}\nIV: ${display}%`;
                        } else if (mode === 'change') {
                            const prevIv = prevData[sk];
                            if (prevIv != null) {
                                const chg = (iv - prevIv) * 100;
                                bg      = chgColor(chg, 1.0);
                                fg      = contrast(bg);
                                display = (chg >= 0 ? '+' : '') + chg.toFixed(2);
                                title   = `${t}  K=${s}\nChg: ${display} pp\nIV: ${(iv*100).toFixed(2)}%  Prev: ${(prevIv*100).toFixed(2)}%`;
                            } else {
                                bg      = '#252525';
                                fg      = '#555';
                                display = (iv * 100).toFixed(2);
                                title   = `${t}  K=${s}\nIV: ${display}% (no prior)`;
                            }
                        } else {
                            // open_chg: vs 09:35
                            const openIv = openData[sk];
                            if (openIv != null && t !== '09:35') {
                                const chg = (iv - openIv) * 100;
                                bg      = chgColor(chg, 2.0);
                                fg      = contrast(bg);
                                display = (chg >= 0 ? '+' : '') + chg.toFixed(2);
                                title   = `${t}  K=${s}\nVs Open: ${display} pp\nIV: ${(iv*100).toFixed(2)}%  Open: ${(openIv*100).toFixed(2)}%`;
                            } else if (openIv != null && t === '09:35') {
                                // reference row — show IV, no color
                                bg      = '#252525';
                                fg      = '#777';
                                display = (iv * 100).toFixed(2);
                                title   = `${t}  K=${s}\nIV (open reference): ${display}%`;
                            } else {
                                bg      = '#252525';
                                fg      = '#555';
                                display = (iv * 100).toFixed(2);
                                title   = `${t}  K=${s}\nIV: ${display}% (no open ref)`;
                            }
                        }
                    }

                    const esc = title.replace(/"/g, '&quot;');
                    html += `<td style="background:${bg};color:${fg}" title="${esc}">${display}</td>`;
                }
                html += '</tr>';
            }

            html += '</tbody></table>';
            container.innerHTML = html;
        },

        // ── Scatter ───────────────────────────────────────────────────────────
        async loadScatter() {
            try {
                const res  = await fetch(`/api/today/scatter?days=${this.scatterDays}`);
                const data = await res.json();
                this.scatterPoints = data.points ?? [];
                this.renderScatter();
            } catch(e) {
                console.error('scatter load failed', e);
            }
        },

        setScatterDays(d) {
            this.scatterDays = d;
            this.loadScatter();
        },

        setScatterY(y) {
            this.scatterY = y;
            this.renderScatter();
        },

        toggleScatterExpand() {
            this.scatterExpanded = !this.scatterExpanded;
            // Destroy chart on the old canvas; re-render into the new canvas
            // after Alpine reveals/hides the overlay (one animation frame).
            if (_scatterChart) { _scatterChart.destroy(); _scatterChart = null; }
            setTimeout(() => this.renderScatter(), 50);
        },

        renderScatter() {
            const canvasId = this.scatterExpanded ? 'scatter-canvas-fs' : 'scatter-canvas';
            const canvas = document.getElementById(canvasId);
            if (!canvas) return;

            const yKey  = this.scatterY + '_change';
            const allPts = this.scatterPoints;

            // Filter to points that have valid data for both axes
            const filtered = allPts
                .map((p, i) => ({ ...p, origIdx: i }))
                .filter(p => p.spx_return != null && p[yKey] != null);

            if (_scatterChart) { _scatterChart.destroy(); _scatterChart = null; }
            if (!filtered.length) return;

            const n      = allPts.length;
            const data   = filtered.map(p => ({
                x: +(p.spx_return * 100).toFixed(3),
                y: +p[yKey].toFixed(3),
            }));
            // Most recent dot = full accent blue; oldest = very faint
            const colors = filtered.map(p => {
                const alpha = n > 1 ? 0.1 + (p.origIdx / (n - 1)) * 0.9 : 1;
                return `rgba(52,152,219,${alpha.toFixed(3)})`;
            });
            const radii  = filtered.map((_, i) => i === filtered.length - 1 ? 6 : 4);
            const dates  = filtered.map(p => p.date);

            const yLabel = { vix: 'VIX Δ', vix9d: 'VIX9D Δ', vix3m: 'VIX3M Δ' }[this.scatterY];

            // Zero-line color helper for both axes
            const gridColor = ctx => ctx.tick.value === 0
                ? 'rgba(255,255,255,0.20)'
                : 'rgba(255,255,255,0.05)';
            const gridWidth = ctx => ctx.tick.value === 0 ? 1 : 0.5;

            _scatterChart = new Chart(canvas, {
                type: 'scatter',
                data: {
                    datasets: [{
                        data,
                        backgroundColor:  colors,
                        pointRadius:      radii,
                        pointHoverRadius: 8,
                        borderWidth:      0,
                    }],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(20,20,20,0.92)',
                            titleColor:      '#999',
                            bodyColor:       '#ddd',
                            borderColor:     '#444',
                            borderWidth:     1,
                            callbacks: {
                                title: ctx => dates[ctx[0].dataIndex],
                                label: ctx => [
                                    `SPX: ${ctx.parsed.x >= 0 ? '+' : ''}${ctx.parsed.x.toFixed(2)}%`,
                                    `${yLabel}: ${ctx.parsed.y >= 0 ? '+' : ''}${ctx.parsed.y.toFixed(2)}`,
                                ],
                            },
                        },
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text:    'SPX Return (%)',
                                color:   '#555',
                                font:    { size: 10 },
                            },
                            ticks: {
                                color: '#555',
                                font:  { size: 9 },
                                callback: v => v + '%',
                            },
                            grid:   { color: gridColor, lineWidth: gridWidth },
                            border: { color: 'transparent' },
                        },
                        y: {
                            title: {
                                display: true,
                                text:    yLabel,
                                color:   '#555',
                                font:    { size: 10 },
                            },
                            ticks: { color: '#555', font: { size: 9 } },
                            grid:   { color: gridColor, lineWidth: gridWidth },
                            border: { color: 'transparent' },
                        },
                    },
                },
            });
        },

        // ── SPX sparkline ─────────────────────────────────────────────────────
        get latestSpot() {
            return this.spotSeries.length
                ? this.spotSeries[this.spotSeries.length - 1].price
                : null;
        },
        get openSpot() {
            return this.spotSeries.length ? this.spotSeries[0].price : null;
        },
        get latestSpotFmt() {
            const s = this.latestSpot;
            if (!s) return '—';
            return s.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        },
        get spotChgFmt() {
            if (!this.latestSpot || !this.openSpot) return '';
            const chg = this.latestSpot - this.openSpot;
            return (chg >= 0 ? '+' : '') + chg.toFixed(2);
        },
        get spotChgClass() {
            if (!this.latestSpot || !this.openSpot) return '';
            return this.latestSpot >= this.openSpot ? 'up' : 'dn';
        },
        get sparklineSvg() {
            const pts = this.spotSeries;
            if (pts.length < 2) return '';
            const W = 160, H = 28, PAD = 2;
            const prices = pts.map(p => p.price);
            const lo  = Math.min(...prices);
            const hi  = Math.max(...prices);
            const rng = hi - lo || 1;
            const n   = prices.length;
            const xf  = i => (PAD + (i / (n - 1)) * (W - 2*PAD)).toFixed(1);
            const yf  = p => (PAD + (1 - (p - lo) / rng) * (H - 2*PAD)).toFixed(1);
            const path = prices.map((p, i) => `${i === 0 ? 'M' : 'L'}${xf(i)} ${yf(p)}`).join(' ');
            const lx = xf(n-1), ly = yf(prices[n-1]);
            const clr = (this.latestSpot ?? 0) >= (this.openSpot ?? 0) ? '#2ecc71' : '#e74c3c';
            return `<svg width="${W}" height="${H}" style="display:block">` +
                `<path d="${path}" fill="none" stroke="${clr}" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>` +
                `<circle cx="${lx}" cy="${ly}" r="2.5" fill="${clr}"/>` +
                `</svg>`;
        },

        // ── Misc ──────────────────────────────────────────────────────────────
        get dateDisplay() { return this.date ?? '—'; },
        get selectedLabel() {
            const e = this.expirations.find(x => x.expiration === this.selectedExp);
            return e ? e.label : '—';
        },
    }));
});
