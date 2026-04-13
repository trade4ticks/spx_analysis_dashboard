/**
 * heatmap.js — Alpine component for the IV surface heatmap page.
 *
 * Color conventions
 * -----------------
 * IV Raw:
 *   p05 (low IV)  → pink  #ff1a8c
 *   p50 (median)  → dark  #2a2a2a
 *   p95 (high IV) → blue  #1a8cff
 *   Interpolated linearly between these stops.
 *
 * SKEW / TERM Raw:
 *   min → pink, 0 → dark, max → blue (fixed ranges below)
 *
 * 1D Change (all modes):
 *   negative → blue  #1a8cff  (IV fell)
 *   zero     → dark  #2a2a2a
 *   positive → pink  #ff1a8c  (IV rose)
 *   Range: ±0.10 for IV, ±0.30 for SKEW, ±0.05 for TERM
 */

// Fixed display ranges for raw SKEW and TERM
const SKEW_RAW_MIN = -2.0,  SKEW_RAW_MAX = 0.0;
const TERM_RAW_MIN =  0.10, TERM_RAW_MAX = 0.35;

// 1D change half-ranges (symmetric around 0)
const IV_CHG_RANGE   = 0.10;
const SKEW_CHG_RANGE = 0.30;
const TERM_CHG_RANGE = 0.05;

// Color stops
const COLOR_LOW  = [255, 26, 140];   // #ff1a8c  pink
const COLOR_MID  = [42,  42,  42];   // #2a2a2a  dark gray
const COLOR_HIGH = [26, 140, 255];   // #1a8cff  blue

function lerpRGB(a, b, t) {
    t = Math.max(0, Math.min(1, t));
    return [
        Math.round(a[0] + (b[0] - a[0]) * t),
        Math.round(a[1] + (b[1] - a[1]) * t),
        Math.round(a[2] + (b[2] - a[2]) * t),
    ];
}

function toHex([r, g, b]) {
    return '#' + [r, g, b].map(v => v.toString(16).padStart(2, '0')).join('');
}

/** 3-stop interpolation: low→mid at t=0.5, mid→high at t=1 */
function threeStop(t) {
    t = Math.max(0, Math.min(1, t));
    if (t <= 0.5) return lerpRGB(COLOR_LOW, COLOR_MID, t * 2);
    return lerpRGB(COLOR_MID, COLOR_HIGH, (t - 0.5) * 2);
}

/**
 * colorForRaw — color a raw value given a [min, mid, max] triple.
 * min→pink (t=0), mid→dark (t=0.5), max→blue (t=1).
 */
function colorForRaw(v, min, mid, max) {
    if (v == null || isNaN(v)) return '#222';
    let t;
    if (v <= mid) {
        t = (max === min) ? 0.5 : 0.5 * (v - min) / (mid - min || 1);
    } else {
        t = (max === min) ? 0.5 : 0.5 + 0.5 * (v - mid) / (max - mid || 1);
    }
    return toHex(threeStop(t));
}

/**
 * colorForChange — negative→blue, 0→dark, positive→pink.
 * range is the half-range (symmetric).
 */
function colorForChange(v, range) {
    if (v == null || isNaN(v)) return '#222';
    const t = 0.5 - 0.5 * Math.max(-1, Math.min(1, v / range));
    return toHex(threeStop(t));
}

/** Determine text color (black or white) for readability against bg. */
function contrastColor(hex) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    return lum > 0.45 ? '#111' : '#ddd';
}

function fmtPct(v, decimals = 1) {
    if (v == null || isNaN(v)) return '—';
    return (v * 100).toFixed(decimals) + '%';
}

function fmtNum(v, decimals = 3) {
    if (v == null || isNaN(v)) return '—';
    return v.toFixed(decimals);
}

/** Format display value per mode/submode */
function fmtCell(mode, submode, v) {
    if (v == null || isNaN(v)) return '—';
    if (submode === 'change') {
        const sign = v > 0 ? '+' : '';
        if (mode === 'iv') return sign + fmtPct(v, 1);
        return sign + fmtNum(v, 3);
    }
    if (mode === 'iv')   return fmtPct(v, 1);
    if (mode === 'skew') return fmtNum(v, 3);
    if (mode === 'term') return fmtPct(v, 1);
    return fmtNum(v, 3);
}

document.addEventListener('alpine:init', () => {
    Alpine.data('heatmap', () => ({
        // ── State ────────────────────────────────────────────────────────────
        mode:    'iv',       // iv | skew | term
        submode: 'raw',      // raw | change

        dates:  [],
        times:  [],
        date:   null,
        time:   null,

        loading: false,
        error:   null,

        // Raw grid data from API
        current: [],  // [{dte, put_delta, v}, …]
        prev:    [],  // for 1D change
        stats:   {},  // {dte_pd: {p05, p50, p95}} for IV raw coloring

        // Derived sorted axis values
        dtes:       [],
        putDeltas:  [],

        // ── Init ─────────────────────────────────────────────────────────────
        async init() {
            await this.loadDates();
        },

        async loadDates() {
            try {
                const res   = await fetch('/api/meta/dates');
                this.dates  = await res.json();
                if (this.dates.length) {
                    this.date = this.dates[0];
                    await this.loadTimes();
                }
            } catch(e) { this.error = 'Failed to load dates'; }
        },

        async loadTimes() {
            if (!this.date) return;
            try {
                const res  = await fetch(`/api/meta/quote_times?date=${this.date}`);
                this.times = await res.json();
                this.time  = this.closestTime(this.times, this.time ?? '15:45');
                await this.loadGrid();
            } catch(e) { this.error = 'Failed to load times'; }
        },

        closestTime(times, target) {
            if (!times.length) return null;
            const toSec = t => { const [h, m] = t.split(':').map(Number); return h * 60 + m; };
            const tSec  = toSec(target);
            return times.reduce((best, t) =>
                Math.abs(toSec(t) - tSec) < Math.abs(toSec(best) - tSec) ? t : best
            );
        },

        async onDateChange() {
            await this.loadTimes();
        },

        async onTimeChange() {
            await this.loadGrid();
        },

        async onModeChange(m) {
            this.mode = m;
            await this.loadGrid();
        },

        async onSubmodeChange(s) {
            this.submode = s;
            this.renderGrid();   // data already loaded; just recolor
        },

        // ── Data loading ─────────────────────────────────────────────────────
        get prevDate() {
            const idx = this.dates.indexOf(this.date);
            return (idx >= 0 && idx + 1 < this.dates.length) ? this.dates[idx + 1] : null;
        },

        async loadGrid() {
            if (!this.date || !this.time) return;
            this.loading = true;
            this.error   = null;

            try {
                const prev  = this.prevDate;
                const query = prev
                    ? `date=${this.date}&time=${this.time}&prev_date=${prev}&prev_time=15:45`
                    : `date=${this.date}&time=${this.time}`;

                const [gridRes, statsRes] = await Promise.all([
                    fetch(`/api/heatmap/${this.mode}?${query}`),
                    this.mode === 'iv' && !Object.keys(this.stats).length
                        ? fetch('/api/heatmap/node_stats')
                        : Promise.resolve(null),
                ]);

                const grid = await gridRes.json();
                this.current = grid.current ?? [];
                this.prev    = grid.prev    ?? [];

                if (statsRes) {
                    this.stats = await statsRes.json();
                }

                // Derive sorted axes
                const dteSet   = new Set(this.current.map(r => r.dte));
                const pdSet    = new Set(this.current.map(r => r.put_delta));
                this.dtes      = [...dteSet].sort((a, b) => a - b);
                this.putDeltas = [...pdSet].sort((a, b) => a - b);

                this.renderGrid();
            } catch(e) {
                this.error = 'Failed to load heatmap data';
                console.error(e);
            } finally {
                this.loading = false;
            }
        },

        // ── Grid rendering ────────────────────────────────────────────────────
        renderGrid() {
            const container = document.getElementById('heatmap-container');
            if (!container) return;
            if (!this.current.length) {
                container.innerHTML = '<div class="hm-empty">No data for this snapshot.</div>';
                return;
            }

            // Build lookup maps
            const curMap  = new Map(this.current.map(r => [`${r.dte}_${r.put_delta}`, r.v]));
            const prevMap = new Map((this.prev || []).map(r => [`${r.dte}_${r.put_delta}`, r.v]));

            const mode    = this.mode;
            const submode = this.submode;
            const stats   = this.stats;
            const dtes    = this.dtes;
            const pds     = this.putDeltas;

            // Choose change half-range
            const chgRange = mode === 'iv' ? IV_CHG_RANGE
                           : mode === 'skew' ? SKEW_CHG_RANGE
                           : TERM_CHG_RANGE;

            // Delta column header labels (put_delta → display)
            function pdLabel(pd) {
                if (pd === 50) return 'ATM';
                if (pd < 50)  return `${pd}p`;
                return `${100 - pd}c`;
            }

            let html = '<table class="hm-table"><thead><tr><th class="hm-dte-hdr">DTE</th>';
            for (const pd of pds) {
                html += `<th>${pdLabel(pd)}</th>`;
            }
            html += '</tr></thead><tbody>';

            for (const dte of dtes) {
                html += `<tr><td class="hm-dte-cell">${dte}</td>`;
                for (const pd of pds) {
                    const key   = `${dte}_${pd}`;
                    const cur   = curMap.get(key);
                    const prv   = prevMap.get(key);
                    const chg   = (cur != null && prv != null) ? cur - prv : null;

                    let bg, displayVal, title;

                    if (submode === 'change') {
                        bg         = colorForChange(chg, chgRange);
                        displayVal = fmtCell(mode, 'change', chg);
                        const parts = [
                            `DTE=${dte}  Δ=${pdLabel(pd)}`,
                            `Change: ${displayVal}`,
                            `Current: ${fmtCell(mode, 'raw', cur)}`,
                            `Prev:    ${fmtCell(mode, 'raw', prv)}`,
                        ];
                        title = parts.join('\n');
                    } else {
                        // Raw mode
                        if (mode === 'iv') {
                            const st  = stats[key];
                            const p05 = st?.p05 ?? null;
                            const p50 = st?.p50 ?? null;
                            const p95 = st?.p95 ?? null;
                            bg = (cur != null && p05 != null)
                                ? colorForRaw(cur, p05, p50, p95)
                                : '#222';
                            displayVal = fmtCell('iv', 'raw', cur);
                            title = [
                                `DTE=${dte}  Δ=${pdLabel(pd)}`,
                                `IV: ${displayVal}`,
                                `p05=${fmtPct(p05)}  p50=${fmtPct(p50)}  p95=${fmtPct(p95)}`,
                            ].join('\n');
                        } else if (mode === 'skew') {
                            bg = colorForRaw(cur, SKEW_RAW_MIN, (SKEW_RAW_MIN + SKEW_RAW_MAX) / 2, SKEW_RAW_MAX);
                            displayVal = fmtCell('skew', 'raw', cur);
                            title = `DTE=${dte}  Δ=${pdLabel(pd)}\nSkew slope: ${displayVal}`;
                        } else {
                            bg = colorForRaw(cur, TERM_RAW_MIN, (TERM_RAW_MIN + TERM_RAW_MAX) / 2, TERM_RAW_MAX);
                            displayVal = fmtCell('term', 'raw', cur);
                            title = `DTE=${dte}  Δ=${pdLabel(pd)}\nFwd vol (vs 30D): ${displayVal}`;
                        }
                    }

                    const fg = contrastColor(bg);
                    const esc = title.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
                    html += `<td style="background:${bg};color:${fg}" title="${esc}">${displayVal}</td>`;
                }
                html += '</tr>';
            }

            html += '</tbody></table>';
            container.innerHTML = html;
        },

        // ── Helpers ───────────────────────────────────────────────────────────
        get dateDisplay() {
            return this.date ?? '—';
        },
        get timeDisplay() {
            return this.time ?? '—';
        },
    }));
});
