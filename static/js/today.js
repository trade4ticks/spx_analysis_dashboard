/**
 * today.js — Alpine component for the intraday strike IV heatmap.
 *
 * Rows    = time slices 09:35 → 16:00 in 5-min steps (78 rows)
 * Columns = 15 strikes anchored to the opening ATM (10 below, 5 above, ×100)
 *
 * Modes
 * -----
 * IV      — raw IV per cell, displayed as percentage (e.g. 18.25)
 *           neutral dark background, no gradient
 *
 * 5min Chg — change vs the prior 5-min slice (09:35 uses prev-day 16:00)
 *            displayed in percentage points (e.g. +0.22)
 *            pink  #ff1a8c → positive
 *            dark  #2a2a2a → zero
 *            blue  #1a8cff → negative
 *            range ±1.0 pp
 */

// ── Color constants ────────────────────────────────────────────────────────────
const COLOR_LOW_T  = [255, 26, 140];  // #ff1a8c  pink   (positive change / low IV)
const COLOR_MID_T  = [42,  42,  42];  // #2a2a2a  dark
const COLOR_HIGH_T = [26, 140, 255];  // #1a8cff  blue   (negative change / high IV)

function lerpRGB_T(a, b, t) {
    t = Math.max(0, Math.min(1, t));
    return [
        Math.round(a[0] + (b[0] - a[0]) * t),
        Math.round(a[1] + (b[1] - a[1]) * t),
        Math.round(a[2] + (b[2] - a[2]) * t),
    ];
}

function toHex_T([r, g, b]) {
    return '#' + [r, g, b].map(v => v.toString(16).padStart(2, '0')).join('');
}

function threeStop_T(t) {
    t = Math.max(0, Math.min(1, t));
    if (t <= 0.5) return lerpRGB_T(COLOR_LOW_T, COLOR_MID_T, t * 2);
    return lerpRGB_T(COLOR_MID_T, COLOR_HIGH_T, (t - 0.5) * 2);
}

/** positive → pink, zero → dark, negative → blue */
function changeColor(v_pp, range = 1.0) {
    const t = 0.5 - 0.5 * Math.max(-1, Math.min(1, v_pp / range));
    return toHex_T(threeStop_T(t));
}

function contrastColor_T(hex) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255 > 0.45 ? '#111' : '#ddd';
}

// ── Time grid ─────────────────────────────────────────────────────────────────
function allTimeSlots() {
    const slots = [];
    // 09:35 – 09:55
    for (let m = 35; m <= 55; m += 5)
        slots.push(`09:${String(m).padStart(2, '0')}`);
    // 10:00 – 15:55
    for (let h = 10; h <= 15; h++)
        for (let m = 0; m <= 55; m += 5)
            slots.push(`${h}:${String(m).padStart(2, '0')}`);
    // 16:00
    slots.push('16:00');
    return slots;  // 78 entries
}

/** Subtract 5 minutes from HH:MM string. Returns null if underflows. */
function prevSlot(t) {
    const [h, m] = t.split(':').map(Number);
    const total  = h * 60 + m - 5;
    if (total < 0) return null;
    return `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
}

// ── Alpine component ──────────────────────────────────────────────────────────
document.addEventListener('alpine:init', () => {
    Alpine.data('today', () => ({
        // Controls
        mode:        'iv',   // iv | change
        dates:       [],
        date:        null,
        expirations: [],     // [{dte, expiry, label}]
        selectedDte: null,

        // Data
        loading: false,
        error:   null,
        strikes: [],         // [5900, 6000, …]  (integers)
        rows:    [],         // [{time:'09:35', data:{'5900':0.1825, …}}, …]
        prev:    {},         // {'5900':0.1810, …}  prev-day 16:00

        ALL_TIMES: allTimeSlots(),

        // ── Init ─────────────────────────────────────────────────────────────
        async init() {
            await this.loadDates();
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
                const res        = await fetch(`/api/today/expirations?date=${this.date}`);
                this.expirations = await res.json();

                // Default: expiration closest to 30 DTE
                if (this.expirations.length) {
                    const closest = this.expirations.reduce((best, e) =>
                        Math.abs(e.dte - 30) < Math.abs(best.dte - 30) ? e : best
                    );
                    this.selectedDte = closest.dte;
                    await this.loadGrid();
                }
            } catch(e) { this.error = 'Failed to load expirations'; }
        },

        async onExpirationChange() {
            await this.loadGrid();
        },

        async loadGrid() {
            if (!this.date || this.selectedDte == null) return;
            this.loading = true;
            this.error   = null;
            try {
                const res    = await fetch(`/api/today/iv_grid?date=${this.date}&dte=${this.selectedDte}`);
                const data   = await res.json();
                this.strikes = data.strikes;
                this.rows    = data.rows;
                this.prev    = data.prev;
                this.renderGrid();
            } catch(e) {
                this.error = 'Failed to load grid';
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

            // Build lookup: time → {strikeStr → iv}
            const dataMap = {};
            for (const row of this.rows) dataMap[row.time] = row.data;

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
                const prevTime = prevSlot(t);
                const prevData = (prevTime && dataMap[prevTime]) ? dataMap[prevTime]
                               : (t === '09:35') ? prev
                               : {};

                for (const s of strikes) {
                    const sk  = String(s);
                    const iv  = rowData[sk];

                    let bg = '#1e1e1e', fg = '#555', display = '', title = '';

                    if (iv != null) {
                        if (mode === 'iv') {
                            bg      = '#252525';
                            fg      = '#ccc';
                            display = (iv * 100).toFixed(2);
                            title   = `${t}  K=${s}\nIV: ${display}%`;
                        } else {
                            const prevIv = prevData[sk];
                            if (prevIv != null) {
                                const chg_pp = (iv - prevIv) * 100;
                                bg      = changeColor(chg_pp, 1.0);
                                fg      = contrastColor_T(bg);
                                const sign = chg_pp >= 0 ? '+' : '';
                                display = sign + chg_pp.toFixed(2);
                                title   = `${t}  K=${s}\nChg: ${display} pp\nIV now: ${(iv*100).toFixed(2)}%\nIV prev: ${(prevIv*100).toFixed(2)}%`;
                            } else {
                                // Have current IV but no prior — show IV with dim style
                                bg      = '#252525';
                                fg      = '#555';
                                display = (iv * 100).toFixed(2);
                                title   = `${t}  K=${s}\nIV: ${display}% (no prior slice)`;
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

        // ── Helpers ───────────────────────────────────────────────────────────
        get dateDisplay() { return this.date ?? '—'; },
        get selectedLabel() {
            const e = this.expirations.find(x => x.dte === this.selectedDte);
            return e ? e.label : '—';
        },
    }));
});
