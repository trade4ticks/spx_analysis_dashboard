/**
 * oi_signals.js — OI Signal Dashboard
 * Inline decile sparklines with today's position marker.
 */
'use strict';

document.addEventListener('alpine:init', () => {
  Alpine.data('oiSignals', () => ({
    loading:    true,
    error:      null,
    metrics:    [],
    outcome:    '',
    tickers:    [],
    sortCol:    '',      // metric name to sort by
    sortDir:    'asc',   // 'asc' = D1 first (bullish extremes first)
    _charts:    {},

    // Co-occurrence
    coocMetric:  '',
    coocDecile:  1,
    coocData:    null,
    coocLoading: false,

    async init() {
      await this.loadData();
    },

    async loadData() {
      this.loading = true;
      this.error   = null;
      try {
        const r = await fetch('/api/oi-signals/data');
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        if (data.error) throw new Error(data.error);
        this.metrics  = data.metrics || [];
        this.outcome  = data.outcome || '';
        this.tickers  = data.tickers || [];
        if (!this.sortCol && this.metrics.length) this.sortCol = this.metrics[0];
        if (!this.coocMetric && this.metrics.length) this.coocMetric = this.metrics[0];

        await this.$nextTick();
        await this.$nextTick();
        setTimeout(() => this.renderAllSparklines(), 100);
      } catch (e) {
        this.error = e.message;
      } finally {
        this.loading = false;
      }
    },

    get sortedTickers() {
      if (!this.sortCol) return this.tickers;
      if (this.sortCol === '_ticker') {
        return [...this.tickers].sort((a, b) =>
          this.sortDir === 'asc'
            ? a.ticker.localeCompare(b.ticker)
            : b.ticker.localeCompare(a.ticker));
      }
      return [...this.tickers].sort((a, b) => {
        const da = a.metrics?.[this.sortCol]?.today_decile ?? 5;
        const db = b.metrics?.[this.sortCol]?.today_decile ?? 5;
        return this.sortDir === 'asc' ? da - db : db - da;
      });
    },

    toggleSort(col) {
      if (this.sortCol === col) {
        this.sortDir = this.sortDir === 'asc' ? 'desc' : 'asc';
      } else {
        this.sortCol = col;
        this.sortDir = 'asc';
      }
    },

    sortArrow(metric) {
      if (this.sortCol !== metric) return '';
      return this.sortDir === 'asc' ? ' ▲' : ' ▼';
    },

    metricShort(m) {
      return m.replace('zscore_oi_', 'z_').replace('_3m', '').replace('d1_oi_', 'd1_')
              .replace('weighted_strike_all_div_spot', 'wt_k/s')
              .replace('above_below_ratio', 'ab_ratio')
              .replace('_change', '_chg');
    },

    decileClass(d) {
      if (d == null) return '';
      if (d === 1) return 'decile-d1';
      if (d === 2) return 'decile-d2';
      if (d === 9) return 'decile-d9';
      if (d === 10) return 'decile-d10';
      return '';
    },

    // Get the avg return for today's decile
    todayAvgRet(md) {
      if (!md || !md.today_decile || !md.deciles) return null;
      const d = md.deciles.find(b => b && b.bucket === md.today_decile);
      return d ? d.avg_ret : null;
    },

    renderAllSparklines() {
      // Destroy existing
      Object.values(this._charts).forEach(c => c.destroy());
      this._charts = {};

      for (const t of this.tickers) {
        for (const m of this.metrics) {
          const md = t.metrics?.[m];
          if (!md || md.error || !md.deciles) continue;
          const key = `spark-${t.ticker}-${m}`;
          const el = document.getElementById(key);
          if (!el) continue;

          const deciles = md.deciles.filter(d => d != null);
          const avgs = deciles.map(d => d.avg_ret);
          const todayD = md.today_decile;

          const bgColors = deciles.map((d, i) => {
            if (d.bucket === todayD) return '#ffffff';
            return d.avg_ret >= 0 ? 'rgba(52,152,219,0.6)' : 'rgba(232,67,147,0.6)';
          });
          const borderColors = deciles.map(d =>
            d.bucket === todayD ? '#ffffff' : 'transparent');

          this._charts[key] = new Chart(el, {
            type: 'bar',
            data: {
              labels: deciles.map(d => d.bucket),
              datasets: [{
                data: avgs,
                backgroundColor: bgColors,
                borderColor: borderColors,
                borderWidth: deciles.map(d => d.bucket === todayD ? 2 : 0),
              }],
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              animation: false,
              plugins: {
                legend: { display: false },
                tooltip: {
                  backgroundColor: 'rgba(20,20,20,0.95)',
                  titleColor: '#999', bodyColor: '#ddd',
                  borderColor: '#444', borderWidth: 1,
                  callbacks: {
                    title: (items) => `D${items[0].label}`,
                    label: (ctx) => {
                      const d = deciles[ctx.dataIndex];
                      return [
                        `Avg: ${d.avg_ret.toFixed(3)}%`,
                        `WR: ${d.win_rate.toFixed(0)}%`,
                        `n: ${d.n}`,
                      ];
                    },
                  },
                },
              },
              scales: {
                x: { display: false },
                y: { display: false },
              },
            },
          });
        }
      }
    },

    // Co-occurrence
    async loadCooccurrence() {
      this.coocLoading = true;
      try {
        const r = await fetch(
          `/api/oi-signals/cooccurrence?metric=${this.coocMetric}&decile=${this.coocDecile}`);
        if (r.ok) this.coocData = await r.json();
      } catch (_) {}
      this.coocLoading = false;
    },

    coocColor(pct) {
      if (pct >= 80) return 'color:#e84393';     // high overlap = warning
      if (pct >= 50) return 'color:#f39c12';
      if (pct >= 30) return 'color:#ccc';
      return 'color:#3498db';                     // low overlap = good
    },
  }));
});
