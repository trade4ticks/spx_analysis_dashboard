'use strict';

// Shared signal-thumbnail helpers.
//
// Single source of truth for the heatmap-cell color/opacity scales and the
// per-signal SVG thumbnail. Both the Factor Analysis page (oi_analysis.js)
// and the Signals page (oi_signals.js) call into this object so a given
// avg-ret looks identical wherever it renders — there is no parallel copy
// to drift against.
//
// Loaded BEFORE the page-specific JS in both templates. Both pages' Alpine
// components delegate `this.cellColor` / `this.cellOpacity` /
// `this.signalThumbnailSVG` here, so legacy call sites keep working
// untouched.
window.SignalThumb = {

  // Per-cell color on a fixed ±3% divergent scale. Same canonical hex
  // anchors as .pos / .neg CSS across the project — no per-signal or
  // per-global max_abs calibration. A cell's color means the same return
  // in every thumbnail and every grid cell.
  cellColor(avgRet) {
    if (avgRet === null || avgRet === undefined) return 'rgb(60,60,60)';
    const r = Math.max(-0.03, Math.min(0.03, +avgRet));
    const t = r / 0.03;   // -1 → +1
    // Neutral rgb(60,60,60). Blue end #3498db (52,152,219). Pink end #e84393 (232,67,147).
    let R, G, B;
    if (t > 0) {
      R = Math.round(60 + (52  - 60) * t);
      G = Math.round(60 + (152 - 60) * t);
      B = Math.round(60 + (219 - 60) * t);
    } else if (t < 0) {
      const u = -t;
      R = Math.round(60 + (232 - 60) * u);
      G = Math.round(60 + (67  - 60) * u);
      B = Math.round(60 + (147 - 60) * u);
    } else {
      return 'rgb(60,60,60)';
    }
    return `rgb(${R},${G},${B})`;
  },

  // Linear small-n dim: 0.35 below n=10, 1.0 at n>=100, smooth between.
  // Honesty cue paired with cellColor — a dark color on a thin sample
  // reads as muted, not authoritative.
  cellOpacity(n) {
    if (n === null || n === undefined || n < 10) return 0.35;
    if (n >= 100) return 1.0;
    return 0.35 + 0.65 * (n - 10) / 90;
  },

  // Build the signal thumbnail SVG as a string for x-html injection.
  // Alpine <template x-for> can't be used inside an <svg> element —
  // the browser parses <template> in HTML mode and cloned <rect>/<line>
  // children land in the HTML namespace (no paint). Stringifying the
  // SVG and injecting via x-html lets innerHTML's inline-SVG parser
  // place every node in the SVG namespace.
  //
  // Width/height are overridable so callers can render at column-header
  // size (e.g. 80) without rebuilding the function — the viewBox stays
  // 0..20 so the scale is intrinsic.
  thumbnailSVG(sig, opts) {
    const width    = (opts && opts.width)  || 110;
    const height   = (opts && opts.height) || width;
    // Default to clickable cursor — matches the FA Saved Signals card
    // affordance (the wrapping div carries the @click). Callers that
    // want a plain cursor (e.g. the read-only column header on the
    // Signals page grid) pass { clickable: false }.
    const cursor   = (opts && opts.clickable === false) ? 'default' : 'pointer';
    // Frame stroke is opt-up only — caller passes { frameStrong: true }
    // to read clearly against a dark pane (Signals page firing grid).
    // Default keeps the FA Saved Signals cards' existing appearance
    // exactly as-is; opt-in is the only way to bump it.
    const fStroke  = (opts && opts.frameStrong) ? 'rgba(255,255,255,.55)'
                                                 : 'rgba(255,255,255,.18)';
    const fWidth   = (opts && opts.frameStrong) ? '0.16' : '0.08';
    const n        = (sig && sig.n_bins) || 0;
    const step     = n > 0 ? 20 / n : 20;
    const cells    = (sig && sig.per_cell_stats) || [];

    const parts = [
      '<svg width="' + width + '" height="' + height + '" viewBox="0 0 20 20"'
        + ' style="display:block;background:#1a1a1a;cursor:' + cursor + '">',
      // Frame
      '<rect x="0" y="0" width="20" height="20" fill="none"'
        + ' stroke="' + fStroke + '" stroke-width="' + fWidth + '"/>',
    ];
    // Gridlines at the signal's own n_bins resolution.
    for (let i = 1; i < n; i++) {
      const pos = (i * step).toFixed(4);
      parts.push(
        '<line x1="' + pos + '" y1="0" x2="' + pos + '" y2="20"'
          + ' stroke="rgba(255,255,255,.08)" stroke-width="0.05"/>',
        '<line x1="0" y1="' + pos + '" x2="20" y2="' + pos + '"'
          + ' stroke="rgba(255,255,255,.08)" stroke-width="0.05"/>'
      );
    }
    // Selected cells. No Y flip — matches the main 2D heatmap orientation
    // (y=0 at top; high iy at bottom).
    for (const cell of cells) {
      const x       = (cell.ix * step).toFixed(4);
      const y       = (cell.iy * step).toFixed(4);
      const w       = step.toFixed(4);
      const fill    = window.SignalThumb.cellColor(cell.avg_ret);
      const opacity = window.SignalThumb.cellOpacity(cell.n).toFixed(3);
      parts.push(
        '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + w + '"'
          + ' fill="' + fill + '" fill-opacity="' + opacity + '"'
          + ' stroke="rgba(255,255,255,.30)" stroke-width="0.06"/>'
      );
    }
    parts.push('</svg>');
    return parts.join('');
  },
};
