"""Factor Trades' sector chips, DRIVEN BY CLICKING, in headless Edge.

The zone payloads are the REAL /zone handler's, over check_ft_sectors'
fixture (eleven tickers across seven buckets plus one the map does not know,
days where the daily cap binds, both windows). The page JS and template are
the shipped files; fetch is stubbed in the browser. Real time over the
DevTools protocol (scripts/edge_cdp.py), no requestAnimationFrame shim.

What it asks, as a person would:
  + every bucket has a chip, plus Unassigned; the counts add up to the window
  + the peak-sector line and the map check are on screen
  + with every chip on, the equity pane is the old renderer's
  + "none" empties the stat bar; one chip gives that sector's numbers in the
    stat bar AND every pane (exit reasons, P&L distribution, price bins,
    ticker breakdown, annual P&L), a coloured line against the grey total,
    a legend that names exactly those lines, and the "not additive" notes
  + the grey line is the unfiltered curve; annual P&L sums to the sector line
  + five chips draw one combined line
  + the peak line follows the chips (one chip = that sector's own peak),
    with its share of the whole book, and returns with "all"
  + the Configure-sweep chips wrap and never clip
  + a locked card follows the same chips
  + the filter survives TRAIN -> TEST, and "all" restores the old page
And the console stays clean.

No Edge: SKIP, not PASS.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import edge_cdp                                           # noqa: E402

EXIT_SKIPPED = 3
EDGE = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        "/usr/bin/microsoft-edge", "/usr/bin/chromium", "/usr/bin/google-chrome"]


def payloads() -> dict:
    import asyncio
    import check_ft_sectors as cs
    import app.routers.factor_trades as ft
    import app.routers.oi_analysis as oia
    from app import factor_sectors as fs
    h = cs._harness()
    rows = cs.fixture_rows(h)
    zones = cs.zones(h, ft, rows)
    out = {"sectors": fs.payload()}
    for win in ("train", "test"):
        oia._TT_CUTOFF_CACHED = None
        r = asyncio.run(ft.run(req=ft.RunReq(primary_metric="a", secondary_metric="b", n_bins=20,
                                             rule_keys=["fixed_stop__5"], max_strike=1000,
                                             window=win), pool=h.FakePool()))
        r = json.loads(json.dumps(r, default=str))
        g = [[None] * 20 for _ in range(20)]
        g[2][1] = {"n": 120, "avg_ret": 0.004, "test_n": 40, "test_avg": 0.001}
        r["grid"] = g
        out["run " + win] = r
        out["zone " + win] = zones[win]
    smap = fs.sector_map()
    exp = {}
    for win in ("train", "test"):
        w = zones[win]["window_trades"]
        per = {}
        for t in w:
            s = smap.get(t["ticker"], fs.UNASSIGNED)
            per[s] = per.get(s, 0) + 1
        exp[win] = {"n": len(w), "per": per,
                    "crypto_tickers": sorted({t["ticker"] for t in w if smap.get(t["ticker"]) == "Crypto"})}
    out["exp"] = exp
    return out


DRIVER = r"""
<script>
(function () {
  const errs = [];
  window.addEventListener('error', e => errs.push('onerror: ' + e.message));
  const ce = console.error.bind(console);
  console.error = (...a) => { errs.push('console.error: ' + a.map(String).join(' ')); ce(...a); };
  const LIVE = PAY.sectors.buckets.flatMap(b => b.tickers).filter(t => t !== 'BAC').concat(['ZZZZ']);
  window.fetch = async (url, opts) => {
    const u = String(url);
    const body = opts && opts.body ? JSON.parse(opts.body) : null;
    const ok = b => ({ ok: true, status: 200, json: async () => b, text: async () => JSON.stringify(b) });
    if (u.includes('/factor-analysis/columns')) return ok({ features: ['a', 'b'], outcomes: [], feature_families: [] });
    if (u.includes('/factor-analysis/tickers')) return ok(LIVE);
    if (u.includes('/tt-cutoff')) return ok({ cutoff_date: '2023-01-02' });
    if (u.includes('/factor-trades/sectors')) return ok(PAY.sectors);
    if (u.includes('/factor-trades/rules')) return ok({ groups: [] });
    if (u.includes('/factor-trades/signals')) return ok({ signals: [], cutoff_date: '2023-01-02', max_selectable: 31 });
    const w = (body && body.window) || 'train';
    if (u.endsWith('/factor-trades/run')) return ok(PAY['run ' + w]);
    if (u.endsWith('/factor-trades/zone')) return ok(PAY['zone ' + w]);
    return { ok: false, status: 404, json: async () => ({}), text: async () => 'no stub ' + u };
  };
  const out = [];
  const step = s => { window.__gateStep = s; const h = document.documentElement;
                      h.setAttribute('data-progress', s);
                      h.setAttribute('data-trail', (h.getAttribute('data-trail') || '') + ' > ' + s); };
  const eq = (name, got, want) => out.push(`${JSON.stringify(got) === JSON.stringify(want) ? 'ok' : 'FAIL'}|${name}|${JSON.stringify(got)}|${JSON.stringify(want)}`);
  const near = (name, got, want, tol) => out.push(`${(got != null && want != null && Math.abs(got - want) <= tol) ? 'ok' : 'FAIL'}|${name}|${JSON.stringify(got)}|${JSON.stringify(want)}`);
  const wait = ms => new Promise(r => setTimeout(r, ms));
  async function until(what, fn, ms = 8000) {
    const t0 = Date.now();
    while (Date.now() - t0 < ms) { try { if (fn()) return; } catch (e) {} await wait(50); }
    throw new Error('timed out waiting for: ' + what);
  }
  const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
  const txt = el => (el ? el.textContent.trim().replace(/\s+/g, ' ') : null);
  const vis = el => !!(el && el.offsetParent !== null);
  const btn = t => $$('button').find(b => txt(b) === t);
  const C = () => Alpine.$data(document.body);
  const chips = () => $$('.ft-sectors .chip').filter(vis);
  const chip = name => chips().find(c => txt(c).replace(/ \d+$/, '') === name);
  const act = t => $$('.ft-chip-act').find(a => txt(a) === t);
  const rowBox = (rowLabel, box) => {
    const row = $$('.ft-statrow').find(r => txt(r.querySelector('.ft-statlbl')).startsWith(rowLabel));
    if (!row) return null;
    const it = [...row.querySelectorAll('.stat-item')].find(i => txt(i.querySelector('.stat-label')) === box);
    return it ? txt(it.querySelector('.stat-value')) : null;
  };
  const chart = id => C()._charts[id];
  const labels = id => chart(id) ? chart(id).data.datasets.map(d => d.label) : null;
  const lastY = (id, label) => { const d = chart(id).data.datasets.find(x => x.label === label); return d.data[d.data.length - 1].y; };
  const sum = a => a.reduce((p, x) => p + (+x || 0), 0);
  const legend = () => $$('.ft-legend > span').filter(vis).map(s => txt(s));
  const peak = () => txt($$('.ft-sector-lines b')[0]);
  const fmtN = n => n.toLocaleString();
  const E = PAY.exp;

  async function run() {
    step('load');
    await until('the Run button', () => btn('Run'));
    await until('the sector map', () => C().sectorIdx);
    btn('Run').click();
    step('run');
    await until('a selectable cell', () => $$('.hm-cell-selectable').length);
    $$('.hm-cell-selectable').find(c => (c.getAttribute('title') || '').length).click();
    step('zone');
    await until('the zone and its chips', () => C().zoneData && chips().length);
    await wait(300);

    // ── every sector on ────────────────────────────────────────────────
    const names = PAY.sectors.buckets.map(b => b.name).concat(['Unassigned']);
    eq('a chip per bucket, plus Unassigned for the unmapped ticker', chips().map(c => txt(c).replace(/ \d+$/, '')), names);
    eq('every chip starts on', chips().every(c => c.classList.contains('on')), true);
    eq('chip counts add up to the window', sum(chips().map(c => +txt(c).match(/(\d+)$/)[1])), E.train.n);
    eq('the stat bar shows every trade', rowBox('Current', 'N'), fmtN(E.train.n));
    eq('the equity pane is the old renderer', labels('ft-equity'), ['Equity ($)', 'Drawdown ($)']);
    eq('no legend with every sector on', legend(), []);
    const pk = peak();
    const PEAK = /^\$[\d.,]+k? (.+), \d{4}-\d{2}-\d{2} — \d+% of \$[\d.,]+k? deployed$/;
    eq('the peak sector line names a sector, a date and its share of the book', PEAK.test(pk || ''), true);
    const scope = () => txt($$('.ft-sector-lines > div').find(d => /^Peak sector exposure/.test(txt(d))));
    eq('and says it covers every sector', /^Peak sector exposure \(every sector, train window\)/.test(scope() || ''), true);
    const mapNote = txt($$('.ft-sector-lines > div').find(d => /^Sector map/.test(txt(d))));
    eq('the map check names the mapped ticker missing from daily_features and the unmapped one',
       !!mapNote && mapNote.includes('not in daily_features: BAC') && mapNote.includes('no bucket (shown as Unassigned): ZZZZ'), true);
    const totalAll = lastY('ft-equity', 'Equity ($)');

    // ── the exit-reasons pane: one label column, three aligned charts ────
    step('exit pane');
    const EX = ['ft-reasons', 'ft-hold', 'ft-exitret'];
    const exRows = () => C().zoneView().exit_reasons;
    const exAligned = () => {
      // Each label's centre against each chart's bar centre, in page pixels.
      const ls = [...$('#ft-exit-labels').children].map(d => { const r = d.getBoundingClientRect(); return r.top + r.height / 2; });
      let worst = 0;
      for (const id of EX) {
        const ch = chart(id), top = ch.canvas.getBoundingClientRect().top;
        ch.getDatasetMeta(0).data.forEach((b, i) => { worst = Math.max(worst, Math.abs(top + b.y - ls[i])); });
      }
      return worst;
    };
    eq('exit pane: three charts drawn', EX.map(id => !!chart(id)), [true, true, true]);
    eq('the rule names are drawn ONCE, in the label column, in row order',
       $$('#ft-exit-labels > .ft-exit-label').map(txt), exRows().map(r => r.label));
    eq('no chart draws its own y labels', EX.map(id => chart(id).options.scales.y.display), [false, false, false]);
    eq('each chart keeps its own x axis', EX.map(id => chart(id).options.scales.x.display !== false && !!chart(id).scales.x), [true, true, true]);
    const labR = $('#ft-exit-labels').getBoundingClientRect().right;
    eq('the label column sits left of every chart', EX.every(id => chart(id).canvas.getBoundingClientRect().left >= labR - 0.5), true);
    near('every label is centred on its row in all three charts (px)', exAligned(), 0, 1.5);
    near('avg return is the rows\' avg_ret, in %', sum(chart('ft-exitret').data.datasets[0].data),
         sum(exRows().map(r => r.avg_ret * 100)), 1e-3);
    const zx = chart('ft-exitret').scales.x;
    eq('avg return: zero is inside its axis', zx.min <= 0 && zx.max >= 0, true);
    // (bases sit within half a border width of their anchor: Chart.js offsets them by it)
    eq('the fixture has rules on both sides of zero, so both directions are drawn',
       [exRows().some(r => r.avg_ret > 0), exRows().some(r => r.avg_ret < 0)], [true, true]);
    const z0 = zx.getPixelForValue(0);
    eq('avg return: every bar starts at the zero line and runs its own sign\'s way',
       chart('ft-exitret').getDatasetMeta(0).data.every((b, i) => Math.abs(b.base - z0) <= 1.01
         && Math.sign(Math.round((b.x - z0) * 100)) === Math.sign(Math.round(exRows()[i].avg_ret * 1e6)) ), true);
    eq('the zero line is drawn brighter than the grid',
       chart('ft-exitret').config.options.scales.x.grid.color({ tick: { value: 0 } }) !== chart('ft-exitret').config.options.scales.x.grid.color({ tick: { value: 1 } }), true);
    eq('share and hold stay left-anchored',
       ['ft-reasons', 'ft-hold'].map(id => { const ch = chart(id); return ch.getDatasetMeta(0).data.every(b => Math.abs(b.base - ch.chartArea.left) <= 1.01); }),
       [true, true]);
    const tipOf = id => [].concat(chart(id).options.plugins.tooltip.callbacks.label({ dataIndex: 0 })).join(' | ');
    const r0 = exRows()[0];
    eq('every chart\'s hover gives all three values and the trade count',
       EX.map(id => { const t = tipOf(id); return /share of exits/.test(t) && /avg hold/.test(t) && /avg return/.test(t) && t.includes('n = ' + r0.n.toLocaleString() + ' trades'); }),
       [true, true, true]);
    eq('no value labels on the bars (no datalabels plugin)',
       EX.map(id => !!(chart(id).options.plugins.datalabels) || (chart(id).config.plugins || []).some(p => p.id === 'datalabels')), [false, false, false]);
    // Full screen carries the label column, still aligned, and returns it.
    $$('.chart-title').find(t => /^Exit reasons/.test(txt(t))).querySelector('.ft-fs').click();
    await until('the exit pane full screen', () => $('#ft-ov-body #ft-exit-labels') && $('#ft-ov-body #ft-exitret'));
    await wait(300);
    near('full screen: the labels stay on their rows (px)', exAligned(), 0, 1.5);
    btn('✕ Close').click();
    await until('the pane back home', () => $('.ft-exit #ft-exit-labels') && $('.ft-exit #ft-exitret'));
    await wait(300);
    near('back from full screen: still aligned (px)', exAligned(), 0, 1.5);

    // ── how the chips render ───────────────────────────────────────────
    step('chip rendering');
    const cs = el => getComputedStyle(el);
    eq('no chip is clipped: its text fits its box', chips().filter(c => c.scrollWidth > c.clientWidth + 1).map(txt), []);
    eq('no chip ellipsis', chips().filter(c => cs(c).textOverflow === 'ellipsis').map(txt), []);
    eq('every chip shows its full name and count', chips().every(c => /^.+ \d+$/.test(txt(c))), true);
    eq('the chips use the page font, not monospace', [...new Set(chips().map(c => cs(c).fontFamily))],
       [cs(document.body).fontFamily]);
    eq('padding is even left and right', chips().every(c => cs(c).paddingLeft === cs(c).paddingRight), true);
    const accent = (() => { const d = document.createElement('div'); d.style.color = 'var(--accent)';
                            document.body.appendChild(d); const v = cs(d).color; d.remove(); return v; })();
    eq('an on chip wears the theme --accent (the site chip), not a page-local blue',
       [cs(chip('Crypto')).borderTopColor, cs(chip('Crypto')).color], [accent, accent]);
    eq('--accent is the site blue', accent, 'rgb(52, 152, 219)');
    const row = $('.ft-sectors'), rr = row.getBoundingClientRect();
    eq('every chip sits inside the row', chips().every(c => c.getBoundingClientRect().right <= rr.right + 0.5), true);
    row.style.width = '420px'; await wait(100);
    const tops = [...new Set(chips().map(c => Math.round(c.getBoundingClientRect().top)))];
    eq('in a narrow column the chips wrap onto more rows, still unclipped',
       [tops.length > 2, chips().filter(c => c.scrollWidth > c.clientWidth + 1).length,
        chips().every(c => c.getBoundingClientRect().right <= row.getBoundingClientRect().right + 0.5)], [true, 0, true]);
    row.style.width = ''; await wait(50);

    // ── none, then one ─────────────────────────────────────────────────
    step('none');
    act('none').click();
    await until('an empty stat bar', () => rowBox('Current', 'N') === '0');
    eq('"none" turns every chip off', chips().some(c => c.classList.contains('on')), false);
    eq('with nothing selected only the grey total is drawn', labels('ft-equity'), ['all sectors', 'all sectors drawdown']);
    step('crypto');
    chip('Crypto').click();
    const nc = E.train.per['Crypto'];
    await until('the crypto stat bar', () => rowBox('Current', 'N') === fmtN(nc));
    eq('one chip: the stat bar is that sector\'s', rowBox('Current', 'N'), fmtN(nc));
    eq('its Tickers box counts its tickers', rowBox('Current', 'Tickers'), String(E.train.crypto_tickers.length));
    eq('the equity pane: grey total plus the sector', labels('ft-equity'),
       ['all sectors', 'all sectors drawdown', 'Crypto', 'Crypto drawdown']);
    eq('the legend names exactly the drawn lines', legend(), ['Crypto', 'all sectors']);
    eq('the chip wears its line colour', vis(chip('Crypto').querySelector('i')), true);
    near('the grey line is the unfiltered curve', lastY('ft-equity', 'all sectors'), totalAll, 0.011);
    eq('the "subset\'s own stats" note is shown',
       ($$('.ft-sector-lines > div').map(txt).find(t => /^Filtered to/.test(t || '')) || '').startsWith('Filtered to 1 of 15 sectors'), true);
    eq('the drawdown note is shown', $$('.ft-note').filter(vis).some(n => /do not: each is measured from that line's own peak/.test(txt(n))), true);
    eq('exit reasons: the shares are the sector\'s (sum 100%)', Math.round(sum(chart('ft-reasons').data.datasets[0].data)), 100);
    eq('P&L distribution counts the sector\'s trades', sum(chart('ft-pnldist').data.datasets[0].data), nc);
    eq('ticker breakdown: only the sector\'s tickers', labels('ft-bubble-edited').slice().sort(), E.train.crypto_tickers);
    const pbN = (C().zoneView().price_bins || []).reduce((p, b) => p + b.n, 0);
    eq('price bins count the sector\'s trades', pbN, nc);
    near('annual P&L adds up to the sector\'s line', sum(chart('ft-yearly').data.datasets[0].data),
         lastY('ft-equity', 'Crypto'), 0.05);
    eq('one chip: the peak line is the sector own peak', [(peak() || '').match(PEAK)?.[1], /^Peak sector exposure \(Crypto,/.test(scope() || '')], ['Crypto', true]);

    // ── five ───────────────────────────────────────────────────────────
    step('five');
    for (const n of ['Information Technology', 'Financials', 'Health Care', 'Unassigned']) { chip(n).click(); await wait(80); }
    await until('five selected', () => (C().sectorKeep || []).length === 5);
    await wait(200);
    eq('five chips draw one combined line', labels('ft-equity'), ['all sectors', 'all sectors drawdown', '5 sectors', '5 sectors drawdown']);
    eq('and the legend says so', legend(), ['5 sectors combined', 'all sectors']);
    const n5 = ['Crypto', 'Information Technology', 'Financials', 'Health Care', 'Unassigned'].reduce((p, s) => p + (E.train.per[s] || 0), 0);
    eq('the stat bar counts the five', rowBox('Current', 'N'), fmtN(n5));

    // ── a locked card follows ─────────────────────────────────────────
    step('lock');
    act('all').click(); await wait(100);
    btn('+ Save to strip').click();
    await until('a saved card', () => $$('.ft-lock').some(l => { try { return !!Alpine.$data(l).c.saved; } catch (e) { return false; } }));
    $$('.ft-lock').find(l => { try { return !!Alpine.$data(l).c.saved; } catch (e) { return false; } }).click();
    await until('a Locked row', () => rowBox('Locked', 'N'));
    act('none').click(); await wait(80);
    chip('Crypto').click();
    await until('the locked row filtered', () => rowBox('Locked', 'N') === fmtN(nc));
    eq('the locked row follows the chips', rowBox('Locked', 'N'), fmtN(nc));
    eq('the current row too', rowBox('Current', 'N') || rowBox('Edited', 'N'), fmtN(nc));
    eq('the locked ticker breakdown follows', labels('ft-bubble-locked').slice().sort(), E.train.crypto_tickers);

    // ── TRAIN -> TEST ──────────────────────────────────────────────────
    step('test');
    $$('span').find(s => txt(s) === 'TEST').click();
    await until('the test zone', () => C().window === 'test' && C().zoneData && C().zoneData.window === 'test');
    await wait(400);
    eq('the filter survives the window switch', C().sectorKeep, ['Crypto']);
    const rowN = rowBox('Edited', 'N') || rowBox('Current', 'N');
    eq('and applies to the test window', rowN, fmtN(E.test.per['Crypto']));

    // ── all ────────────────────────────────────────────────────────────
    step('all');
    act('all').click();
    await until('every chip on', () => C().sectorKeep === null);
    await wait(200);
    eq('"all" restores the old equity renderer', labels('ft-equity'), ['Equity ($)', 'Drawdown ($)']);
    eq('and the whole window in the stat bar', rowBox('Edited', 'N') || rowBox('Current', 'N'), fmtN(E.test.n));
    eq('"all" puts the peak line back to every sector', /^Peak sector exposure \(every sector, test window\)/.test(scope() || ''), true);
    eq('and drops the filtered note', $$('.ft-sector-lines > div').filter(vis).some(d => /^Filtered to/.test(txt(d))), false);

    // ── the Configure-sweep chips, and the page's one blue ──────────────
    step('sweep chips');
    const box = document.createElement('div');
    box.className = 'ft-chips'; box.style.width = '300px';
    for (const l of ['max_days 20', 'trailing_stop 7.5% off peak', 'fixed_target 12%', 'x']) {
      const c = document.createElement('span'); c.className = 'chip'; c.textContent = l; box.appendChild(c);
    }
    box.children[1].classList.add('on');
    $('.ft-sectors').after(box); await wait(50);
    const sc = [...box.children];
    eq('a sweep chip never clips its label', sc.filter(c => c.scrollWidth > c.clientWidth + 1).map(txt), []);
    eq('sweep chips wrap inside their row', [new Set(sc.map(c => Math.round(c.getBoundingClientRect().top))).size > 1,
       sc.every(c => c.getBoundingClientRect().right <= box.getBoundingClientRect().right + 0.5)], [true, true]);
    eq('a sweep chip on wears --accent, in the page font',
       [cs(sc[1]).borderTopColor, cs(sc[1]).fontFamily], [accent, cs(document.body).fontFamily]);
    box.remove();
    const sweepMarkup = [...document.querySelectorAll('template')].some(tp => /class="ft-chip"/.test(tp.innerHTML));
    eq('the sweep chip markup uses .chip, not the removed .ft-chip', sweepMarkup, false);
    eq('the Run button is the site primary (--accent)', cs(btn('Run')).backgroundColor, accent);
    const save = btn('+ Save to strip');
    eq('"+ Save to strip" is too', save ? cs(save).backgroundColor : null, accent);
    const old = [...document.querySelectorAll('*')].filter(el => {
      const c = cs(el);
      return [c.backgroundColor, c.borderTopColor, c.color].some(v => v === 'rgb(42, 90, 130)' || v === 'rgb(58, 122, 176)');
    }).map(el => el.tagName + '.' + el.className);
    eq('nothing on the page paints the old #2a5a82 / #3a7ab0', old, []);
    // :hover cannot be produced from script, so read the rules themselves:
    // the page's primaries must hover exactly as theme.css's .btn.primary.
    const rules = [...document.styleSheets].flatMap(sh => { try { return [...sh.cssRules]; } catch (e) { return []; } });
    const hoverBg = sel => { const r = rules.find(x => x.selectorText === sel); return r ? r.style.backgroundColor : null; };
    const site = hoverBg('.btn.primary:hover');
    eq('the site primary hover is #2980b9', site, 'rgb(41, 128, 185)');
    eq('Run / Run suite / Done and Save hover as the site primary',
       [hoverBg('.ft-btn.primary:hover:not(:disabled)'), hoverBg('.ft-savebtn:hover:not(:disabled)')], [site, site]);

    report();
  }
  function report() {
    const pre = document.createElement('pre');
    pre.id = 'report';
    pre.textContent = out.join('\n') + '\nerrs|' + (errs.length ? errs.join(' ;; ') : 'clean');
    document.body.appendChild(pre);
  }
  document.addEventListener('alpine:initialized', () => setTimeout(() => run().catch(err => {
    out.push(`FAIL|the driver stopped at ${window.__gateStep}|${String(err && err.message || err)}|nothing`);
    report();
  }), 50));
})();
</script>
"""


def build_page(pay: dict) -> str:
    import jinja2
    from app.assets import asset

    class _URL:
        hostname = "localhost"; scheme = "http"; path = "/"
        def __str__(self): return "http://localhost/"

    class _Req:
        url = _URL(); headers = {}; query_params = {}; scope = {"type": "http"}

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(ROOT / "templates")),
                             autoescape=True, keep_trailing_newline=True)
    env.globals["asset"] = asset
    env.globals["live_port"] = 8001
    html = env.get_template("factor_trades.html").render(request=_Req())
    html = html.replace('<script defer src="https://', '<script defer crossorigin="anonymous" src="https://')
    html = html.replace('<script src="https://', '<script crossorigin="anonymous" src="https://')
    html = re.sub(r'<link rel="stylesheet" href=[^>]*css/([a-z_]+\.css)[^>]*>',
                  lambda m: "<style>\n" + (ROOT / "static/css" / m.group(1)).read_text(encoding="utf-8") + "\n</style>",
                  html)
    stub = f"<script>const PAY = {json.dumps(pay)};</script>" + DRIVER
    first = [True]

    def inline(m):
        pre = stub if first[0] else ""
        first[0] = False
        return pre + "<script>\n" + (ROOT / "static/js" / m.group(1)).read_text(encoding="utf-8") + "\n</script>"
    return re.sub(r"<script src=[^>]*?/static/js/([a-z_]+\.js)[^>]*></script>", inline, html)


def main() -> int:
    browser = next((p for p in EDGE if Path(p).exists()), None)
    if browser is None:
        print("  SKIPPED: the sector chips -- no Edge/Chromium on this host, the page was NOT driven")
        return EXIT_SKIPPED
    tmp = ROOT / "scripts" / "_ft_sectors_ui.html"
    tmp.write_text(build_page(payloads()), encoding="utf-8")
    try:
        text = edge_cdp.run_page(browser, tmp, timeout_s=150, window="1700,1300")
    finally:
        tmp.unlink(missing_ok=True)
    fails = n = 0
    for line in text.strip().splitlines():
        parts = line.split("|", 3)
        if parts[0] == "errs":
            if parts[1] != "clean":
                print(f"  FAIL  the console is not clean: {parts[1][:1500]}")
                fails += 1
            continue
        n += 1
        ok = parts[0] == "ok"
        print(("  ok    " if ok else "  FAIL  ") + parts[1] + ("" if ok else f": {parts[2][:300]} (want {parts[3][:200]})"))
        fails += not ok
    print(f"factor-trades sector chips UI: {n} assertions, failures: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
