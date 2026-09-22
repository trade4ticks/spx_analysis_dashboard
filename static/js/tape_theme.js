/* The tape's colours. ONE DEFINITION, read by every page that draws a tape.
 *
 * Equities Live and the Equities Wall draw the same picture at two sizes — a
 * wall pane is meant to look like a small live pane — so a colour defined
 * twice is two pages that agree until one of them is edited. This file is
 * the definition; both bundles read these names and no page carries its own
 * literal.
 *
 * Loaded BEFORE the page bundle and not deferred: top-level `const` in a
 * classic script is visible to the scripts that follow it, and a deferred
 * one would run after the bundle that reads it.
 */
'use strict';

const TAPE_BLUE = '#3498db';
const TAPE_PINK = '#e84393';

/* THE BID IS BLUE AND THE ASK IS PINK, lightened so a 1px line at a small
 * size still reads as its colour rather than as dark grey. */
const TAPE_BID = 'rgba(130,190,235,0.95)';
const TAPE_ASK = 'rgba(235,150,190,0.95)';

/* PRINTS ARE NEUTRAL.
 *
 * They were once the same blue as the bid line, which reads as "buys" before
 * any decision to read it that way. A print is a print; the colour says
 * nothing unless the tape page's bichrome toggle is deliberately turned on.
 *
 * TRANSLUCENT WITH A RIM, so overlap reads as overlap: at a solid fill one
 * 400-share print and four 100-share prints at the same price are the same
 * disc, while with alpha the stack darkens and with a rim the individual
 * prints stay countable. (The rim is for discs big enough to have one — on
 * the wall's small panes most prints are under that size.) */
const TAPE_TRADE_FILL = 'rgba(206,212,220,0.30)';
const TAPE_TRADE_RIM = 'rgba(228,233,240,0.80)';

/* The same grey at a legend's opacity: the tape page's size key draws single
 * discs that nothing overlaps, so the alpha that makes a stack readable
 * makes a lone disc look faint. Here rather than written out again, so the
 * GREY has one definition however many opacities it is drawn at. */
const TAPE_TRADE_LEGEND = 'rgba(206,212,220,0.42)';
