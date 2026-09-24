/* The topbar nav's six category menus.
 *
 * A DISCLOSURE NAVIGATION MENU, not an application menu. Each category is a
 * button carrying aria-expanded, and its menu holds ordinary links that Tab
 * reaches in document order — the W3C pattern for site navigation. The
 * alternative (role="menu"/"menuitem" with a roving tabindex) takes the
 * links OUT of the tab order and is for application menus with commands, not
 * for a list of pages.
 *
 * PLAIN JAVASCRIPT, NOT ALPINE. This nav is included by every page, each of
 * which has its own Alpine component on <body>; a nested component here
 * would be a second thing for every page's gates to know about, and a page
 * that ever stopped loading Alpine would lose its navigation rather than
 * some feature of its own.
 *
 * KEYBOARD, because seventeen pages behind six buttons is a thing people
 * drive without reaching for the mouse:
 *   on a category   Enter/Space   toggle (the button's own behaviour)
 *                   Down/Up       open, focus the first/last page
 *                   Left/Right    move to the next category, keeping an open
 *                                 menu open so you can run along the bar
 *                   Escape        close
 *   in a menu       Down/Up       move between pages, wrapping
 *                   Home/End      first/last
 *                   Left/Right    the next category's menu
 *                   Escape        close and put focus back on the category,
 *                                 which is where it came from
 *                   Tab           close and carry on into the page
 */
'use strict';

(function () {
  function init() {
    const nav = document.querySelector('.topbar-nav');
    if (!nav) return;
    const cats = Array.prototype.slice.call(nav.querySelectorAll('.nav-cat'));
    if (!cats.length) return;

    const btn = (c) => c.querySelector('.nav-cat-btn');
    const links = (c) => Array.prototype.slice.call(
      c.querySelectorAll('.nav-menu .nav-link'));

    function close(cat) {
      cat.classList.remove('open');
      btn(cat).setAttribute('aria-expanded', 'false');
    }

    function closeAll(except) {
      cats.forEach((c) => { if (c !== except) close(c); });
    }

    /* `focus` is 'first', 'last' or nothing: opening with the mouse should
     * not move focus off the button, while opening with a key should land on
     * a page or the keyboard user is left nowhere. */
    function open(cat, focus) {
      closeAll(cat);
      cat.classList.add('open');
      btn(cat).setAttribute('aria-expanded', 'true');
      const ls = links(cat);
      if (focus === 'first' && ls.length) ls[0].focus();
      if (focus === 'last' && ls.length) ls[ls.length - 1].focus();
    }

    function isOpen(cat) { return cat.classList.contains('open'); }

    function sibling(cat, delta, focus) {
      const i = cats.indexOf(cat);
      const next = cats[(i + delta + cats.length) % cats.length];
      // RUNNING ALONG THE BAR KEEPS THE MENU OPEN if one already was, and
      // just moves the focus if none was: arrowing sideways should not open
      // menus you were only passing.
      if (isOpen(cat)) open(next, focus || 'first');
      else { closeAll(); btn(next).focus(); }
    }

    cats.forEach((cat) => {
      const b = btn(cat);

      b.addEventListener('click', (e) => {
        e.preventDefault();
        if (isOpen(cat)) close(cat); else open(cat);
      });

      b.addEventListener('keydown', (e) => {
        const k = e.key;
        if (k === 'ArrowDown') { e.preventDefault(); open(cat, 'first'); }
        else if (k === 'ArrowUp') { e.preventDefault(); open(cat, 'last'); }
        else if (k === 'ArrowRight') { e.preventDefault(); sibling(cat, 1); }
        else if (k === 'ArrowLeft') { e.preventDefault(); sibling(cat, -1); }
        else if (k === 'Escape') { close(cat); }
      });

      cat.addEventListener('keydown', (e) => {
        // THE BUTTON'S KEYS ARE THE BUTTON'S. This listener is on the
        // category, which CONTAINS the button, so a keydown handled above
        // bubbles down here as well — and by the time it does, the handler
        // above has already moved focus into the menu. Down on the button
        // then opened the menu, landed on the first page, and immediately
        // stepped to the second; every arrow moved two. Caught by driving
        // the keyboard in a real browser, which is the only place the two
        // handlers actually meet.
        if (e.target === b) return;
        const ls = links(cat);
        const i = ls.indexOf(document.activeElement);
        const k = e.key;
        if (k === 'Escape') {
          // Focus goes back to the button it came from. Leaving it on a
          // hidden link strands the keyboard at the top of the document.
          close(cat);
          b.focus();
          return;
        }
        if (k === 'Tab') { close(cat); return; }
        if (i < 0) return;                       // focus is on the button
        if (k === 'ArrowDown') {
          e.preventDefault();
          ls[(i + 1) % ls.length].focus();
        } else if (k === 'ArrowUp') {
          e.preventDefault();
          ls[(i - 1 + ls.length) % ls.length].focus();
        } else if (k === 'Home') {
          e.preventDefault();
          ls[0].focus();
        } else if (k === 'End') {
          e.preventDefault();
          ls[ls.length - 1].focus();
        } else if (k === 'ArrowRight') {
          e.preventDefault();
          sibling(cat, 1, 'first');
        } else if (k === 'ArrowLeft') {
          e.preventDefault();
          sibling(cat, -1, 'first');
        }
      });
    });

    // A click anywhere else closes. `mousedown` rather than `click` so the
    // menu is gone before whatever was clicked does its own work.
    document.addEventListener('mousedown', (e) => {
      if (!nav.contains(e.target)) closeAll();
    });
    // And focus leaving the nav entirely — Tab out of the last link, or a
    // click into a text field — leaves nothing hanging open.
    document.addEventListener('focusin', (e) => {
      if (!nav.contains(e.target)) closeAll();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
