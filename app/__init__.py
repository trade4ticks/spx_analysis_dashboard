"""The dashboard package.

THE `scalp` ALIAS, and why it is here rather than anywhere else.

`app/scalp_config.py` is vendored VERBATIM from the equities-scalp pipeline,
and the pipeline's copy now contains:

    from scalp.quiet import MIN_TRADES as QUIET_MIN_TRADES, ...

Verbatim means byte-identical -- that is the whole point, so check_vendored
can diff the file and catch a threshold moving upstream -- so the line cannot
be edited here. And the standing rule is that `rm -rf scalp/` leaves this app
standing: the dashboard must never reach into the pipeline.

Both hold at once by binding the name to the copy WE ALREADY VENDOR.
`app/scalp_quiet.py` is in check_vendored's VERBATIM list and is held
byte-identical to `scalp/quiet.py`, so `scalp.quiet` resolves to the same
source either way and nothing leaves this repo.

It is registered in the package's __init__ because it has to be in place
before ANY import of `app.scalp_config`, and this is the one module
guaranteed to run first.

The find_spec guard exists so a machine that DOES have the pipeline
importable keeps using it: that machine is a developer's, the two files are
gated byte-identical, and silently shadowing a real package is a worse thing
to do than deferring to it.
"""
import importlib
import importlib.util
import sys
import types

if importlib.util.find_spec("scalp") is None:        # the normal case
    _quiet = importlib.import_module("app.scalp_quiet")
    _pkg = types.ModuleType("scalp")
    _pkg.__path__ = []                                # a package, with no files
    _pkg.quiet = _quiet
    sys.modules.setdefault("scalp", _pkg)
    sys.modules.setdefault("scalp.quiet", _quiet)
    del _quiet, _pkg
