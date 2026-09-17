"""Which broker the service is talking to.

ONE ACTIVE ADAPTER, chosen by `LIVE_BROKER` and resolved once. Adapters are
imported LAZILY, inside `active()`: importing Schwab's module on a box
configured for DAS would pull in credentials and a token file that are not
this broker's business, and the reverse holds too.

Adding a broker is: write the adapter against live/brokers/base.Broker, add
one line to `_ADAPTERS`, set LIVE_BROKER. Nothing in the pane, the endpoints
or the arming logic changes -- they only ever see the façade in
live/broker.py, which is also where the switches and guards stay.
"""
from __future__ import annotations

import logging

from live import config
from live.brokers.base import Broker, BrokerError, BrokerIndeterminate

log = logging.getLogger("live.brokers")

__all__ = ["Broker", "BrokerError", "BrokerIndeterminate", "active", "name", "reset"]


def _schwab() -> Broker:
    from live.brokers.schwab import SchwabBroker
    return SchwabBroker()


# key -> factory. The key is what LIVE_BROKER is set to.
_ADAPTERS = {
    "schwab": _schwab,
}

_active: Broker | None = None


def name() -> str:
    """The configured key, whether or not it resolves."""
    return (config.BROKER or "schwab").strip().lower()


def active() -> Broker:
    """The one adapter this process trades through.

    A misconfigured LIVE_BROKER raises rather than silently falling back to
    Schwab: a box that was meant to be pointed at another broker must not
    quietly keep trading through the old one.
    """
    global _active
    if _active is None:
        key = name()
        if key not in _ADAPTERS:
            raise BrokerError(
                f"LIVE_BROKER is {key!r}, which is not a broker this service "
                f"has an adapter for ({', '.join(sorted(_ADAPTERS))}).")
        _active = _ADAPTERS[key]()
        log.info("broker adapter: %s", _active.name)
    return _active


def reset() -> None:
    """Drop the resolved adapter. For checks that swap brokers in-process."""
    global _active
    _active = None
