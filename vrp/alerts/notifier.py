"""Alert delivery to console, log file, and (optionally) Discord webhook.

Console output goes via the project's logger so it is also captured in the
rotating log file. Discord is a no-op unless a webhook URL is configured;
this keeps the build self-contained and unit-testable without a network.
"""
from __future__ import annotations

import logging
from typing import Protocol

import requests

from vrp.config import AlertsConfig

logger = logging.getLogger(__name__)

_DISCORD_TIMEOUT = 5.0


class Notifier(Protocol):
    """Anything that can publish a notification."""

    def notify(self, title: str, body: str) -> None: ...


class ConsoleNotifier:
    """Logs the title and body via the standard logger."""

    def notify(self, title: str, body: str) -> None:
        logger.info("%s: %s", title, body.replace("\n", " | "))


class DiscordNotifier:
    """Posts to a Discord webhook URL. Silently no-ops if no URL is set."""

    def __init__(self, webhook_url: str | None, username: str = "VRP") -> None:
        self.url = webhook_url
        self.username = username

    def notify(self, title: str, body: str) -> None:
        if not self.url:
            return
        payload = {"username": self.username, "content": f"**{title}**\n{body}"}
        try:
            resp = requests.post(self.url, json=payload, timeout=_DISCORD_TIMEOUT)
            if resp.status_code >= 300:
                logger.warning("discord webhook %d: %s", resp.status_code, resp.text)
        except requests.RequestException as exc:
            logger.warning("discord post failed: %s", exc)


class MultiNotifier:
    """Fan-out: send each notification to every wrapped notifier."""

    def __init__(self, notifiers: list[Notifier]) -> None:
        self.notifiers = notifiers

    def notify(self, title: str, body: str) -> None:
        for n in self.notifiers:
            n.notify(title, body)


def build_notifier(cfg: AlertsConfig) -> Notifier:
    """Construct the default fan-out notifier (console + Discord stub)."""
    return MultiNotifier(
        [
            ConsoleNotifier(),
            DiscordNotifier(cfg.discord_webhook_url, cfg.discord_username),
        ]
    )
