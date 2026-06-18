"""FL base class re-export for Module 5.

The shared ``BaseClient`` and ``BaseServer`` still live in ``common``.  Module
5's attack-aware robust servers are implemented in ``defensive_servers.py`` so
they can reuse Module 4's malicious-client pipeline and override aggregation
without changing the attack recipe.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.federated_core import BaseClient, BaseServer, StateDict  # noqa: F401

__all__ = ["BaseClient", "BaseServer", "StateDict"]
