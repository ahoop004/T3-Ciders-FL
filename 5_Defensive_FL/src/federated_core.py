"""FL base class re-export for Module 5.

The shared ``BaseClient`` and ``BaseServer`` still live in ``common``.  Module
5's attack-aware robust servers are implemented in ``defensive_servers.py`` so
they can reuse Module 4's malicious-client pipeline and override aggregation
without changing the attack recipe.
"""

import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_DIR = os.path.dirname(SRC_DIR)
REPO_ROOT = os.path.dirname(MODULE_DIR)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.federated_core import BaseClient, BaseServer, StateDict  # noqa: F401

__all__ = ["BaseClient", "BaseServer", "StateDict"]
