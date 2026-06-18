"""Re-export shim — FL base classes now live in common/federated_core.py."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.federated_core import BaseClient, BaseServer, StateDict  # noqa: F401

__all__ = ["BaseClient", "BaseServer", "StateDict"]
