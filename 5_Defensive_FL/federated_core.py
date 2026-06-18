"""FL base classes for Module 5 — defensive federated learning.

Import BaseClient and BaseServer from here, then extend BaseServer and
override ``aggregate()`` to implement robust aggregation rules such as
coordinate-wise median, trimmed mean, or Krum.

Example::

    from federated_core import BaseClient, BaseServer

    class MedianServer(BaseServer):
        def aggregate(self, local_states):
            # replace simple average with coordinate-wise median
            ...
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.federated_core import BaseClient, BaseServer, StateDict  # noqa: F401

__all__ = ["BaseClient", "BaseServer", "StateDict"]
