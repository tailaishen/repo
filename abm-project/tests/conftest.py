"""Pytest configuration: makes the ``abm-project`` root importable.

Adds the project root to ``sys.path`` so tests can ``import abm`` regardless of
where pytest is invoked from.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
