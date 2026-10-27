#!/usr/bin/env python3
"""Standalone entry point for the Calix Telco-Bench read-only monitor."""
from __future__ import annotations

import sys

from calix_telco_bench.cli import main


if __name__ == "__main__":
    sys.argv.insert(1, "tui")
    main()
