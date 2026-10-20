#!/usr/bin/env python3
"""Standalone entry point for the TelcoBench read-only monitor."""
from __future__ import annotations

import sys

from network_llm_eval.cli import main


if __name__ == "__main__":
    sys.argv.insert(1, "tui")
    main()
