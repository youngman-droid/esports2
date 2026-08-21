"""Entry point usable from any cwd: python3 /path/to/run_dashboard.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.argv = [sys.argv[0], "dashboard"] + sys.argv[1:]

from lol_ticker.__main__ import main  # noqa: E402

sys.exit(main())
