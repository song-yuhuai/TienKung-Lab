# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

from pathlib import Path
import runpy

_X2_MODULE_PATH = Path(__file__).resolve().parent / "X2_URDF-v1.3.0" / "x2.py"
if not _X2_MODULE_PATH.exists():
    raise FileNotFoundError(f"Expected X2 asset config at '{_X2_MODULE_PATH}'.")

_module = runpy.run_path(str(_X2_MODULE_PATH))

X2_CFG = _module["X2_CFG"]
X2_USD_PATH = _module["X2_USD_PATH"]
validate_x2_usd_path = _module["validate_x2_usd_path"]

__all__ = ["X2_CFG", "X2_USD_PATH", "validate_x2_usd_path"]