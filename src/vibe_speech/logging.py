from __future__ import annotations

import logging
from typing import Dict, Optional


def setup_logging(level: str = "INFO", module_levels: Optional[Dict[str, str]] = None) -> None:
    root_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=root_level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    if module_levels:
        for name, lvl in module_levels.items():
            logger = logging.getLogger(name)
            logger.setLevel(getattr(logging, lvl.upper(), root_level))
