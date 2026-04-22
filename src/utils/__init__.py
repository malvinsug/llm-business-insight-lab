"""
Utility functions for the LLM Business Insight Lab.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import yaml

from loguru import logger


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config {path}: {e}")
        return {}


def load_json_config(path: Path) -> Dict[str, Any]:
    """Load JSON configuration file."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load config {path}: {e}")
        return {}


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        return f"{seconds/60:.1f}m"


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy and special types."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


def to_json(obj: Any, **kwargs) -> str:
    """Serialize to JSON with custom encoder."""
    return json.dumps(obj, cls=JSONEncoder, **kwargs)
