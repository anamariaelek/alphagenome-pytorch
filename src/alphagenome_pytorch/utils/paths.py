"""Path expansion utilities for handling ~ and environment variables.

This module provides utilities for expanding paths in configuration files
and command-line arguments to support portable configs across different users.
"""

import os
from pathlib import Path
from typing import Any


def expand_path(path_str: str | None) -> str | None:
    """Expand ~ and environment variables in path strings.
    
    Supports:
    - `~` expansion to user home directory
    - `$VAR` and `${VAR}` environment variable expansion
    
    Args:
        path_str: Path string to expand, or None.
        
    Returns:
        Expanded path string, or None if input was None.
        
    Examples:
        >>> expand_path("~/data/genome.fa")
        '/home/user/data/genome.fa'
        
        >>> expand_path("$HOME/data/genome.fa")
        '/home/user/data/genome.fa'
        
        >>> expand_path("${DATA_DIR}/genome.fa")
        '/path/to/data/genome.fa'
    """
    if path_str is None or not isinstance(path_str, str):
        return path_str
    # Expand environment variables ($VAR or ${VAR})
    expanded = os.path.expandvars(path_str)
    # Expand ~ to home directory
    expanded = os.path.expanduser(expanded)
    return expanded


def expand_paths_in_dict(d: dict, path_keys: set[str]) -> dict:
    """Recursively expand paths in a dictionary for specified keys.
    
    This function walks through a nested dictionary structure and expands
    any string values whose keys match the provided path_keys set.
    
    Args:
        d: Dictionary to process (possibly nested).
        path_keys: Set of key names that should be treated as paths.
        
    Returns:
        New dictionary with expanded paths.
        
    Examples:
        >>> config = {
        ...     "genome": "~/data/genome.fa",
        ...     "other": "not_a_path",
        ...     "nested": {"genome": "$HOME/data/other.fa"}
        ... }
        >>> expand_paths_in_dict(config, {"genome"})
        {
            'genome': '/home/user/data/genome.fa',
            'other': 'not_a_path',
            'nested': {'genome': '/home/user/data/other.fa'}
        }
    """
    result = {}
    for key, value in d.items():
        if key in path_keys and isinstance(value, str):
            result[key] = expand_path(value)
        elif isinstance(value, dict):
            result[key] = expand_paths_in_dict(value, path_keys)
        elif isinstance(value, list):
            result[key] = [
                expand_paths_in_dict(item, path_keys) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


__all__ = [
    "expand_path",
    "expand_paths_in_dict",
]
