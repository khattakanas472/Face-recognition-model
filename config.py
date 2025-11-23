"""Simple configuration loader."""

import yaml


def load_config(path='configs/config.yaml'):
    """Load configuration from YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)

