import os
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        return yaml.safe_load(f)


def path(*parts):
    return os.path.join(ROOT, *parts)


def ensure_dirs():
    for parts in [("outputs", "figures"), ("outputs", "tables"), ("artifacts",), ("data",)]:
        os.makedirs(path(*parts), exist_ok=True)