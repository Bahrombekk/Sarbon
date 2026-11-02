"""YAML run configs for the command-line tools.

A config file holds argparse option names (dashes or underscores) and their values; options given
on the command line win over the file. The file path is stored in the parsed args as `config`.

    # configs/train/railsafe_s.yaml
    taxonomy: railsafe
    yolo: C:/data/pereezd_split
    epochs: 24
"""
import argparse
import sys
from pathlib import Path

import yaml


def load_config(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping of option: value")
    return {k.replace("-", "_"): v for k, v in data.items()}


def parse_with_config(ap: argparse.ArgumentParser, argv=None):
    """Parse argv with `--config FILE` support: the file sets defaults, explicit flags override."""
    ap.add_argument("--config", help="YAML file with option: value pairs (flags override it)")
    argv = sys.argv[1:] if argv is None else argv
    pre, _ = ap.parse_known_args(argv)
    if pre.config:
        cfg = load_config(pre.config)
        known = {a.dest for a in ap._actions}
        unknown = sorted(set(cfg) - known)
        if unknown:
            ap.error(f"{pre.config}: unknown option(s) {', '.join(unknown)}")
        ap.set_defaults(**cfg)
        # options marked required may now be satisfied by the file
        for a in ap._actions:
            if a.dest in cfg:
                a.required = False
    return ap.parse_args(argv)
