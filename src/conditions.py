"""Single source of truth for the foveation-condition bracket used by the offline analysis.

Both compute_metrics.py and compare.py iterate exactly these condition dirs, so the pipeline is
config-driven and model-agnostic: changing model.slug + re-running regenerates everything, and
temperature-sweep dirs (fov-*-t<T>) never leak into the main bracket.
"""
import os
import re

_TEMP = re.compile(r"-t[0-9.]+$")   # temperature-sweep tag, e.g. fov-sharp-t0.2


def condition_dirs(cfg, raw_dir, prefix="fov-"):
    """Ordered list of condition dirs present under `raw_dir`.

    If cfg['analysis']['bracket'] is set, use exactly that list (filtered to existing dirs), giving
    a deterministic order and excluding anything not named in the bracket. Otherwise fall back to
    sorted on-disk discovery by `prefix`, excluding temperature-sweep dirs."""
    if not os.path.isdir(raw_dir):
        return []
    bracket = (cfg.get("analysis") or {}).get("bracket")
    if bracket:
        return [d for d in bracket if d.startswith(prefix) and os.path.isdir(os.path.join(raw_dir, d))]
    return sorted(d for d in os.listdir(raw_dir)
                  if d.startswith(prefix) and not _TEMP.search(d)
                  and os.path.isdir(os.path.join(raw_dir, d)))
