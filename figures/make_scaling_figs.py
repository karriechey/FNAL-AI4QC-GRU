"""
Rebuild every scaling figure, table and the summary PDF.

Created: 2026-10-05
Last updated: 2026-10-05

All logic lives in scaling_figs.py (importable, also used by
notebooks/scaling_laws.ipynb); this file is only the command-line entry point.
Rerun it whenever the all-runs CSV gains rows:

    python figures/make_scaling_figs.py [--csv GRU_all_runs_<date>.csv] [--extra-root <checkout>]
"""
import matplotlib

matplotlib.use("Agg")

from scaling_figs import main  # noqa: E402  (backend must be set first)

if __name__ == "__main__":
    main()
