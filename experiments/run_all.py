"""
Regenerate every figure, every recorded number, and every report table.

Real data experiments are skipped with a message when the benchmark is absent,
so a clean clone still reproduces all synthetic results.

    python experiments/run_all.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)

import exp2_diligent
import exp3_sweep_c
import exp4_ablations
import exp5_landscape
import exp6_exponent
import make_tables
from diligent import is_available
from results_io import RESULTS_PATH


def main():
    steps = [
        ("DiLiGenT baseline", exp2_diligent.run, True),
        ("Minnaert exponent sweep", exp3_sweep_c.run, False),
        ("Ablations", exp4_ablations.run, False),
        ("Published method landscape", exp5_landscape.run, True),
        ("Effective exponent and correction", exp6_exponent.run, True),
    ]

    for title, fn, needs_data in steps:
        print(f"\n===== {title} =====")
        if needs_data and not is_available():
            print("benchmark not found, skipping. See README for the download.")
            continue
        fn()

    # Tables are generated last so they reflect the run that just completed.
    print("\n===== report tables =====")
    make_tables.run()
    print(f"\nAll done. Numbers in {RESULTS_PATH}")


if __name__ == "__main__":
    main()
