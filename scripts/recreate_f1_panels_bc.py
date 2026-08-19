#!/usr/bin/env python3
"""Create a readable B/C-only companion figure from the saved F1 raster.

This is a figure-only operation: it crops and recomposes the existing image and
does not load or recompute any analysis tables, states, models, or metrics.
"""

from pathlib import Path
import subprocess


ROOT = Path(
    "/home/myriamcharfeddine/CGM/SSM-CGM/outputs/"
    "static_phenotype_trajectory_stratified_v2/neighbor_transition_drivers/"
    "direct_variable_level_figure_v2"
)
SOURCE = ROOT / "figures/figure_F1_direct_neighborhood_transition_drivers.png"
OUT_PNG = ROOT / "figures/figure_F1_panels_B_C_direct_transition_drivers.png"
OUT_PDF = ROOT / "figures/figure_F1_panels_B_C_direct_transition_drivers.pdf"
TMP_B = Path("/tmp/f1_panel_b.png")
TMP_C = Path("/tmp/f1_panel_c.png")


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    # Bounds include each panel title, axes, legends, and labels, while
    # excluding the neighboring panels. Coordinates refer to the 4061x3993
    # source PNG.
    run(
        "convert",
        str(SOURCE),
        "-crop",
        "1830x1680+2180+340",
        "+repage",
        "-bordercolor",
        "white",
        "-border",
        "30",
        str(TMP_B),
    )
    run(
        "convert",
        str(SOURCE),
        "-crop",
        "1900x1650+80+2300",
        "+repage",
        "-bordercolor",
        "white",
        "-border",
        "30",
        str(TMP_C),
    )
    run(
        "convert",
        str(TMP_B),
        str(TMP_C),
        "+append",
        "-trim",
        "+repage",
        "-background",
        "white",
        "-alpha",
        "remove",
        "-alpha",
        "off",
        str(OUT_PNG),
    )
    run(
        "convert",
        str(OUT_PNG),
        "-density",
        "300",
        str(OUT_PDF),
    )
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
