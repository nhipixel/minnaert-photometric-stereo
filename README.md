# Quantifying the Breakdown of Lambertian Photometric Stereo under Minnaert Reflectance

Research code for a controlled study of how the Lambertian assumption fails in photometric stereo.

## Project Overview

Photometric stereo recovers a per pixel surface normal from several images of a static scene taken
under known, varying illumination. The standard solver assumes the surface is Lambertian. Real
surfaces are not.

This project measures exactly how much that assumption costs. Surfaces are rendered with the
Minnaert reflectance model, whose exponent c interpolates continuously from Lambertian at c equal
to 1 through the simplified Hapke model at c equal to 0.5, and the Lambertian solver is run on all
of it. The result is angular error as a function of a single scalar, predicted in closed form and
confirmed by measurement, then validated on the DiLiGenT benchmark.

Note the distinction between the simplified Hapke model used here and the full Hapke model. Only
the simplified form is a special case of Minnaert. The full model adds multiple scattering,
shadowing and porosity and is not.

## Layout

| Path | Contents |
|---|---|
| `src/geometry.py` | Unit sphere with analytic normals, so ground truth carries no discretization error |
| `src/lights.py` | Light rig generators and their condition numbers |
| `src/brdf.py` | Lambertian, Minnaert and simplified Hapke image formation |
| `src/solver.py` | Woodham least squares solver, plain and shadow aware |
| `src/metrics.py` | Angular error in degrees, with degenerate ground truth excluded |
| `src/theory.py` | Closed form error predictions |
| `src/diligent.py` | Benchmark loader, reproducing the reference conventions |
| `src/results_io.py` | Accumulator for every reported number |
| `experiments/` | One script per experiment, all driven by `run_all.py` |
| `tests/` | Every quality gate, encoded as assertions |
| `results/` | Generated figures and `results.json` |
| `paper/` | CVPR format report, with generated tables and figure copies |
| `presentation/` | Timed script for the ten minute video |

## Running

Install dependencies:

```
pip install -r requirements.txt
```

Run every quality gate:

```
python -m pytest tests/ -v
```

Regenerate every figure and every reported number:

```
python experiments/run_all.py
```

Numbers quoted in the report come only from `results/results.json`, produced by that run. Nothing
is transcribed by hand.

## DiLiGenT data

Download `DiLiGenT.zip` from
https://sites.google.com/site/photometricstereodata/single and extract it so that the object
folders sit under `data/DiLiGenT/pmsData/`, for example `data/DiLiGenT/pmsData/ballPNG/`.
A flattened `data/pmsData/` is also accepted.

Each object folder holds 96 16-bit PNG images, `light_directions.txt`,
`light_intensities.txt`, `filenames.txt`, `mask.png` and a ground truth normal map. The benchmark
uses an orthographic, single view setup, which is the same assumption the derivation in the report
makes.

The real data experiments are skipped with a message when `data/DiLiGenT/pmsData/` is absent, so a clean
clone still reproduces all synthetic results.