# GitHub Release Checklist

This project is a research workspace with two heavy runtime stages:

1. texture synthesis with EASI-Tex-style code;
2. mesh-guided Gaussian Splatting with GaMeS/3DGS-style code.

Do not publish generated outputs, model checkpoints, downloaded weights, or large example datasets directly to GitHub.

## Recommended Repository Contents

Commit:

- root `README.md`
- root `.gitignore`
- project scripts under `scripts/`
- lightweight configs under `configs/`
- paper source files under `paper/`
- small demo assets under `assets/`
- local code changes that are part of the method

Do not commit:

- `outputs/`
- `data/scenes/`
- `*.pth`, `*.pt`, `*.ckpt`, `*.safetensors`
- downloaded ControlNet/IP-Adapter/Stable-Diffusion weights
- `gaussian-mesh-splatting/data/`
- `gaussian-mesh-splatting/output/`
- local installers such as `*.deb`

## Environment Strategy

Keep two environments instead of forcing a single root `requirements.txt`.

Texture synthesis:

```bash
conda create -n easitex python=3.10
conda activate easitex
cd easi-tex
pip install -r requirements.txt
```

Gaussian Splatting:

```bash
cd gaussian-mesh-splatting
conda env create --file environment.yml
conda activate gaussian_splatting_mesh
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

The CUDA/PyTorch extension requirements are different enough that a single environment is more fragile.

## Third-party Code

Before publishing, decide how to handle upstream projects:

- If a folder is unchanged upstream code, prefer a Git submodule or a clear installation instruction.
- If a folder contains local modifications, publish it as your own fork or include it directly after removing nested `.git` metadata.
- Avoid committing nested Git repositories accidentally. Git will warn about embedded repositories if `easi-tex/` or `gaussian-mesh-splatting/` still contain their own `.git` directories.

## Suggested First Commit

From the repository root:

```bash
git init
git status --short
git add README.md .gitignore docs scripts configs paper
git commit -m "Initial project structure"
```

Add third-party code only after deciding whether it should be vendored or added as submodules.
