# Texture Synthesis to Mesh-guided Gaussian Splatting

This repository is a research workspace for converting a single reference image and an input mesh into an editable novel-view rendering representation. The overall pipeline first synthesizes a textured mesh from a reference image, then uses the textured mesh or its rendered multi-view images to train a mesh-guided Gaussian Splatting representation.

The current project combines three main parts:

- `easi-tex/`: reference-guided mesh texture synthesis based on EASI-Tex-style rendering, ControlNet, IP-Adapter, inpainting, and UV back-projection.
- `easi-dyc/`: local modified texture synthesis workspace used for project-specific experiments.
- `gaussian-mesh-splatting/`: mesh-guided Gaussian Splatting training and rendering based on GaMeS and 3D Gaussian Splatting.
- `paper/`: LaTeX writing for the texture synthesis and Gaussian Splatting method sections.

## Pipeline

The intended workflow is:

1. Start from an input triangular mesh and a reference RGB image.
2. Render mesh views with depth, normal, edge, and mask information.
3. Transfer or synthesize view-consistent texture using diffusion features, DINO/IP-Adapter/ControlNet guidance, and inpainting.
4. Back-project the synthesized views into UV space to obtain a textured mesh
   `M_hat = (V, F, T)`.
5. Render or collect multi-view RGB images from the textured mesh.
6. Train a mesh-guided Gaussian Splatting model where splats are bound to mesh faces.
7. Render novel views from modified or held-out camera poses.

In short:

```text
reference image + input mesh
        -> synthesized multi-view texture images
        -> textured mesh
        -> mesh-guided Gaussian representation
        -> novel-view / editable rendering
```

## Repository Layout

```text
.
├── easi-tex/
│   ├── bash/run.sh
│   ├── scripts/generate_texture.py
│   ├── lib/
│   ├── data/
│   └── outputs*/
├── easi-dyc/
│   └── ...                    # local modified texture synthesis workspace
├── gaussian-mesh-splatting/
│   ├── train.py
│   ├── scripts/render.py
│   ├── metrics.py
│   ├── games/mesh_splatting/
│   ├── data/
│   └── output/
├── configs/
│   ├── texture/
│   └── gaussian/
├── scripts/
│   ├── camera/
│   ├── dataset/
│   ├── render/
│   └── train/
├── data/
│   └── scenes/
│       └── cat_more_face/
├── outputs/
│   ├── texture_synthesis/
│   └── gaussian_splatting/
├── assets/
│   ├── figures/
│   └── readme/
├── docs/
├── paper/
│   ├── texture synthesis (image transfer part)/
│   ├── texture_synthesis__gaussian_splatting_part/
│   └── Gaussian splatting part README.md
└── README.md
```

## Stage 1: Reference-guided Texture Synthesis

The first stage is implemented under `easi-tex/`, with project-specific variants under `easi-dyc/`. It takes an input mesh and a reference texture image, renders a set of mesh views, synthesizes/inpaints texture views, and back-projects them to the UV texture map.

### Environment

The original setup is described in `easi-tex/README.md`. A typical environment is:

```bash
cd easi-tex
conda create -n easitex python=3.10
conda activate easitex

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
conda install xformers -c xformers

pip install -r requirements.txt
```

The following pretrained assets are expected by the texture synthesis code:

- ControlNet Canny checkpoints under `easi-tex/models/ControlNet/models/`
- IP-Adapter image encoder under `easi-tex/ip_adapter/image_encoder/`
- IP-Adapter weights such as `ip-adapter-plus_sd15.bin` under `easi-tex/ip_adapter/`

### Demo

The demo command is:

```bash
cd easi-tex
./bash/run.sh
```

The script calls `scripts/generate_texture.py` with a mesh from `data/meshes/` and a reference image from `data/texture_images/`. It writes intermediate renders, inpainted views, masks, and textured mesh files to an output directory.

Useful output folders include:

- `generate/rendering/`: rendered input views
- `generate/edges/`: ControlNet edge conditions
- `generate/inpainted/`: synthesized or refined RGB images
- `generate/mask/`: projection and update masks
- `generate/mesh/`: intermediate textured meshes and texture maps
- `update/mesh/`: refined textured mesh results when update/refinement is enabled

## Stage 2: Mesh-guided Gaussian Splatting

The second stage is implemented under `gaussian-mesh-splatting/`. It trains a Gaussian representation from multi-view images and camera files. For the mesh-guided setting, the dataset should contain a mesh named `mesh.obj`, and the Gaussian centers are initialized on mesh faces.

### Environment

The original setup is described in `gaussian-mesh-splatting/README.md`. A typical environment is:

```bash
cd gaussian-mesh-splatting
conda env create --file environment.yml
conda activate gaussian_splatting_mesh

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

### Expected Dataset Format

For `gs_mesh`, each scene should follow the NeRF synthetic-style layout. Project-level scenes are stored under `data/scenes/`; the upstream GaMeS examples remain under `gaussian-mesh-splatting/data/`.

```text
data/scenes/<scene>/
├── mesh.obj
├── train/
│   ├── 0.png
│   ├── 1.png
│   └── ...
├── transforms_train.json
├── transforms_test.json
└── transforms_val.json      # optional
```

The `train/*.png` images can be:

- rendered RGB images from the final textured mesh, or
- view-wise synthesized/inpainted images from the texture synthesis stage.

The camera poses and intrinsics should be consistent with the image paths in `transforms_train.json` and `transforms_test.json`. If the test camera file is edited, the renderer will produce images from the new camera coordinates; full-reference metrics are only meaningful when the test image and test pose still correspond to the same rays.

### Training

Example mesh-guided training command:

```bash
cd gaussian-mesh-splatting
python train.py \
  --eval \
  -s ../data/scenes/cat_more_face \
  -m output/cat_more_face_gs_mesh \
  --gs_type gs_mesh \
  --num_splats 5 \
  -w
```

Important options:

- `-s` / `--source_path`: scene directory containing images, camera JSON files, and `mesh.obj`
- `-m` / `--model_path`: output directory for checkpoints, point clouds, and logs
- `--gs_type gs_mesh`: mesh-bound GaMeS-style Gaussian model
- `--num_splats`: number of Gaussian splats initialized per mesh face
- `-w`: use white background

### Rendering and Metrics

Render trained views:

```bash
cd gaussian-mesh-splatting
python scripts/render.py \
  -m output/cat_more_face_gs_mesh \
  --gs_type gs_mesh
```

Compute standard metrics:

```bash
cd gaussian-mesh-splatting
python metrics.py \
  -m output/cat_more_face_gs_mesh \
  --gs_type gs_mesh
```

When test cameras are modified for novel-view inspection, PSNR/SSIM/LPIPS should be interpreted carefully because the target image may no longer be a strict ground-truth match for the rendered rays. In that case, geometry-aware checks such as silhouette coverage, hole ratio, leakage ratio, and foreground sharpness are more appropriate.

## Connecting the Two Stages

The bridge between texture synthesis and Gaussian Splatting is a multi-view supervised dataset. After Stage 1, prepare a scene folder under `data/scenes/` for Stage 2 by collecting:

1. the final textured mesh as `mesh.obj`;
2. rendered or synthesized RGB supervision images under `train/`;
3. aligned camera files `transforms_train.json` and `transforms_test.json`;
4. optional validation cameras/images if needed.

The mesh-guided Gaussian model then learns a representation aligned with the textured surface instead of freely placing Gaussians in empty 3D space. This is useful for:

- stable geometry under sparse or synthesized supervision;
- alignment with the UV texture synthesis result;
- novel-view rendering from edited camera poses;
- downstream mesh-level editing because Gaussian centers and covariances remain tied to mesh faces.

## Project-level Pipeline Runner

This repository includes a lightweight runner that connects an EASI-DYC texture output folder to GaMeS training. The runner does not import either heavy codebase directly; it launches each stage as a subprocess so the texture and Gaussian stages can keep their own conda environments.

The example config is:

```bash
configs/pipeline/cat_more_face.json
```

Prepare a GaMeS scene from an existing texture synthesis output:

```bash
python scripts/run_pipeline.py \
  --config configs/pipeline/cat_more_face.json \
  --stage prepare_gs
```

This writes:

```text
data/scenes/cat_more_face_pipeline/
├── mesh.obj
├── 19_post.mtl
├── 19_post.png
├── train/
├── transforms_train.json
├── transforms_test.json
├── transforms_val.json
└── prepare_gs_scene_manifest.json
```

Run a quick Gaussian smoke test:

```bash
python scripts/run_pipeline.py \
  --config configs/pipeline/cat_more_face.json \
  --stage train_gs \
  --train-iterations 5

python scripts/run_pipeline.py \
  --config configs/pipeline/cat_more_face.json \
  --stage render_gs
```

For a full run, omit `--train-iterations` and use the `gaussian.iterations` value in the config:

```bash
python scripts/run_pipeline.py \
  --config configs/pipeline/cat_more_face.json \
  --stage train_gs
```

The runner exposes common experiment controls as command-line overrides, so a JSON config can be reused across several camera/image settings. For example, use ten synthesized views from `generate/inpainted` and an existing ten-camera transform file:

```bash
python scripts/run_pipeline.py \
  --config configs/pipeline/cat_more_face.json \
  --stage all \
  --image-dir generate/inpainted \
  --max-images 10 \
  --train-transforms data/scenes/cat_more_face/transforms_train.json \
  --model-path outputs/gaussian_splatting/cat_more_face_generate10 \
  --train-iterations 5
```

Useful overrides include:

- `--image-dir`, `--image-pattern`, `--image-filter`, `--max-images`: choose which synthesized images become GS supervision.
- `--train-transforms`, `--test-transforms`: choose explicit camera files instead of relying on the texture output's `viewpoints.json`.
- `--scene-dir`, `--model-path`: choose the prepared dataset folder and GS output folder.
- `--num-splats`, `--train-iterations`, `--gs-type`, `--white-background`, `--black-background`: control GaMeS training.
- `--render-iteration`, `--skip-train-render`, `--skip-test-render`: control rendering.

If multiple selected images resolve to repeated camera poses, scene preparation stops unless `--allow-repeat-cameras` is set. This catches the common mistake of treating update-step images from a locked camera as multi-view supervision.

## Paper Files

The writing materials are under `paper/`.

Important files:

- `paper/texture synthesis (image transfer part)/main.tex`: image transfer / texture synthesis method draft.
- `paper/texture synthesis (image transfer part)/references.bib`: references for the image transfer part.
- `paper/texture_synthesis__gaussian_splatting_part/main.tex`: standalone Gaussian Splatting method draft.
- `paper/texture_synthesis__gaussian_splatting_part/reference.bib`: references for the Gaussian Splatting part.
- `paper/Gaussian splatting part README.md`: Chinese outline and translation notes for the GS method section.

Project-level helper scripts are grouped by task:

- `scripts/camera/`: camera pose export and camera transform utilities.
- `scripts/dataset/`: texture output to Gaussian scene conversion utilities.
- `scripts/render/`: project-level render helpers or batch render commands.
- `scripts/train/`: project-level training launchers or batch scripts.

To compile the Gaussian Splatting part:

```bash
cd "paper/texture_synthesis__gaussian_splatting_part"
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Main References

The Gaussian Splatting part currently cites:

- Kerbl et al., 3D Gaussian Splatting for Real-Time Radiance Field Rendering.
- Waczynska et al., GaMeS: Mesh-Based Adapting and Modification of Gaussian Splatting.
- Mildenhall et al., NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.

The texture synthesis part is based on reference-guided mesh texturing, diffusion-based image generation/inpainting, ControlNet/IP-Adapter conditioning, feature matching, and UV back-projection.

## Notes

- This repository contains generated outputs and experiment folders under `outputs/`. These can be large and are mainly useful for local inspection.
- The root directory is a research workspace rather than a clean packaged library.
- Some commands require CUDA and compiled PyTorch extensions.
- For detailed dependencies and upstream usage, read the README files inside `easi-tex/` and `gaussian-mesh-splatting/`.
