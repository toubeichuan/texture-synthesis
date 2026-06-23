#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-texture-synthesis}"
cd "${PROJECT_ROOT}"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda env update -n "${ENV_NAME}" -f "${PROJECT_ROOT}/environment.yml"
else
  conda env create -n "${ENV_NAME}" -f "${PROJECT_ROOT}/environment.yml"
fi

CC_BIN="${CC:-$(command -v gcc-10 || command -v gcc)}"
CXX_BIN="${CXX:-$(command -v g++-10 || command -v g++)}"

for extension in diff-gaussian-rasterization simple-knn; do
  conda run -n "${ENV_NAME}" env \
    PYTHONNOUSERSITE=1 \
    CC="${CC_BIN}" \
    CXX="${CXX_BIN}" \
    python -m pip install \
      --no-build-isolation \
      --no-deps \
      --force-reinstall \
      "${PROJECT_ROOT}/gaussian-mesh-splatting/submodules/${extension}"
done

conda run -n "${ENV_NAME}" env PYTHONNOUSERSITE=1 python -c \
  "import diff_gaussian_rasterization, simple_knn, torch; print(f'Environment ready: torch={torch.__version__}, cuda={torch.version.cuda}')"
