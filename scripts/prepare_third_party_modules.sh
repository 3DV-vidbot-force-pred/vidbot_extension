#!/usr/bin/env bash
#
# One-shot setup: clone all submodules, install packages, download weights.
# Safe to re-run — skips steps that are already done.
#
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON="python3"
PIP="pip3"
# Prefer uv if available (ensures we use the project venv)
if command -v uv &>/dev/null; then
    PIP="uv pip"
    PYTHON="uv run python3"
fi

# ══════════════════════════════════════════════════════════════════════════════
# 1. Clone / update all submodules
# ══════════════════════════════════════════════════════════════════════════════
echo "==> Initializing git submodules"
git submodule sync --recursive
# Clone each submodule individually so one failure doesn't block the rest
for mod in GroundingDINO EfficientSAM graspnetAPI graspness_unofficial Metric3D DepthAnythingV3; do
    if [ -d "third_party/$mod/.git" ] || [ -f "third_party/$mod/.git" ]; then
        echo "   $mod: already cloned"
    else
        echo "   $mod: cloning..."
        git submodule update --init --recursive "third_party/$mod" 2>&1 || echo "   WARNING: $mod clone failed (may need SSH keys or network)"
    fi
done
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# 2. GroundingDINO — open-vocabulary object detector
# ══════════════════════════════════════════════════════════════════════════════
echo "==> GroundingDINO"
# Install as editable package (provides groundingdino module)
if $PYTHON -c "import groundingdino" 2>/dev/null; then
    echo "   Already installed"
else
    echo "   Installing from source..."
    $PIP install -e third_party/GroundingDINO --no-build-isolation
fi
# Download weight
GDINO_W="third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth"
if [ -f "$GDINO_W" ]; then
    echo "   Weight already present"
else
    echo "   Downloading weight..."
    mkdir -p "$(dirname "$GDINO_W")"
    wget -q --show-progress -O "$GDINO_W" \
        https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# 3. EfficientSAM — segmentation
# ══════════════════════════════════════════════════════════════════════════════
echo "==> EfficientSAM"
if [ -f "third_party/EfficientSAM/weights/efficient_sam_vitt.pt" ]; then
    echo "   Weight already present (ships with repo clone)"
else
    echo "   WARNING: Weight not found — check that the submodule cloned correctly"
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# 4. graspnetAPI — grasp evaluation
# ══════════════════════════════════════════════════════════════════════════════
echo "==> graspnetAPI"
if $PYTHON -c "from graspnetAPI.graspnet_eval import GraspGroup" 2>/dev/null; then
    echo "   Already installed"
else
    if [ -d "third_party/graspnetAPI" ] && [ -f "third_party/graspnetAPI/setup.py" ]; then
        echo "   Installing from source..."
        # transforms3d==0.3.1 (pinned by graspnetAPI) doesn't build on Python 3.12+
        $PIP install "transforms3d>=0.4" 2>/dev/null || true
        $PIP install -e third_party/graspnetAPI --no-deps
        # Install remaining deps that are compatible
        $PIP install autolab_core trimesh 2>/dev/null || true
    else
        echo "   SKIP: submodule not cloned (optional — needed for grasp evaluation)"
    fi
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# 5. graspness_unofficial — grasp detection model
# ══════════════════════════════════════════════════════════════════════════════
echo "==> graspness_unofficial"
GRASP_W="third_party/graspness_unofficial/weights/minkuresunet_kinect.tar"
if [ -f "$GRASP_W" ]; then
    echo "   Weight already present"
elif [ -d "third_party/graspness_unofficial" ]; then
    echo "   Downloading weight..."
    mkdir -p "$(dirname "$GRASP_W")"
    $PYTHON -m gdown -O "$GRASP_W" "https://drive.google.com/uc?id=10o5fc8LQsbI8H0pIC2RTJMNapW9eczqF"
else
    echo "   SKIP: submodule not cloned (optional — needed for learned grasp detection)"
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# 6. Metric3D — monocular metric depth estimation
# ══════════════════════════════════════════════════════════════════════════════
echo "==> Metric3D"
if [ -d "third_party/Metric3D" ]; then
    # mmcv/mmengine are needed at runtime
    if $PYTHON -c "import mmcv" 2>/dev/null; then
        echo "   mmcv already installed"
    else
        echo "   Installing mmcv + mmengine..."
        $PIP install setuptools mmcv==2.1.0 mmengine --no-build-isolation 2>/dev/null \
            || echo "   WARNING: mmcv install failed — install manually if using Metric3D"
    fi
    # timm is also needed
    if $PYTHON -c "import timm" 2>/dev/null; then
        echo "   timm already installed"
    else
        echo "   Installing timm..."
        $PIP install timm
    fi
    echo "   Weights auto-download from HuggingFace on first use"
else
    echo "   SKIP: submodule not cloned"
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# 7. Depth Anything V3 — monocular depth estimation
# ══════════════════════════════════════════════════════════════════════════════
echo "==> Depth Anything V3"
if $PYTHON -c "from depth_anything_3 import DepthAnything3" 2>/dev/null; then
    echo "   Already installed"
elif [ -d "third_party/DepthAnythingV3" ]; then
    echo "   Installing from source (--no-deps to skip xformers, which needs CUDA)..."
    $PIP install -e third_party/DepthAnythingV3 --no-deps
    # Install the deps we actually need (skip xformers, pycolmap, gsplat)
    $PIP install trimesh einops huggingface_hub imageio safetensors e3nn plyfile pillow_heif 2>/dev/null || true
    echo "   Weights auto-download from HuggingFace on first use"
else
    echo "   SKIP: submodule not cloned"
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
echo "==> Done! All third-party modules are set up."
