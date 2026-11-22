# Shared environment setup for the cluster jobs. Sourced by the *.sbatch wrappers.
#
# Create the environment once, from the repository root:
#
#   python3.10 -m venv .venv
#   source .venv/bin/activate
#   pip install --upgrade pip
#   pip install -r requirements-train.txt                                  # CPU only
#   pip install -r requirements-train.txt "tensorflow[and-cuda]==2.16.1"   # with GPU
#
# Override the location with VENV=/path/to/venv if it does not live in the repo root.

VENV="${VENV:-$PWD/.venv}"
if [ ! -f "$VENV/bin/activate" ]; then
    echo "No virtual environment at $VENV -- see the header of scripts/training/activate_env.sh" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# TensorFlow's GPU libraries ship inside the nvidia-* pip wheels in the venv. Point the
# dynamic loader at them explicitly: on a shared cluster LD_LIBRARY_PATH usually already
# names a system CUDA of a different version, which wins otherwise and silently sends
# TensorFlow to the CPU -- turning a one-hour job into an overnight one.
NVIDIA_LIB_DIRS=$(python - <<'PY'
import glob, os, sysconfig
sp = sysconfig.get_paths()["purelib"]
print(":".join(sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))))
PY
)
if [ -n "$NVIDIA_LIB_DIRS" ]; then
    export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
    echo "WARNING: no nvidia-* wheels in the venv; expect CPU" >&2
fi

# The TensorFlow wheels need a newer libstdc++ than some cluster images ship, and the
# import then fails with "GLIBCXX_3.4.30 not found". If that happens, load a newer GCC
# runtime before submitting, e.g. `module load GCC/12.2.0`.

# Keras 3 cannot load the legacy SavedModel with Flipout layers. This must be set before
# TensorFlow is imported, which the scripts also do defensively.
export TF_USE_LEGACY_KERAS=1
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
