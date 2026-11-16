#!/usr/bin/env bash
# Lightweight stand-in for `js2 setup`, run ON the GPU machine.
# Why: the class-wide volume quota was full when we launched (990/1000 GB), so the instance has no
# /workspace volume. `js2 setup` installs vLLM (>10 GB), which does not fit the root disk; we only
# need torch + transformers. Everything (venv, model cache, data, results) lives on the root disk.
set -euo pipefail
MOUNT=/workspace
sudo mkdir -p "$MOUNT" && sudo chown "$USER:$USER" "$MOUNT"
mkdir -p "$MOUNT/experiments"

# Course staff key, exactly as `js2 setup` does (lets TAs help and shelve idle instances).
STAFF_KEY_URL="https://raw.githubusercontent.com/Saipraneet/jetstream2-guide/main/staff/staff_key.pub"
if k=$(curl -fsSL "$STAFF_KEY_URL" 2>/dev/null) && [ -n "$k" ]; then
  mkdir -p ~/.ssh; chmod 700 ~/.ssh; touch ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys
  grep -qxF "$k" ~/.ssh/authorized_keys || { echo "$k" >> ~/.ssh/authorized_keys; echo "added course staff SSH key"; }
fi

cat > "$MOUNT/env.sh" <<EOT
export HF_HOME=$MOUNT/hf
export HF_HUB_DISABLE_XET=1
export PATH="\$HOME/.local/bin:\$PATH"
source $MOUNT/venv/bin/activate
EOT

export PATH="$HOME/.local/bin:$PATH" UV_CACHE_DIR="$MOUNT/.uv-cache"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; }
if [ ! -x "$MOUNT/venv/bin/python" ]; then
  uv venv "$MOUNT/venv" --python 3.12 -q
fi
source "$MOUNT/venv/bin/activate"
uv pip install -q torch transformers accelerate scipy scikit-learn matplotlib pandas sentencepiece protobuf
python - <<'EOT'
import torch, transformers
print(f"SETUP_OK torch={torch.__version__} transformers={transformers.__version__} "
      f"cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")
EOT
df -h --output=target,size,avail / /dev/shm
free -g | head -n 2
