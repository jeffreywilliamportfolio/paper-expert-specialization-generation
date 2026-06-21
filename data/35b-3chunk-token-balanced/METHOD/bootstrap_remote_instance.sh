#!/usr/bin/env bash
set -euo pipefail

LLAMA_REPO_URL="${LLAMA_REPO_URL:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_DIR="${LLAMA_DIR:-/workspace/llama.cpp.new}"
LLAMA_BUILD_DIR="${LLAMA_BUILD_DIR:-$LLAMA_DIR/build}"
LLAMA_PINNED_COMMIT="${LLAMA_PINNED_COMMIT:-1772701f9}"

REMOTE_EXPERIMENT_DIR="${REMOTE_EXPERIMENT_DIR:-/workspace/consciousness-experiment/experiments/qwen3.5-35b-a3b-huahua-strangeloop}"
CAPTURE_SOURCE="${CAPTURE_SOURCE:-$REMOTE_EXPERIMENT_DIR/scripts/capture_activations.cpp}"
CAPTURE_DIR="${CAPTURE_DIR:-$LLAMA_DIR/examples/capture_activations}"
CAPTURE_BINARY_DEST="${CAPTURE_BINARY_DEST:-/workspace/consciousness-experiment/capture_activations}"

MODEL_REPO="${MODEL_REPO:-HauhauCS/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive}"
MODEL_FILE="${MODEL_FILE:-Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf}"
MODEL_DIR="${MODEL_DIR:-/workspace/models/qwen35-hauhau-q8}"
HF_HOME="${HF_HOME:-/workspace/.hf}"

wait_for_other_git() {
  while pgrep -f "git clone .*${LLAMA_DIR}|git -C ${LLAMA_DIR} checkout|git -c fetch.negotiationAlgorithm=noop fetch origin" >/dev/null; do
    sleep 5
  done
}

ensure_workspace() {
  mkdir -p /workspace/src "$MODEL_DIR" /workspace/consciousness-experiment "$HF_HOME"
}

ensure_packages() {
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y git build-essential cmake ninja-build pkg-config python3-pip
}

ensure_hf_cli() {
  python3 -m pip install -U "huggingface_hub[cli]"
}

ensure_llama_checkout() {
  if [[ ! -d "$LLAMA_DIR/.git" ]]; then
    git clone --filter=blob:none --no-checkout "$LLAMA_REPO_URL" "$LLAMA_DIR"
  fi

  wait_for_other_git
  git -C "$LLAMA_DIR" checkout "$LLAMA_PINNED_COMMIT"
}

install_capture_target() {
  if [[ ! -f "$CAPTURE_SOURCE" ]]; then
    echo "missing capture source: $CAPTURE_SOURCE" >&2
    exit 1
  fi

  mkdir -p "$CAPTURE_DIR"
  cp "$CAPTURE_SOURCE" "$CAPTURE_DIR/capture_activations.cpp"

  cat > "$CAPTURE_DIR/CMakeLists.txt" <<'EOF'
set(TARGET llama-capture-activations)
add_executable(${TARGET} capture_activations.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
EOF

  python3 - <<'PY'
from pathlib import Path

path = Path("/workspace/llama.cpp.new/examples/CMakeLists.txt")
text = path.read_text()
line = "    add_subdirectory(capture_activations)\n"
if "add_subdirectory(capture_activations)" not in text:
    needle = "    add_subdirectory(gen-docs)\n"
    if needle in text:
        text = text.replace(needle, needle + line, 1)
    else:
        text += "\nadd_subdirectory(capture_activations)\n"
    path.write_text(text)
PY
}

build_capture_binary() {
  cmake -S "$LLAMA_DIR" -B "$LLAMA_BUILD_DIR" \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_EXAMPLES=ON

  cmake --build "$LLAMA_BUILD_DIR" --target llama-capture-activations -j"$(nproc)"

  cp "$LLAMA_BUILD_DIR/bin/llama-capture-activations" "$CAPTURE_BINARY_DEST"
  chmod +x "$CAPTURE_BINARY_DEST"
}

download_model() {
  export HF_HOME

  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is not set" >&2
    exit 1
  fi

  if ! hf auth whoami >/dev/null 2>&1; then
    hf auth login --token "$HF_TOKEN"
  fi

  hf download "$MODEL_REPO" "$MODEL_FILE" --local-dir "$MODEL_DIR"
}

print_summary() {
  echo "--- PINNED COMMIT ---"
  git -C "$LLAMA_DIR" rev-parse HEAD
  echo "--- CAPTURE VERSION ---"
  "$CAPTURE_BINARY_DEST" --version || true
  echo "--- CAPTURE SHA256 ---"
  sha256sum "$CAPTURE_BINARY_DEST"
  echo "--- MODEL FILE ---"
  ls -lh "$MODEL_DIR/$MODEL_FILE"
}

main() {
  ensure_workspace
  ensure_packages
  ensure_hf_cli
  ensure_llama_checkout
  install_capture_target
  build_capture_binary
  download_model
  print_summary
}

main "$@"
