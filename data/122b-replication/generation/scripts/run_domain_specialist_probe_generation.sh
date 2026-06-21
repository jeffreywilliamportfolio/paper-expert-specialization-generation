#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/consciousness-experiment/experiments/qwen3.5-122B-A10B-huahua-baseline/followups/domain-specialist-generation}"
CAPTURE_BINARY="${CAPTURE_BINARY:-/workspace/consciousness-experiment/capture_activations}"
MODEL_PATH="${MODEL_PATH:-/workspace/models/qwen122-hauhau-q8kp/Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf}"
PROMPT_TSV="${PROMPT_TSV:-$ROOT_DIR/PROMPTS/domain_specialist_probe_60_no_think.tsv}"
LLAMA_BUILD_BIN="${LLAMA_BUILD_BIN:-/workspace/llama.cpp.new/build/bin}"

RUN_LABEL="${RUN_LABEL:-qwen122_domain_specialist_probe_60}"
RAW_DIR="${RAW_DIR:-$ROOT_DIR/raw}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT_DIR/RESULTS}"
N_PREDICT="${N_PREDICT:-2048}"
NGL="${NGL:-999}"
CTX="${CTX:-16384}"
THREADS="${THREADS:-16}"
FLASH_ATTN="${FLASH_ATTN:-on}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
SEED="${SEED:-42}"
MODEL_SHA256="${MODEL_SHA256:-}"

mkdir -p "$RAW_DIR" "$RESULTS_DIR"
export LD_LIBRARY_PATH="$LLAMA_BUILD_BIN:${LD_LIBRARY_PATH:-}"

now="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="${now}_${RUN_LABEL}_gen_n${N_PREDICT}"
out_dir="$RAW_DIR/$run_id"
log_path="$RESULTS_DIR/${run_id}_capture.log"
cmd_path="$RESULTS_DIR/${run_id}_command.sh"
meta_path="$RESULTS_DIR/${run_id}_run_metadata.json"

mkdir -p "$out_dir"

{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "export LD_LIBRARY_PATH=$LLAMA_BUILD_BIN:\${LD_LIBRARY_PATH:-}"
  printf '%q' "$CAPTURE_BINARY"
  printf ' -m %q --prompt-file %q -o %q' "$MODEL_PATH" "$PROMPT_TSV" "$out_dir"
  printf ' -n %q -ngl %q -c %q -t %q -fa %q --cache-type-k %q --cache-type-v %q --seed %q --temp 0 --top-k 1 --top-p 1 --min-p 0 --repeat-penalty 1 --mirostat 0 --routing-only --no-stream\n' \
    "$N_PREDICT" "$NGL" "$CTX" "$THREADS" "$FLASH_ATTN" "$CACHE_TYPE_K" "$CACHE_TYPE_V" "$SEED"
} > "$cmd_path"
chmod +x "$cmd_path"

python3 - <<PY
import json, hashlib, pathlib, subprocess
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
binary = pathlib.Path("$CAPTURE_BINARY")
model = pathlib.Path("$MODEL_PATH")
cmd = pathlib.Path("$cmd_path").read_text().strip()
version = subprocess.run([str(binary), "--version"], capture_output=True, text=True).stdout.strip()
meta = {
    "run_id": "$run_id",
    "capture_binary": str(binary),
    "capture_binary_sha256": sha256(binary),
    "capture_binary_version": version,
    "model_path": str(model),
    "model_sha256": "$MODEL_SHA256" if "$MODEL_SHA256" else None,
    "model_size_bytes": model.stat().st_size,
    "prompt_tsv": "$PROMPT_TSV",
    "output_dir": "$out_dir",
    "n_predict": int("$N_PREDICT"),
    "seed": int("$SEED"),
    "routing_only": True,
    "command_script": "$cmd_path",
    "command": cmd,
}
pathlib.Path("$meta_path").write_text(json.dumps(meta, indent=2) + "\\n")
PY

echo "=== Running $run_id ==="
echo "Command: $cmd_path"
echo "Metadata: $meta_path"
bash "$cmd_path" | tee "$log_path"
