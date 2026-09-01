#!/usr/bin/env python3
"""Build the curated expert-specialization data repo from the ExternalSSD archive.

Read-only on every source. Writes only under REPO. Nothing on the drive is deleted.
Three actions per source set:
  copy    -> rsync the set (with excludes) into REPO/<dst>
  compact -> per cell, per layer W/S/Q for prefill / generation / generation_trimmed,
             written as one small .npz per cell; raw tensors NOT copied, but every raw
             file's sha256 + source path is recorded in raw_manifest.tsv
  select  -> HVAC only: copy raw for the 18 distinct prompts (of 180 byte-identical repeats)
Run: python3 build_repo.py  (log to build.log)
"""
import os, sys, json, hashlib, subprocess, glob, re, time
import numpy as np

SSD = "/Volumes/ExternalSSD"
REPO = f"{SSD}/expert-specialization-data"
LOG = open(f"{REPO}/build.log", "a")
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()

def sha256(path, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()

def rsync(src, dst, excludes=()):
    os.makedirs(dst, exist_ok=True)
    cmd = ["rsync", "-a", "--no-perms", "--no-owner", "--no-group"]
    for e in excludes:
        cmd += ["--exclude", e]
    cmd += [src.rstrip("/") + "/", dst.rstrip("/") + "/"]
    log("rsync", src, "->", dst, "excl", list(excludes))
    subprocess.run(cmd, check=True)

# ----------------------------------------------------------------------------- routing
QWEN_EOG = {248044, 248045, 248046}
GLM_EOG = {154827, 154828, 154820, 154829}

def qwen_weights(logits):
    x = logits.astype(np.float64)
    x = x - x.max(1, keepdims=True)
    p = np.exp(x); p /= p.sum(1, keepdims=True)
    idx = np.argsort(-p, 1)[:, :8]
    w = np.zeros_like(p)
    rows = np.arange(p.shape[0])[:, None]
    top = p[rows, idx]
    w[rows, idx] = top / top.sum(1, keepdims=True)
    return w

def glm_weights(logits, bias):
    s = 1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -30, 30)))
    idx = np.argsort(-(s + bias[None, :]), 1)[:, :4]
    w = np.zeros_like(s)
    rows = np.arange(s.shape[0])[:, None]
    top = s[rows, idx]
    w[rows, idx] = top / top.sum(1, keepdims=True)
    return w

def wsq(w):
    """w: [n_tok, n_exp] renormalized weights (0 where unselected)."""
    if w.shape[0] == 0:
        n = w.shape[1]
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    sel = w > 0
    W = w.mean(0)
    S = sel.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        Q = np.where(sel.sum(0) > 0, w.sum(0) / np.maximum(sel.sum(0), 1), 0.0)
    return W, S, Q

def read_meta(cell):
    m = {}
    with open(os.path.join(cell, "metadata.txt"), errors="replace") as f:
        for line in f:
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                m[k] = v
    return m

def gen_ids(cell):
    p = os.path.join(cell, "generated_tokens.json")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8", errors="replace") as f:
        g = json.load(f)
    if isinstance(g, dict):
        g = g.get("tokens", g.get("generated_tokens", []))
    return [t["token_id"] if isinstance(t, dict) else int(t) for t in g]

def compact_cell(cell, model, bias=None, raw_manifest=None, set_name=""):
    meta = read_meta(cell)
    n_prompt = int(meta.get("n_tokens_prompt", 0))
    n_gen = int(meta.get("n_tokens_generated", 0))
    ids = gen_ids(cell)
    eog = QWEN_EOG if model == "qwen" else GLM_EOG
    trim = next((i for i, t in enumerate(ids) if t in eog), None)
    files = sorted(glob.glob(os.path.join(cell, "router", "ffn_moe_logits-*.npy")),
                   key=lambda p: int(re.search(r"-(\d+)\.npy$", p).group(1)))
    layers = [int(re.search(r"-(\d+)\.npy$", p).group(1)) for p in files]
    n_exp = 256 if model == "qwen" else 64
    L = (max(layers) + 1) if layers else 0
    out = {k: np.full((L, n_exp), np.nan) for k in
           ("prefill_W", "prefill_S", "prefill_Q", "gen_W", "gen_S", "gen_Q", "gentrim_W", "gentrim_S", "gentrim_Q")}
    rows = np.zeros(L, dtype=np.int64)
    note = []
    for p, l in zip(files, layers):
        if raw_manifest is not None:
            raw_manifest.write(f"{set_name}\t{os.path.relpath(p, SSD)}\t{os.path.getsize(p)}\t{sha256(p)}\n")
        arr = np.load(p)
        rows[l] = arr.shape[0]
        w = qwen_weights(arr) if model == "qwen" else glm_weights(arr, bias[str(l)])
        if arr.shape[0] == n_prompt + n_gen:
            pre, gen = w[:n_prompt], w[n_prompt:]
        elif arr.shape[0] == n_gen + 1 and n_gen > 0:
            pre, gen = w[:0], w[1:]          # last-layer quirk: 1 prefill row (last prompt token) + gen
            note.append(f"L{l}:lastlayer_quirk")
        elif arr.shape[0] == 1 and n_gen == 0:
            pre, gen = w[:0], w[:0]
            note.append(f"L{l}:lastlayer_prefill_only_row")
        elif arr.shape[0] == n_prompt and n_gen == 0:
            pre, gen = w, w[:0]
        else:
            note.append(f"L{l}:rows{arr.shape[0]}_unexpected")
            pre, gen = w[:min(n_prompt, arr.shape[0])], w[min(n_prompt, arr.shape[0]):]
        for tag, blk in (("prefill", pre), ("gen", gen), ("gentrim", gen[:trim] if trim is not None else gen)):
            W, S, Q = wsq(blk)
            out[f"{tag}_W"][l], out[f"{tag}_S"][l], out[f"{tag}_Q"][l] = W, S, Q
    out.update(dict(n_prompt=n_prompt, n_gen=n_gen, trim_index=-1 if trim is None else trim,
                    rows_per_layer=rows, layers=np.array(layers), model=model,
                    reconstruction=("softmax256_top8_renorm" if model == "qwen" else "sigmoid_bias_top4_sigmoidrenorm"),
                    source=os.path.relpath(cell, SSD), notes="|".join(note),
                    prompt_id=meta.get("prompt_id", os.path.basename(cell))))
    return out

def compact_set(set_name, cell_roots, model, dst, bias_path=None, raw_manifest=None):
    """cell_roots: list of glob patterns whose matches are cell dirs (contain router/)."""
    bias = None
    if bias_path:
        z = np.load(bias_path); bias = {k: z[k] for k in z.files}
    cells = []
    for pat in cell_roots:
        cells += [c for c in sorted(glob.glob(pat)) if os.path.isdir(os.path.join(c, "router"))]
    os.makedirs(dst, exist_ok=True)
    log(f"compact {set_name}: {len(cells)} cells -> {dst}")
    index = []
    for i, c in enumerate(cells):
        rel = os.path.relpath(c, SSD)
        name = re.sub(r"[^A-Za-z0-9_.-]+", "__", rel)[-180:]
        outp = os.path.join(dst, name + ".npz")
        if os.path.exists(outp):
            continue
        d = compact_cell(c, model, bias, raw_manifest, set_name)
        np.savez_compressed(outp, **d)
        index.append(dict(cell=rel, npz=os.path.relpath(outp, REPO), n_prompt=int(d["n_prompt"]),
                          n_gen=int(d["n_gen"]), trim_index=int(d["trim_index"]), notes=d["notes"],
                          prompt_id=d["prompt_id"]))
        if i % 25 == 0:
            log(f"  {set_name} {i+1}/{len(cells)}")
    with open(os.path.join(dst, "INDEX.jsonl"), "a") as f:
        for r in index:
            f.write(json.dumps(r) + "\n")

# ----------------------------------------------------------------------------- plan
A = f"{SSD}/moe-routing-organized/qwen3.5-35b-a3b-and-huahua/35B"
COPY = [
    # Tier 1: load-bearing, small
    (f"{A}/qwen-huahua-expert-identification", "qwen35b/domain_probe_60prompt", ()),
    (f"{SSD}/paper-expert-specialization-generation/data/35b-3chunk-token-balanced", "qwen35b/token_balanced_3chunk", ()),
    (f"{SSD}/git-updates-moe/qwen3.5-35b-a3b-huahua-philosophy-experts-bias", "qwen35b/philosophy_experts_bias_causal",
        ("*.tar", "*.zip")),
    (f"{A}/qwen35b-a3b-vs-hauhaucs-uncensored-run1", "qwen35b/base_vs_hauhau_prefill", ()),
    (f"{A}/qwen-huahua-6cond-hvac", "qwen35b/controls/hvac_6cond_results", ()),
    (f"{SSD}/paper-expert-specialization-generation/data/122b-replication", "qwen122b/domain_probe_60prompt", ()),
    (f"{SSD}/moe-routing-organized/qwen3.5-122b-a10b-huahua/qwen3.5-122B-A10B-huahua-five-cond-experience-probe", "qwen122b/five_cond_experience_probe", ()),
    (f"{SSD}/moe-routing-organized/qwen3.5-122b-a10b-huahua/qwen3.5-122B-A10B-huahua-single-prompt-processing-hum", "qwen122b/single_prompt_hum", ()),
    (f"{SSD}/moe-routing-organized/qwen3.5-122b-a10b-huahua/qwen3.5-122B-A10B-huahua-six-cond-hvac", "qwen122b/six_cond_hvac_E114only", ()),
    (f"{SSD}/moe-routing-organized/qwen3.5-122b-a10b-huahua/qwen3.5-122B-A10B-huahua-baseline/followups", "qwen122b/baseline_followups_per_token", ("*.zip",)),
    (f"{SSD}/glm47-flash-domain-routing/raw", "glm47flash/domain_exploratory_raw_prefill", ()),
    (f"{SSD}/glm47-flash-domain-routing/raw_powered", "glm47flash/domain_powered_raw_prefill", ()),
    (f"{SSD}/glm47-flash-domain-routing/analysis", "glm47flash/analysis", ()),
    (f"{SSD}/glm47-flash-domain-routing/register_run", "glm47flash/register_run_meta",
        ("explore_*", "pulled_*", "kae_qwen_*")),
    (f"{SSD}/aave-registers-cleaned/analysis_all_models_expert_routing", "qwen35b/controls/aave_routing_tables", ()),
    # Tier 2: raw controls, moderate size, copied whole
    (f"{A}/l1l3_a_only_hauhau", "qwen35b/controls/l1l3_register_hauhau_raw", ()),
    (f"{A}/l1l3_a_only_vanilla", "qwen35b/controls/l1l3_register_base_raw", ()),
    (f"{A}/qwen3.5-35b-a3b-huahua-five-cond-experience-probe", "qwen35b/controls/five_cond_experience_probe_raw", ()),
]
GLM_DOCS = ["README.md", "RESULTS.md", "model_card.md", "model_info.md", "domain_battery.json",
            "domain_battery_powered.json", "fire_battery.json", "FIREbank_blockF_v1.tsv"]

COMPACT = [
    ("aave_5-5_register_run", "qwen",
        [f"{SSD}/aave-registers-cleaned/5-5-26_initial_50_pair_register_run/runs/*/*/*"],
        "qwen35b/controls/compact/aave_5-5_register_run", None),
    ("aave_5-15_medical", "qwen",
        [f"{SSD}/aave-registers-cleaned/5-15-26/aave-register-medical/runs/*/*",
         f"{SSD}/aave-registers-cleaned/5-15-26/aave-register-medical/chest_pain_plus/runs/*/*",
         f"{SSD}/aave-registers-cleaned/5-15-26/aave-register-medical/financial_stress_pair/runs/*/*"],
        "qwen35b/controls/compact/aave_5-15_medical", None),
    ("hvac_6cond_l1l3_all180", "qwen",
        [f"{SSD}/hvac_cal_water_treatment_6cond_l1l3_hauhau/*"],
        "qwen35b/controls/compact/hvac_6cond_l1l3_all180", None),
    ("glm_register_run", "glm",
        [f"{SSD}/glm47-flash-domain-routing/register_run/explore_*/*"],
        "glm47flash/compact/register_run", f"{SSD}/glm47-flash-domain-routing/register_run/base_bias.npz"),
]

def hvac_select():
    """Copy raw for one cell per distinct prompt (18 of 180)."""
    root = f"{SSD}/hvac_cal_water_treatment_6cond_l1l3_hauhau"
    dst = f"{REPO}/qwen35b/controls/hvac_6cond_l1l3_distinct18_raw"
    seen = {}
    for c in sorted(glob.glob(root + "/*")):
        if not os.path.isdir(os.path.join(c, "router")):
            continue
        h = sha256(os.path.join(c, "metadata.txt"))
        # metadata contains prompt_id which differs per repeat; hash the prompt text instead
        m = read_meta(c); h = hashlib.sha256(m.get("prompt", "").encode()).hexdigest()
        seen.setdefault(h, []).append(c)
    log(f"hvac: {len(seen)} distinct prompts among {sum(len(v) for v in seen.values())} cells")
    os.makedirs(dst, exist_ok=True)
    with open(f"{dst}/REPEATS.tsv", "w") as f:
        f.write("kept_cell\tprompt_sha256\tn_identical_cells\tall_cells\n")
        for h, cells in seen.items():
            keep = cells[0]
            rsync(keep, os.path.join(dst, os.path.basename(keep)))
            f.write(f"{os.path.basename(keep)}\t{h}\t{len(cells)}\t{','.join(os.path.basename(x) for x in cells)}\n")

def main():
    os.makedirs(REPO, exist_ok=True)
    log("=== build start")
    for src, dst, ex in COPY:
        if not os.path.exists(src):
            log("MISSING", src); continue
        rsync(src, os.path.join(REPO, dst), ex)
    for d in GLM_DOCS:
        p = f"{SSD}/glm47-flash-domain-routing/{d}"
        if os.path.exists(p):
            os.makedirs(f"{REPO}/glm47flash", exist_ok=True)
            subprocess.run(["cp", "-p", p, f"{REPO}/glm47flash/{d}"], check=True)
    # 122B domain-run copies from moe-routing-organized carry the analysis scripts; paper data has them too. skip.
    hvac_select()
    rm = open(f"{REPO}/raw_manifest.tsv", "a")
    if os.path.getsize(f"{REPO}/raw_manifest.tsv") == 0:
        rm.write("set\tsource_path_rel_ExternalSSD\tbytes\tsha256\n")
    for name, model, pats, dst, bias in COMPACT:
        compact_set(name, pats, model, os.path.join(REPO, dst), bias, rm)
    rm.close()
    # repo-wide manifest
    log("hashing repo")
    with open(f"{REPO}/MANIFEST.sha256", "w") as f:
        for dp, dn, fn in os.walk(REPO):
            if "/.git" in dp: continue
            for x in sorted(fn):
                p = os.path.join(dp, x)
                if x in ("MANIFEST.sha256", "build.log"): continue
                f.write(f"{sha256(p)}  {os.path.relpath(p, REPO)}\n")
    log("=== build done")

if __name__ == "__main__":
    main()
