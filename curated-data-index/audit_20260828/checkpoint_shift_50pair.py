# Official vs HauhauCS routing shift on matched prompts (50-pair run + medical), with corrected floors:
# same-type = all pairs of different prompts within the same prompt type (same checkpoint);
# all-pairs = all pairs of different prompts. Added 2026-08-30 (first version used consecutive ids, which
# alternated matched AAVE/AE pairs and understated the shift; audit 2026-08-30).
import numpy as np, json, itertools, collections
def load(idx):
    return {json.loads(l)['cell']:np.load(json.loads(l)['npz'],allow_pickle=True) for l in open(idx)}
top=lambda v,k=8:set(np.argsort(-v)[:k].tolist()); jac=lambda a,b:len(a&b)/len(a|b)
pool=lambda z,k:np.nanmean(z[k],0)
for name,idx,split in (('50-pair','qwen35b/controls/compact/aave_5-5_register_run/INDEX.jsonl',('run_base_nothink','run_hauhau_nothink')),
                       ('medical nothink','qwen35b/controls/compact/aave_5-15_medical/INDEX.jsonl',('run_base_nothink','run_hauhau_nothink'))):
    cells=load(idx)
    base={k.split(split[0]+'/')[1]:v for k,v in cells.items() if split[0]+'/' in k}
    hh={k.split(split[1]+'/')[1]:v for k,v in cells.items() if split[1]+'/' in k}
    common=sorted(set(base)&set(hh)); print(f'\n== {name}: {len(common)} matched prompts')
    for blk,key in (('prefill','prefill_W'),('generation','gentrim_W')):
        sets={p:top(pool(base[p],key)) for p in common}
        same=[jac(sets[p],top(pool(hh[p],key))) for p in common]
        # type = prompt id prefix before the last underscore (e.g. '001_aave' -> type unknown here); use metadata if present
        typ={}
        for p in common:
            m=base[p]['prompt_id'].item() if 'prompt_id' in base[p].files else p
            typ[p]=str(m).split('_')[0] if isinstance(m,str) else p
        st=[jac(sets[a],sets[b]) for a,b in itertools.combinations(common,2) if typ[a]==typ[b]]
        ap=[jac(sets[a],sets[b]) for a,b in itertools.combinations(common,2)]
        print(f'  {blk:10s} same prompt official vs HauhauCS {np.mean(same):.3f} (min {np.min(same):.2f}) | floors: same-type {np.mean(st) if st else float("nan"):.3f} (n={len(st)}), all pairs {np.mean(ap):.3f}')
