# GLM-4.7-Flash register batteries (prefill only): pooled top-4 prefill/generation sets per battery and
# checkpoint; E10 rank. Run from repo root. Added 2026-08-30 (numbers first computed inline 2026-08-29 13:05).
import numpy as np, json, collections
R='glm47flash/compact/register_run/'
cells=[json.loads(l) for l in open(R+'INDEX.jsonl')]
by=collections.defaultdict(list)
for c in cells:
    z=np.load(R+c['npz'].split('register_run/')[-1],allow_pickle=True)
    by[c['cell'].split('/')[2]].append((np.nanmean(z['prefill_W'],0),np.nanmean(z['gentrim_W'],0)))
top=lambda v,k=4:set(np.argsort(-v)[:k].tolist()); jac=lambda a,b:len(a&b)/len(a|b)
pooled={t:(np.mean([p for p,g in v],0),np.mean([g for p,g in v],0)) for t,v in by.items()}
for t,(P,G) in pooled.items():
    print(f'{t:28s} n={len(by[t]):2d} prefill top-4 {sorted(top(P))} E10 rank {int((P>P[10]).sum())+1:2d} | generation top-4 {sorted(top(G))}')
a,b=pooled['explore_base_q8_boxB'][0],pooled['explore_hauhau_q8_boxB'][0]
print('prefill top-4 Jaccard, official vs HauhauCS, main battery:', jac(top(a),top(b)))
