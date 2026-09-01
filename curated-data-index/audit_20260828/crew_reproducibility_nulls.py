import numpy as np, glob, json, itertools, re, warnings
warnings.filterwarnings('ignore')
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/Users/jeffreyshorthill/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/59d61f3ce65a6d9863b86d2e96597125219dc754')
P='qwen35b/token_balanced_3chunk'
pj=json.load(open(f'{P}/PROMPTS/domain_expert_probe_3chunk_prompts.json'))
order=pj[0]['source_domains']
maps={'A':order[:12],'B':order[:15],'C':['history','economics','political_science','archaeology','physics','chemistry','biology','medicine','neuroscience','mathematics','statistics','computer_science','software_engineering','cybersecurity','philosophy','comparative_religion','linguistics','psychology','environmental_science']}
segs=json.load(open('/Users/jeffreyshorthill/.claude/jobs/0e9421a7/tmp/segs_raw.json'))
D='qwen35b/domain_probe_60prompt/results'
j60=json.load(open(glob.glob(f'{D}/*20260408T235839Z.json')[0]))
def fd(o):
    if isinstance(o,dict):
        for k,v in o.items():
            if k in('domains','domain_order','domain_names') and isinstance(v,list) and len(v)==20: return v
            r=fd(v)
            if r: return r
    if isinstance(o,list):
        for v in o:
            r=fd(v)
            if r: return r
dom60=fd(j60); z60=np.load(glob.glob(f'{D}/*20260408T235839Z.npz')[0],allow_pickle=True)
G60=z60['generation_trimmed_domain_W']; P60=z60['prefill_domain_W']
topk=lambda v,k=8:set(np.argsort(-v)[:k].tolist()); jac=lambda a,b:len(a&b)/len(a|b)
gen={}; pre={}; gseg={}
for ch in 'ABC':
    z=np.load(glob.glob(f'{P}/results/per_token_*/*_20{ch}_chunk_per_token.npz')[0],allow_pickle=True)
    W=z['generation_mean_W']; PW=z['prefill_mean_W']
    for (a,b),d in zip(segs[ch],maps[ch]): gen[(ch,d)]=W[a:b].mean(0); gseg[(ch,d)]=(a,b)
    mpath=glob.glob(f'{P}/raw/*/domain_expert_probe_20{ch}_chunk/metadata.txt')[0]
    meta=dict(l.rstrip('\n').split('=',1) for l in open(mpath) if '=' in l)
    prompt=meta['prompt']; enc=tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
    n=len(enc['input_ids']); offs=enc['offset_mapping']
    rec=[p for p in pj if p['chunk']==ch][0]; qs=rec['prompt'].replace('\\n','\n').split('\n')
    print(f'chunk {ch}: tokenizer count {n} vs capture {PW.shape[0]}; {len(qs)} questions')
    pos=0; spans=[]
    for q in qs:
        i=prompt.find(q[:40], pos); assert i>=0, q[:40]; j=i+len(q); pos=j
        ta=next(k for k,(s,e) in enumerate(offs) if e>i); tb=next((k for k,(s,e) in enumerate(offs) if s>=j), n)
        spans.append((ta,tb))
    scale=PW.shape[0]/n
    for (ta,tb),d in zip(spans,order):
        a,b=int(round(ta*scale)),int(round(tb*scale)); pre[(ch,d)]=PW[a:b].mean(0)
for i,d in enumerate(dom60): gen[('P',d)]=G60[i]; pre[('P',d)]=P60[i]

def ident(vec, pairs, pool=None):
    h=t=0; mj=[]; mm=[]
    for X,Y in pairs:
        dy=[d for d in order if (Y,d) in vec]
        for d in order:
            if (X,d) not in vec or (Y,d) not in vec: continue
            cand=dy if pool is None else [e for e in pool(Y,d,dy) if e in dy]
            s={e:jac(topk(vec[(X,d)]),topk(vec[(Y,e)])) for e in cand}; b=max(s,key=s.get)
            t+=1; h+=(b==d); mj.append(s[d]); mm+=[v for e,v in s.items() if e!=d]
    return h,t,np.mean(mj),np.mean(mm)
allp=list(itertools.permutations('ABCP',2)); noB=[p for p in allp if 'B' not in p]

print('\n#4 random top-8-of-256 chance Jaccard: analytic', round(0.25/(16-0.25),4), end='; ')
rng=np.random.default_rng(1); sim=[jac(set(rng.choice(256,8,replace=False).tolist()),set(rng.choice(256,8,replace=False).tolist())) for _ in range(20000)]; print('simulated', round(float(np.mean(sim)),4))
h,t,mj,mm=ident(gen,allp); print(f'   generation: matched {mj:.3f}, mismatched {mm:.3f} ({mm/0.016:.1f}x chance)')

print('\n#5 position-matched null (A vs C: same token position, different subject because C reordered)')
same_pos=[]; same_subj=[]; cnt=0
for d in maps['A']:
    a,b=gseg[('A',d)]
    if ('C',d) not in gseg: continue
    cseg=[e for e in maps['C'] if gseg[('C',e)][0]<= (a+b)//2 < gseg[('C',e)][1]]
    if not cseg: continue
    e=cseg[0]
    same_pos.append(jac(topk(gen[('A',d)]),topk(gen[('C',e)]))); same_subj.append(jac(topk(gen[('A',d)]),topk(gen[('C',d)]))); cnt+=1
    print(f'   A:{d:20s} same-position C subject = {e:20s} pos-Jaccard {same_pos[-1]:.3f}   same-subject Jaccard {same_subj[-1]:.3f}')
print(f'   mean same-position(different subject) {np.mean(same_pos):.3f} vs same-subject(different position) {np.mean(same_subj):.3f}; subject wins {sum(s>p for s,p in zip(same_subj,same_pos))}/{cnt}, ties {sum(s==p for s,p in zip(same_subj,same_pos))}')

print('\n#6 adjacent-segment null: candidate pool = {previous, same, next} subject in target order')
def adj(Y,d,dy):
    o=maps.get(Y,dom60); i=o.index(d); return [o[k] for k in (i-1,i,i+1) if 0<=k<len(o)]
h,t,mj,mm=ident(gen,allp,adj); print(f'   all pairs: {h}/{t} = {h/t:.2f} (chance ~1/3); matched {mj:.3f} vs adjacent {mm:.3f}')
h,t,mj,mm=ident(gen,noB,adj); print(f'   excl. B : {h}/{t} = {h/t:.2f}')

print('\n#7 PREFILL control: same identification on the prefill rows of each question')
for name,pp in (('all pairs',allp),('excl. B',noB),('A<->P',[('A','P'),('P','A')]),('C<->P',[('C','P'),('P','C')]),('A<->C',[('A','C'),('C','A')])):
    hg,tg,mjg,mmg=ident(gen,pp); hp,tp,mjp,mmp=ident(pre,pp)
    print(f'   {name:10s} generation {hg:3d}/{tg} = {hg/tg:.2f} (matched {mjg:.3f} / mism {mmg:.3f})   prefill {hp:3d}/{tp} = {hp/tp:.2f} (matched {mjp:.3f} / mism {mmp:.3f})')
rng=np.random.default_rng(7); null=[]
for _ in range(100):
    h=t=0
    for X,Y in allp:
        dy=[d for d in order if (Y,d) in pre]; perm=dict(zip(dy,rng.permutation(dy)))
        for d in order:
            if (X,d) not in pre or (Y,d) not in pre: continue
            s={e:jac(topk(pre[(X,d)]),topk(pre[(Y,perm[e])])) for e in dy}; t+=1; h+=(max(s,key=s.get)==d)
    null.append(h/t)
print(f'   prefill label-shuffle null: mean {np.mean(null):.3f}, 95th {np.percentile(null,95):.3f}')
# prefill within-chunk between-question Jaccard and E224 presence
for ch in 'ABC':
    ds=[d for d in maps[ch]]; print(f'   prefill between-question Jaccard chunk {ch}: {np.mean([jac(topk(pre[(ch,x)]),topk(pre[(ch,y)])) for x,y in itertools.combinations(ds,2)]):.3f}; 224 in top-8 of {sum(224 in topk(pre[(ch,d)]) for d in ds)}/{len(ds)} questions')

# --- added 2026-08-29: seeded generation label-shuffle null (200 draws) so the paper can cite seed + draws
rng=np.random.default_rng(11); gnull=[]
for _ in range(200):
    h=t=0
    for X,Y in allp:
        dy=[d for d in order if (Y,d) in gen]; perm=dict(zip(dy,rng.permutation(dy)))
        for d in order:
            if (X,d) not in gen or (Y,d) not in gen: continue
            s={e:jac(topk(gen[(X,d)]),topk(gen[(Y,perm[e])])) for e in dy}; t+=1; h+=(max(s,key=s.get)==d)
    gnull.append(h/t)
print(f'   GENERATION label-shuffle null (seed 11, 200 draws): mean {np.mean(gnull):.3f}, 95th {np.percentile(gnull,95):.3f}, max {np.max(gnull):.3f}')

# --- added 2026-08-30: per-pair denominators (Methods table) and exact adjacency chance
print('\nPER-PAIR TABLE (generation): X->Y  shared  hits  (prefill: hits/40)')
tot=0
for X,Y in allp:
    hg,tg,_,_=ident(gen,[(X,Y)]); hp,tp,_,_=ident(pre,[(X,Y)]); tot+=tg
    print(f'   {X}->{Y}  shared {tg:2d}  gen {hg:2d}/{tg:2d}  prefill {hp:2d}/{tp}')
print('   total generation queries', tot)
# adjacency: exact chance = mean over queries of 1/|candidate pool|; count edge queries
inv=[]; edges=0; n=0
for X,Y in allp:
    dy=[d for d in order if (Y,d) in gen]
    for d in order:
        if (X,d) not in gen or (Y,d) not in gen: continue
        cand=[e for e in adj(Y,d,dy) if e in dy]; inv.append(1/len(cand)); edges+=(len(cand)<3); n+=1
print(f'   adjacency: {n} queries, {edges} edge queries with 2 candidates, exact chance = {np.mean(inv):.3f}')
