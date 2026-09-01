import json,glob,numpy as np,itertools
EOG={248044,248045,248046}
def wsq(logits):
    x=logits-logits.max(1,keepdims=True); p=np.exp(x); p/=p.sum(1,keepdims=True)
    idx=np.argsort(-p,1)[:,:8]; W=np.zeros_like(p)
    for i in range(len(p)):
        sel=p[i,idx[i]]; W[i,idx[i]]=sel/sel.sum()
    return W
cells={}
for d in sorted(glob.glob('T*')):
    npr=len(json.load(open(d+'/prompt_tokens.json'))); gt=[t['token_id'] for t in json.load(open(d+'/generated_tokens.json'))]
    hit=[i for i,t in enumerate(gt) if t in EOG]; trim=hit[0] if hit else len(gt)
    P=[];G=[]
    for L in range(40):
        a=np.load(f'{d}/router/ffn_moe_logits-{L}.npy'); W=wsq(a)
        if a.shape[0]==npr+len(gt): P.append(W[:npr].mean(0)); G.append(W[npr:npr+trim].mean(0))
        else: G.append(W[1:1+trim].mean(0))
    cells[d]=dict(P=np.mean(P,0),G=np.mean(G,0)); print(d,npr,len(gt),'trim',trim)
names=list(cells); dom=lambda n:n.split('_')[1]
def top(v,k=8): return set(np.argsort(-v)[:k].tolist())
def jac(a,b): return len(a&b)/len(a|b)
for blk in ('P','G'):
    win=[];bet=[]
    for a,b in itertools.combinations(names,2):
        j=jac(top(cells[a][blk]),top(cells[b][blk])); (win if dom(a)==dom(b) else bet).append((a,b,j))
    print(f'\n[{blk}] within-domain pairs:'); [print(f'  {a[:12]:12s} {b[:12]:12s} {j:.3f}') for a,b,j in win]
    print(f'[{blk}] within mean {np.mean([j for *_,j in win]):.3f}  between mean {np.mean([j for *_,j in bet]):.3f}  (n={len(win)}/{len(bet)})')
    hits=sum(dom(max((b for b in names if b!=a),key=lambda b:jac(top(cells[a][blk]),top(cells[b][blk]))))==dom(a) for a in names)
    print(f'[{blk}] nearest-neighbour same domain: {hits}/{len(names)}  (chance 1/11)')
    print(f'[{blk}] winners:',{n[:7]:int(np.argmax(cells[n][blk])) for n in names})
