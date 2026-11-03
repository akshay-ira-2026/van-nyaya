import os, time, json, math, base64, urllib.request, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
T0=time.time(); torch.set_num_threads(os.cpu_count()); torch.manual_seed(0)
M=os.environ.get('MODEL','1-800-LLMs/tiny-aya-global'); DATA=os.environ['DATA']; RUN=os.environ.get('RUN_NAME','dpo')
BUDGET=float(os.environ.get('BUDGET_S','18600')); ACC=int(os.environ.get('ACC','4')); LR=float(os.environ.get('LR','2e-5')); MAXLEN=int(os.environ.get('MAXLEN','1280')); BETA=float(os.environ.get('BETA','0.1'))
OUT='out_adapter'; os.makedirs(OUT,exist_ok=True)
tok=AutoTokenizer.from_pretrained(M)
def ids_of(m,**k):
    o=tok.apply_chat_template(m,tokenize=True,**k); return o if isinstance(o,list) else o['input_ids']
rows=[json.loads(l) for l in open(DATA)]; N=len(rows)
model=AutoModelForCausalLM.from_pretrained(M,torch_dtype=torch.float32,low_cpu_mem_usage=True)
if os.path.exists('sft_adapter/adapter_config.json'):
    model=PeftModel.from_pretrained(model,'sft_adapter').merge_and_unload(); print('merged sft adapter (reference = sft model)',flush=True)
model.config.use_cache=False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False}); model.enable_input_require_grads()
names={n.split('.')[-1] for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)}
TM=[x for x in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'] if x in names]
if os.path.exists('resume/adapter_config.json'):
    model=PeftModel.from_pretrained(model,'resume',is_trainable=True); st=json.load(open('resume/state.json'))
else:
    model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.0,target_modules=TM,task_type='CAUSAL_LM'))
    st={'next':0,'opt_step':0,'seen':0,'skipped':0,'log':[],'runs':0}
st['runs']+=1
params=[p for p in model.parameters() if p.requires_grad]
opt=torch.optim.AdamW(params,lr=LR,weight_decay=0.0)
if os.path.exists('resume/opt.pt'): opt.load_state_dict(torch.load('resume/opt.pt'))
TOT=math.ceil(N/ACC)
def lr_at(s):
    if s<5: return LR*(s+1)/5
    p=min(1,(s-5)/max(1,TOT-5)); return LR*(0.1+0.9*0.5*(1+math.cos(math.pi*p)))
def push():
    try:
        tokn=os.environ['GH_TOKEN']; repo=os.environ['REPO']; path=f'progress/{RUN}.json'
        url=f'https://api.github.com/repos/{repo}/contents/{path}'; H={'Authorization':'Bearer '+tokn,'Accept':'application/vnd.github+json'}
        sha=None
        try: sha=json.load(urllib.request.urlopen(urllib.request.Request(url+'?ref=progress',headers=H)))['sha']
        except Exception: pass
        body={'message':f'{RUN} step {st["opt_step"]}','branch':'progress','content':base64.b64encode(json.dumps({**st,'elapsed_s':round(time.time()-T0),'N':N},indent=0).encode()).decode()}
        if sha: body['sha']=sha
        urllib.request.urlopen(urllib.request.Request(url,data=json.dumps(body).encode(),headers=H,method='PUT'))
    except Exception as e: print('push fail',e,flush=True)
def seq(prompt,ans):
    f=ids_of(prompt+ans); p=ids_of(prompt,add_generation_prompt=True)
    cp=0
    while cp<min(len(f),len(p)) and f[cp]==p[cp]: cp+=1
    return f,len(f)-cp
def logp(x,a):
    try: lg=model(input_ids=x,logits_to_keep=a+1).logits[0,:-1]
    except TypeError: lg=model(input_ids=x).logits[0,-a-1:-1]
    return -F.cross_entropy(lg.float(),x[0,-a:],reduction='sum')
print('loaded',round(time.time()-T0),'s, N',N,'resume at',st['next'],flush=True)
model.train(); i=st['next']; acc=0; L=[]; A=[]; MG=[]; ts=time.time()
def step():
    global acc,L,A,MG,ts
    for g in opt.param_groups: g['lr']=lr_at(st['opt_step'])
    gn=torch.nn.utils.clip_grad_norm_(params,1.0).item(); opt.step(); opt.zero_grad(); st['opt_step']+=1
    e={'step':st['opt_step'],'i':i,'loss':round(sum(L)/len(L),4),'reward_acc':round(sum(A)/len(A),3),'margin':round(sum(MG)/len(MG),3),'gn':round(gn,3),'lr':round(lr_at(st['opt_step']-1),8),'s':round(time.time()-ts,1)}
    st['log'].append(e); print(e,flush=True); acc=0; L=[]; A=[]; MG=[]; ts=time.time()
    if st['opt_step']%3==0: push()
while i<N and time.time()-T0<BUDGET:
    r=rows[i]; i+=1
    fc,ac=seq(r['prompt'],r['chosen']); fr,ar=seq(r['prompt'],r['rejected'])
    if min(ac,ar)<=0 or max(ac,ar)>=400 or max(len(fc),len(fr))>MAXLEN: st['skipped']+=1; continue
    xc=torch.tensor([fc]); xr=torch.tensor([fr])
    with torch.no_grad(), model.disable_adapter(): rc=logp(xc,ac); rr=logp(xr,ar)
    pc=logp(xc,ac); pr=logp(xr,ar)
    z=(pc-rc)-(pr-rr); loss=-F.logsigmoid(BETA*z)
    (loss/ACC).backward(); L.append(loss.item()); A.append(float(z.item()>0)); MG.append(z.item()); acc+=1; st['seen']+=1
    if acc==ACC: step()
if acc>0: step()
st['next']=i
model.save_pretrained(OUT); torch.save(opt.state_dict(),f'{OUT}/opt.pt'); json.dump(st,open(f'{OUT}/state.json','w'),indent=0)
if i<N: open(f'{OUT}/CONTINUE','w').write(str(i))
push(); print('saved next',i,'of',N,'elapsed',round(time.time()-T0),flush=True)
