import os, time, json, torch, platform
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
torch.set_num_threads(os.cpu_count())
M=os.environ.get('MODEL','1-800-LLMs/tiny-aya-global')
R={'cpus':os.cpu_count()}
t=time.time(); tok=AutoTokenizer.from_pretrained(M)
model=AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
R['load_s']=round(time.time()-t,1); print(R,flush=True)
rows=[json.loads(l) for l in open('data/sft_mix.jsonl')][:8]
ev=[json.loads(l) for l in open('data/eval_all.jsonl')][:2]
print('eval keys',list(ev[0].keys()),flush=True)
# generation probe
model.eval()
for r in rows[:2]:
    msgs=[m for m in r['messages'] if m['role']!='assistant']
    ids=tok.apply_chat_template(msgs,add_generation_prompt=True,return_tensors='pt')
    t=time.time()
    with torch.no_grad(): out=model.generate(ids,max_new_tokens=64,do_sample=False)
    n=out.shape[1]-ids.shape[1]; dt=time.time()-t
    R.setdefault('gen',[]).append({'prompt_tok':ids.shape[1],'new':n,'s':round(dt,1)})
    print('GEN',tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True)[:300],flush=True)
print(R,flush=True)
# train probe
model.train(); model.gradient_checkpointing_enable(); model.enable_input_require_grads()
names=sorted({n.split('.')[-1] for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)})
print('linear names',names,flush=True)
tm=[x for x in ['q_proj','k_proj','v_proj','o_proj'] if x in names]
model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.0,target_modules=tm,task_type='CAUSAL_LM'))
model.print_trainable_parameters()
opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
L=int(os.environ.get('MAXLEN','384'))
for i,r in enumerate(rows[:5]):
    ids=tok.apply_chat_template(r['messages'],return_tensors='pt')[:,:L]
    t=time.time(); out=model(input_ids=ids,labels=ids); out.loss.backward(); opt.step(); opt.zero_grad()
    R.setdefault('train',[]).append({'len':ids.shape[1],'loss':round(out.loss.item(),3),'s':round(time.time()-t,1)})
    print(R['train'][-1],flush=True)
json.dump(R,open('probe.json','w'),indent=1); print('FINAL',json.dumps(R),flush=True)
