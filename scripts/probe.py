import os, time, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
torch.set_num_threads(os.cpu_count())
DT={'float32':torch.float32,'bfloat16':torch.bfloat16}[os.environ.get('DTYPE','float32')]
M=os.environ.get('MODEL','1-800-LLMs/tiny-aya-global')
R={'cpus':os.cpu_count(),'dtype':str(DT)}
def save(): json.dump(R,open('probe.json','w'),indent=1); print('R',json.dumps(R),flush=True)
def T(tok,msgs,**k):
    o=tok.apply_chat_template(msgs,return_tensors='pt',**k)
    return o if torch.is_tensor(o) else o['input_ids']
t=time.time(); tok=AutoTokenizer.from_pretrained(M)
model=AutoModelForCausalLM.from_pretrained(M, torch_dtype=DT, low_cpu_mem_usage=True)
R['load_s']=round(time.time()-t,1); R['vocab']=model.config.vocab_size; R['layers']=model.config.num_hidden_layers; save()
rows=[json.loads(l) for l in open('data/sft_mix.jsonl')][:6]
model.eval()
msgs=[m for m in rows[0]['messages'] if m['role']!='assistant']
ids=T(tok,msgs,add_generation_prompt=True)
t=time.time()
with torch.no_grad(): out=model(input_ids=ids)
R['prefill']={'tok':ids.shape[1],'s':round(time.time()-t,1)}; save()
t=time.time()
with torch.no_grad(): out=model.generate(ids,max_new_tokens=16,do_sample=False)
R['gen16_s']=round(time.time()-t,1); R['gen_text']=tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True); save()
model.train(); model.gradient_checkpointing_enable(); model.enable_input_require_grads()
model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.0,target_modules=['q_proj','k_proj','v_proj','o_proj'],task_type='CAUSAL_LM'))
opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
L=int(os.environ.get('MAXLEN','256'))
for r in rows[:4]:
    ids=T(tok,r['messages'])[:,:L]
    t=time.time(); out=model(input_ids=ids,labels=ids); out.loss.backward(); opt.step(); opt.zero_grad()
    R.setdefault('train',[]).append({'len':ids.shape[1],'loss':round(out.loss.item(),3),'s':round(time.time()-t,1)}); save()
R['done']=True; save()
