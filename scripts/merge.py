import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
M=os.environ.get('MODEL','1-800-LLMs/tiny-aya-global'); OUT=os.environ['OUT']
tok=AutoTokenizer.from_pretrained(M)
m=AutoModelForCausalLM.from_pretrained(M,torch_dtype=torch.float32,low_cpu_mem_usage=True)
for d in ['sft_adapter','dpo_adapter']:
    if os.path.exists(f'{d}/adapter_config.json'):
        m=PeftModel.from_pretrained(m,d).merge_and_unload(); print('merged',d,flush=True)
m=m.to(torch.bfloat16); m.save_pretrained(OUT,safe_serialization=True); tok.save_pretrained(OUT); print('saved',OUT)
