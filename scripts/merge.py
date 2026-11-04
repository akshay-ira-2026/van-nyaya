import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
M=os.environ.get('MODEL','1-800-LLMs/tiny-aya-global'); OUT=os.environ['OUT']
if not (M.startswith('1-800-LLMs/') or os.path.isdir(M)): raise SystemExit('MODEL not allowed: '+M)
tok=AutoTokenizer.from_pretrained(M)
m=AutoModelForCausalLM.from_pretrained(M,torch_dtype=torch.float32,low_cpu_mem_usage=True)
for d in ['sft_adapter','dpo_adapter']:
    if os.path.exists(f'{d}/adapter_config.json'):
        m=PeftModel.from_pretrained(m,d).merge_and_unload(); print('merged',d,flush=True)
m=m.to(torch.bfloat16); m.save_pretrained(OUT,safe_serialization=True); tok.save_pretrained(OUT)
from huggingface_hub import hf_hub_download
import shutil
for f in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja']:
    try: shutil.copy(hf_hub_download(M,f),os.path.join(OUT,f)); print('orig',f)
    except Exception as e: print('skip',f,type(e).__name__)
print('saved',OUT)
