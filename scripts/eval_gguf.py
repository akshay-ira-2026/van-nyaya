import os,json,time,urllib.request
S=int(os.environ['SHARDS']); k=int(os.environ['SHARD']); TAG=os.environ['TAG']
rows=[json.loads(l) for l in open('data/eval_all.jsonl')]
out=open(f'gen_{TAG}_{k}.jsonl','w')
for i,r in enumerate(rows):
    if i%S!=k: continue
    t=time.time()
    try:
        req=urllib.request.Request('http://127.0.0.1:8080/v1/chat/completions',data=json.dumps({'messages':r['messages'],'temperature':0,'max_tokens':300}).encode(),headers={'Content-Type':'application/json'})
        d=json.load(urllib.request.urlopen(req,timeout=900)); txt=d['choices'][0]['message']['content']; u=d.get('usage',{})
    except Exception as e: txt=None; u={'err':str(e)}
    rec={'idx':i,'lang':r.get('lang'),'type':r.get('type'),'ref':r.get('ref'),'pred':txt,'s':round(time.time()-t,1),'usage':u}
    out.write(json.dumps(rec,ensure_ascii=False)+'\n'); out.flush(); print(json.dumps(rec,ensure_ascii=False)[:400],flush=True)
