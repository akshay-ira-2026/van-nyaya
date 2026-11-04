"""Offline AuditKIT cross-check. No external model or judge calls.
Usage: python audit_offline.py base.jsonl sft.jsonl
"""
import sys,json,collections,importlib.metadata
from auditkit import Sample,F1Score,Abstention
import runpy
from pathlib import Path
MARKERS=runpy.run_path(str(Path(__file__).with_name("score.py")))["REFUSAL_PHRASES"]
report={'auditkit_version':importlib.metadata.version('auditkit'),'method':'AuditKIT offline F1Score and Abstention on stored predictions. Not a legal-expert review.','models':{}}
for path in sys.argv[1:]:
 rows=[json.loads(l) for l in open(path) if l.strip()];groups=collections.defaultdict(list);raw=[]
 for r in rows:
  sample=Sample(input='',target=r['ref'],id=str(r['idx']),metadata={'rag_case':{'case_id':str(r['idx']),'question':'','answerability':'unanswerable' if r['type']=='unanswerable' else 'answerable','gold_answer':r['ref']}})
  f=F1Score().score(sample,r['pred']);a=Abstention(refusal_markers=MARKERS).score(sample,r['pred'])[0]
  scores={'f1':f.value,'expected_action_accuracy':a.value}; groups[r['lang']+'_'+r['type']].append(scores);raw.append({'idx':r['idx'],**scores})
 summary={k:{'n':len(v),'f1':sum(x['f1'] for x in v)/len(v),'expected_action_accuracy':sum(x['expected_action_accuracy'] for x in v)/len(v)} for k,v in groups.items()}
 report['models'][path]={'summary':summary,'per_example':raw};print(path,json.dumps(summary,ensure_ascii=False))
json.dump(report,open('auditkit_report.json','w'),ensure_ascii=False,indent=2)
