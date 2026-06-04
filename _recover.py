# -*- coding: utf-8 -*-
import csv, os, re, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
OUT='papers'
rows = list(csv.DictReader(open(os.path.join(OUT,'_manifest_descargas.csv'),encoding='utf-8')))

def alt_urls(url):
    cands=[]
    m = re.search(r'PMC(\d+)', url)
    if 'pmc.ncbi.nlm.nih.gov' in url and m:
        pid='PMC'+m.group(1)
        cands.append(f'https://europepmc.org/articles/{pid}?pdf=render')
        cands.append(f'https://www.ncbi.nlm.nih.gov/pmc/articles/{pid}/pdf/')
    elif 'mdpi.com' in url:
        base = url.split('/pdf')[0]
        cands.append(base + '/pdf?version=1')  # often fails (cloudflare)
        cands.append(url)
    elif 'aclanthology.org' in url:
        cands.append(url)
    return cands

def fetch(row):
    fn=row['filename']; url=row['url']
    path=os.path.join(OUT,fn)
    if os.path.exists(path) and os.path.getsize(path)>20000:
        return ('skip',fn,url)
    for cand in alt_urls(url):
        try:
            r=requests.get(cand,headers={'User-Agent':UA,'Accept':'application/pdf,*/*','Referer':url},timeout=45,allow_redirects=True)
            if r.content[:4]==b'%PDF':
                open(path,'wb').write(r.content)
                return ('ok',fn,cand)
        except Exception as e:
            pass
    return ('fail',fn,url)

todo=[r for r in rows if not (os.path.exists(os.path.join(OUT,r['filename'])) and os.path.getsize(os.path.join(OUT,r['filename']))>20000)]
print(f'Reintentando {len(todo)} faltantes...')
res=[]
with ThreadPoolExecutor(max_workers=6) as ex:
    futs={ex.submit(fetch,r):r for r in todo}
    for fu in as_completed(futs):
        res.append(fu.result())
ok=[r for r in res if r[0]=='ok']
fail=[r for r in res if r[0]=='fail']
print(f'Recuperados: {len(ok)}   Aun faltan: {len(fail)}')
for s,fn,url in ok: print('  OK   ',fn)
# total now on disk
have=[r for r in rows if os.path.exists(os.path.join(OUT,r['filename'])) and os.path.getsize(os.path.join(OUT,r['filename']))>20000]
print(f'\nTOTAL PDFs en papers/: {len(have)} de {len(rows)} con OA link')
