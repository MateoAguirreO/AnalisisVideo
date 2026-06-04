# -*- coding: utf-8 -*-
import csv, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import requests
    HAVE_REQ = True
except Exception:
    HAVE_REQ = False
    import urllib.request

UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'
OUT = 'papers'
rows = list(csv.DictReader(open(os.path.join(OUT,'_manifest_descargas.csv'),encoding='utf-8')))

def fetch(row):
    fn = row['filename']; url = row['url']; title = row['title']
    path = os.path.join(OUT, fn)
    if os.path.exists(path) and os.path.getsize(path) > 20000:
        return ('skip', fn, url, 'ya existe')
    try:
        if HAVE_REQ:
            r = requests.get(url, headers={'User-Agent':UA,'Accept':'application/pdf,*/*'}, timeout=45, allow_redirects=True)
            data = r.content
        else:
            req = urllib.request.Request(url, headers={'User-Agent':UA})
            data = urllib.request.urlopen(req, timeout=45).read()
        if data[:4] == b'%PDF':
            open(path,'wb').write(data)
            return ('ok', fn, url, f'{len(data)//1024} KB')
        else:
            head = data[:120].decode('latin-1','ignore').replace('\n',' ')
            return ('notpdf', fn, url, f'no-PDF ({len(data)} bytes)')
    except Exception as e:
        return ('err', fn, url, str(e)[:90])

results = []
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(fetch,r): r for r in rows}
    for fu in as_completed(futs):
        results.append(fu.result())

ok      = [r for r in results if r[0] in ('ok','skip')]
notpdf  = [r for r in results if r[0]=='notpdf']
err     = [r for r in results if r[0]=='err']

print(f'\n=== DESCARGA: {len(ok)} OK / {len(notpdf)} no-PDF / {len(err)} error  (de {len(rows)}) ===\n')
print('--- NO PDF (pagina HTML / bloqueado) ---')
for s,fn,url,info in notpdf:
    print(f'  {url}')
print('\n--- ERROR ---')
for s,fn,url,info in err:
    print(f'  {url}  :: {info}')

# write a fallback list (links que no se pudieron bajar) for the dossier
with open(os.path.join(OUT,'_no_descargados.txt'),'w',encoding='utf-8') as f:
    for s,fn,url,info in notpdf+err:
        f.write(f'{url}\t{info}\n')
print('\nGuardado papers/_no_descargados.txt con los links no bajados.')
