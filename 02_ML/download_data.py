"""
Downloads the PhysioNet/CinC Challenge 2019 sepsis training data (open access, CC BY 4.0)
into ./data/training_setA and ./data/training_setB.  Works on Windows, macOS and Linux.

    python download_data.py            (run from the 02_ML folder)

~40,000 small files; 24 parallel connections that are re-used, so it usually takes 5-15 minutes.
Safe to stop and re-run: files already downloaded are skipped.
"""
import http.client, os, re, ssl, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen

HOST = 'physionet.org'                              # file list
BASE = '/files/challenge-2019/1.0.0/training'
FILE_HOST = 'physionet-open.s3.amazonaws.com'       # PhysioNet's official open-data mirror; physionet.org now 302-redirects every file here
FILE_BASE = '/challenge-2019/1.0.0/training'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
THREADS = 24
_local = threading.local()


def conn():
    c = getattr(_local, 'c', None)
    if c is None:
        c = http.client.HTTPSConnection(FILE_HOST, timeout=30, context=ssl.create_default_context())
        _local.c = c
    return c


def fetch(task):
    path, dest = task
    for attempt in range(4):
        try:
            c = conn()
            c.request('GET', path, headers={'Connection': 'keep-alive'})
            r = c.getresponse()
            body = r.read()
            if r.status != 200:
                raise IOError(f'HTTP {r.status}')
            tmp = dest + '.part'
            with open(tmp, 'wb') as f:
                f.write(body)
            os.replace(tmp, dest)
            return True
        except Exception:
            try:
                _local.c.close()
            except Exception:
                pass
            _local.c = None
            time.sleep(1 + attempt)
    return False


def main():
    total_failed = 0
    for s in ('A', 'B'):
        d = f'training_set{s}'
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
        print(f'== {d}: getting the file list ...', flush=True)
        html = urlopen(f'https://{HOST}{BASE}/{d}/', timeout=60).read().decode()
        names = sorted(set(re.findall(r'p\d+\.psv', html)))
        todo = [(f'{FILE_BASE}/{d}/{n}',os.path.join(OUT, d, n)) for n in names
                if not (os.path.exists(os.path.join(OUT, d, n)) and os.path.getsize(os.path.join(OUT, d, n)) > 0)]
        print(f'   {len(names)} files, {len(todo)} to download', flush=True)
        done = failed = 0
        t0 = time.time()
        with ThreadPoolExecutor(THREADS) as ex:
            for ok in ex.map(fetch, todo):
                done += 1
                failed += (not ok)
                if done % 500 == 0 or done == len(todo):
                    rate = done / max(1e-6, time.time() - t0)
                    left = (len(todo) - done) / max(rate, 1e-6)
                    print(f'   {done}/{len(todo)}  ({rate:.0f} files/s, ~{left/60:.1f} min left)', flush=True)
        total_failed += failed
    if total_failed:
        print(f'{total_failed} files failed - run the script again to fetch them.')
        sys.exit(1)
    print('DONE. Data is in', OUT)


if __name__ == '__main__':
    main()
