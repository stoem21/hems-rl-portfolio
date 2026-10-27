#!/usr/bin/env python3
"""Report local artifact readiness without reading thesis, dataset or checkpoint content."""
from pathlib import Path
import os
import sys

backend = Path(__file__).resolve().parents[1] / 'hems-backend'
targets = {
    'Dataset': Path(os.environ.get('HEMS_DATA_PATH', str(backend / 'data/full_dataset.csv'))),
    'Checkpoint directory': Path(os.environ.get('HEMS_AGENTS_DIR', str(backend / 'agents'))),
    'Thesis corpus directory': Path(os.environ.get('HEMS_CORPUS_DIR', str(backend / 'thesis_corpus'))),
}
ready = True
for name, target in targets.items():
    if name == 'Checkpoint directory':
        ok = target.is_dir() and any(target.glob('*.pt'))
    elif name == 'Thesis corpus directory':
        ok = (target / 'my_bib.bib').is_file() and (target / 'sections/method.tex').is_file()
    else:
        ok = target.is_file()
    ready = ready and ok
    print(f'{name}: {"available locally" if ok else "not supplied (expected in a source-only checkout)"}')
sys.exit(0 if ready else 1)
