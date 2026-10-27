"""Persistence, invalidation, concurrency and idle lifecycle of the real RAG index."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pytest

import rag_agent as rag


@pytest.fixture
def cache(tmp_path, monkeypatch):
    corpus = tmp_path / 'corpus'
    (corpus / 'sections').mkdir(parents=True)
    (corpus / 'sections/method.tex').write_text('Original thesis methodology.')
    (corpus / 'my_bib.bib').write_text('Original bibliography.')
    monkeypatch.setattr(rag, '_CORPUS_DIR', str(corpus))
    monkeypatch.setattr(rag, '_SECTION_FILES', ['sections/method.tex'])
    monkeypatch.setenv('RAG_CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setenv('RAG_INDEX_IDLE_SECONDS', '10')
    monkeypatch.setattr(rag, '_RAG_INDEX', None)
    monkeypatch.setattr(rag, '_RAG_INDEX_KEY', None, raising=False)
    monkeypatch.setattr(rag, '_RAG_LAST_USED', 0.0, raising=False)
    clock = [100.0]
    monkeypatch.setattr(rag, 'monotonic', lambda: clock[0], raising=False)
    chunks = [{'text': 'methodology passage', 'source': 'method.tex', 'kind': 'prose'},
              {'text': 'results table', 'source': 'result.tex', 'kind': 'table'}]
    monkeypatch.setattr(rag, '_load_corpus_chunks', lambda: [dict(c) for c in chunks])
    calls = []

    def embed(text):
        calls.append(text)
        return np.array([len(text), 1.0, 2.0], dtype=np.float32)

    monkeypatch.setattr(rag, '_embed', embed)
    return tmp_path / 'cache', corpus, clock, calls


def test_disk_cache_survives_process_memory_reset_without_reembedding(cache, monkeypatch):
    directory, _, _, calls = cache
    first = rag._get_index()
    assert len(calls) == 2
    assert (directory / 'thesis-index.npz').is_file()
    monkeypatch.setattr(rag, '_RAG_INDEX', None)
    second = rag._get_index()
    assert len(calls) == 2, 'Disk reload must not re-embed the thesis'
    np.testing.assert_array_equal(first[0]['vector'], second[0]['vector'])
    assert second[1]['source'] == 'result.tex'


@pytest.mark.parametrize('change', ['thesis', 'bibliography', 'model', 'chunker'])
def test_changed_embedding_inputs_rebuild_the_cache(cache, monkeypatch, change):
    _, corpus, _, calls = cache
    rag._get_index()
    if change == 'thesis':
        (corpus / 'sections/method.tex').write_text('Updated thesis methodology.')
    elif change == 'bibliography':
        (corpus / 'my_bib.bib').write_text('Updated bibliography.')
    elif change == 'model':
        monkeypatch.setenv('OLLAMA_EMBED_MODEL', 'different-model')
    else:
        monkeypatch.setattr(rag, '_MAX_CHUNK_CHARS', 450)
    rag._get_index()
    assert len(calls) == 4, 'Stale vectors must not survive changed embedding inputs'


def test_corrupted_disk_cache_is_rebuilt(cache, monkeypatch):
    directory, _, _, calls = cache
    rag._get_index()
    (directory / 'thesis-index.npz').write_bytes(b'partial or corrupt write')
    monkeypatch.setattr(rag, '_RAG_INDEX', None)
    assert len(rag._get_index()) == 2
    assert len(calls) == 4


def test_idle_eviction_keeps_disk_and_next_search_only_embeds_its_query(cache):
    directory, _, clock, calls = cache
    rag._search('first question')
    assert len(calls) == 3
    clock[0] += 9
    assert rag.evict_idle_index() is False
    rag._search('second question')
    clock[0] += 9
    assert rag.evict_idle_index() is False, 'Active use extends the idle deadline'
    clock[0] += 2
    assert rag.evict_idle_index() is True
    assert rag._RAG_INDEX is None
    assert (directory / 'thesis-index.npz').is_file()
    before = len(calls)
    assert rag._search('third question')
    assert len(calls) == before + 1, 'Only the new question should be embedded'


def test_simultaneous_first_requests_build_only_once(cache, monkeypatch):
    _, _, _, calls = cache
    gate = threading.Barrier(4)
    embed = rag._embed

    def delayed_embed(text):
        # Embedding HTTP calls yield the GIL; reproduce that interleaving.
        threading.Event().wait(0.01)
        return embed(text)

    monkeypatch.setattr(rag, '_embed', delayed_embed)

    def get():
        gate.wait(timeout=2)
        return rag._get_index()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: get(), range(4)))
    assert len(calls) == 2
    assert all(index is results[0] for index in results)


def test_application_startup_is_lazy_and_runs_idle_cleanup(monkeypatch):
    import main

    async def run():
        cleaned = asyncio.Event()
        warmed = []
        monkeypatch.setattr(rag, 'warm_index', lambda: warmed.append(True))
        monkeypatch.setenv('RAG_INDEX_SWEEP_SECONDS', '0.01')

        def evict():
            cleaned.set()

        monkeypatch.setattr(rag, 'evict_idle_index', evict, raising=False)
        async with main._lifespan(main.app):
            await asyncio.wait_for(cleaned.wait(), timeout=1)
            assert warmed == [], 'Startup must not load vectors into RAM'

    asyncio.run(run())
