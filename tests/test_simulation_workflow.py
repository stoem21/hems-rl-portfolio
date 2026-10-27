import asyncio
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main
import rag_agent as rag

client = TestClient(main.app)
REQUEST = dict(household_id=1, start_date='2012-08-01', n_days=1, agent='I-TD3', seed=1)


@pytest.fixture
def data(monkeypatch):
    frame = pd.DataFrame({'customer_id': 1, 'dt': pd.date_range('2012-08-01', periods=96, freq='30min'),
                          'load': 2.0, 'pv': 1.0, 'price': .3})
    monkeypatch.setattr(main, 'get_df', lambda: frame)
    return frame


@pytest.mark.parametrize('change', [dict(household_id=0), dict(household_id=301), dict(n_days=0), dict(n_days=31),
                                  dict(seed=6), dict(start_date='2012-02-30'), dict(agent='made-up'),
                                  dict(error_type='unknown'), dict(error_pct=21)])
def test_invalid_parameters_rejected_before_data_or_model_loading(monkeypatch, change):
    monkeypatch.setattr(main, 'get_df', lambda: pytest.fail('Invalid inputs must not load data'))
    assert client.post('/simulate', json=REQUEST | change).status_code == 422


def test_preflight_checks_the_full_window_and_forecast_lookahead(data):
    response = client.post('/simulate/validate', json=REQUEST)
    assert response.status_code == 200
    assert response.json()['n_steps'] == 48
    assert response.json()['required_end'] == '2012-08-02T16:00:00'


@pytest.mark.parametrize('change', [dict(start_date='2012-07-31'), dict(n_days=2)])
def test_no_shift_to_next_date_or_silent_shortening(data, change):
    response = client.post('/simulate', json=REQUEST | change)
    assert response.status_code == 422
    assert response.json()['detail']['code'] == 'outside_data_range'


def test_gaps_in_lookahead_are_rejected_in_form_and_agent_path(data, monkeypatch):
    monkeypatch.setattr(main, 'get_df', lambda: data.drop(index=50))
    for route in ['/simulate', '/simulate/validate']:
        response = client.post(route, json=REQUEST)
        assert response.status_code == 422
        assert response.json()['detail']['code'] == 'data_gap'
        assert response.json()['detail']['missing_intervals'] == 1


def test_duplicate_measurements_are_not_simulated(data, monkeypatch):
    monkeypatch.setattr(main, 'get_df', lambda: pd.concat([data, data.iloc[[0]]]))
    assert client.post('/simulate/validate', json=REQUEST).json()['detail']['code'] == 'irregular_data'


def test_non_finite_measurements_are_rejected(data):
    data.loc[3, 'load'] = np.inf
    assert client.post('/simulate/validate', json=REQUEST).json()['detail']['code'] == 'invalid_measurements'


@pytest.fixture
def model(monkeypatch):
    calls = []

    class Env:
        def reset(self):
            self.i = 0
            return [0], {}

        def step(self, action):
            self.i += 1
            return [0], 0, self.i == 48, False, dict(soc=.5, action_mapped_kw=0, price=.3, load_kw=2, pv_kw=1, cost=.1)

    def policy(agent, seed):
        calls.append((agent, seed))
        return None

    monkeypatch.setitem(sys.modules, 'agent_loader', SimpleNamespace(load_policy=policy, get_action=lambda p, o: 0, build_sim_env=lambda **kwargs: Env()))
    return calls


def test_compare_uses_same_runner_and_returns_verified_chart_timelines(data, model):
    body = {k: v for k, v in REQUEST.items() if k != 'agent'} | {'agents': ['I-TD3', 'R-TD3']}
    output = client.post('/simulate/compare', json=body)
    assert output.status_code == 200
    results = output.json()['results']
    assert model == [('I-TD3', 1), ('R-TD3', 1)]
    assert all(r['n_steps'] == 48 and len(r['timestamps']) == 48 for r in results)
    assert all(r['cost_timeline'][-1] == pytest.approx(r['period_cost']) for r in results)
    assert results[0]['baseline_cost_timeline'] == results[1]['baseline_cost_timeline']
    assert results[0]['period_cost'] == 4.8


def test_reactive_agent_reports_effective_error_setting(data, model):
    result = client.post('/simulate', json=REQUEST | dict(agent='R-TD3', error_type='noise', error_pct=20)).json()
    assert result['error_type'] == 'none' and result['error_pct'] == 0
    assert result['warnings']


def test_tool_summary_includes_authoritative_data_split_without_chart_arrays(data, model):
    result = client.post('/simulate', json=REQUEST).json()
    summary = rag._simulation_summary(result)
    assert summary['household_role'] == 'test'
    assert summary['period'] == 'test'
    assert summary['start_date'] == '2012-08-01'
    assert 'soc_timeline' not in summary and 'cost_timeline' not in summary


def test_duplicate_comparison_agents_are_rejected_before_loading(monkeypatch):
    monkeypatch.setattr(main, 'get_df', lambda: pytest.fail('Invalid comparison must not load data'))
    body = {k: v for k, v in REQUEST.items() if k != 'agent'} | {'agents': ['I-TD3', 'I-TD3']}
    assert client.post('/simulate/compare', json=body).status_code == 422


def collect(generator):
    async def run():
        return [event async for event in generator]
    return asyncio.run(run())


def test_agent_stream_emits_progress_artifacts_and_text_without_leaking_raw_tool_messages(monkeypatch):
    sample = {'agent': 'I-TD3', 'soc_timeline': [.1, .2]}

    class Graph:
        async def astream_events(self, *args, **kwargs):
            yield dict(event='on_tool_start', name='run_simulation', run_id='sim-1', data={})
            yield dict(event='on_tool_end', name='run_simulation', run_id='sim-1', data={'output': SimpleNamespace(content='LLM summary', artifact={'results': [sample]})})
            yield dict(event='on_chat_model_stream', data={'chunk': SimpleNamespace(content='The result is ready.')})

    monkeypatch.setattr(rag, '_get_graph', lambda *args: Graph())
    events = collect(rag.stream_agent_events('run', 'context', 'ollama', 'model'))
    assert events[0]['state'] == 'running'
    assert events[1]['type'] == 'simulation' and events[1]['results'] == [sample]
    assert events[2]['state'] == 'complete'
    assert events[3]['stage'] == 'summarizing'
    assert ''.join(collect(rag.stream_agent('run', 'context', 'ollama', 'model'))) == 'The result is ready.'


def test_failed_tool_never_emits_fabricated_chart_data(monkeypatch):
    class Graph:
        async def astream_events(self, *args, **kwargs):
            yield dict(event='on_tool_end', name='run_simulation', run_id='bad', data={'output': SimpleNamespace(artifact={'error': 'Missing data', 'results': []})})

    monkeypatch.setattr(rag, '_get_graph', lambda *args: Graph())
    events = collect(rag.stream_agent_events('run', 'context', 'ollama', 'model'))
    assert [e['type'] for e in events] == ['error', 'status']
    assert events[-1]['state'] == 'failed'


def test_tool_schema_error_is_shown_as_failed_progress(monkeypatch):
    class Graph:
        async def astream_events(self, *args, **kwargs):
            yield dict(event='on_tool_end', name='run_simulation', run_id='bad', data={'output': SimpleNamespace(artifact=None, status='error', content='Invalid start date')})
    monkeypatch.setattr(rag, '_get_graph', lambda *args: Graph())
    events = collect(rag.stream_agent_events('run', 'context', 'ollama', 'model'))
    assert events[0]['message'] == 'Invalid start date'
    assert events[-1]['state'] == 'failed'


def test_http_event_stream_preserves_completed_results_when_answer_fails(monkeypatch):
    monkeypatch.setenv('OLLAMA_URL', 'http://fake')

    async def stream(*args):
        yield {'type': 'simulation', 'results': [{'agent': 'I-TD3'}]}
        raise RuntimeError('Model stopped')

    monkeypatch.setattr(rag, 'stream_agent_events', stream)
    response = client.post('/ask', json={'question': 'run', 'stream_events': True})
    assert 'text/event-stream' in response.headers['content-type']
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
    assert events[0]['stage'] == 'thinking'
    assert events[1]['results'][0]['agent'] == 'I-TD3'
    assert events[-2] == {'type': 'error', 'message': 'Model stopped'}
    assert events[-1]['type'] == 'done'
