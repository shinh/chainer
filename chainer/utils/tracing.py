import collections
import contextlib
import json
import threading
import time


TraceEvent = collections.namedtuple(
    'TraceEvent', ('tid', 'cat', 'name', 'start_time', 'end_time'))


_tracing_event_stack = []
_tracing_events = []


tracing_enabled = False


def enable_tracing():
    global tracing_enabled
    tracing_enabled = True


def disable_tracing():
    global tracing_enabled
    tracing_enabled = False


def start_trace(cat, name):
    if not tracing_enabled: return
    start_time = time.time()
    tid = threading.get_ident()
    event = (tid, cat, name, start_time)
    _tracing_event_stack.append(event)


def end_trace(cat, name):
    if not tracing_enabled: return
    end_time = time.time()
    event = _tracing_event_stack.pop()
    event = TraceEvent(event[0], event[1], event[2], event[3], end_time)
    assert event.name == name
    assert event.cat == cat
    _tracing_events.append(event)


@contextlib.contextmanager
def trace(cat, name):
    start_trace(cat, name)
    yield
    end_trace(cat, name)


def trace_to_json():
    events = []
    for event in _tracing_events:
        ts = int(event.start_time * 1000000)
        tts = int((event.end_time - event.start_time) * 1000000)
        events.append({
            'tid': event.tid,
            'name': event.name,
            'ts': ts,
            'dur': tts,
            'pid': 1,
            'cat': event.cat,
            'ph': 'X',
        })

    return events


def dump_trace_to_file(filename):
    obj = trace_to_json()
    with open(filename, 'w') as f:
        json.dump(obj, f)

