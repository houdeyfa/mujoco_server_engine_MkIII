"""Client for the MuJoCo scene server. One file, copy it next to any script.

    with WorldController() as world:
        robots = world.act(Primitive('list_robots'))['robots']
        state = world.act(Primitive('get_state', 'ur'))          # robot_id defaults to 'world'

A Primitive is a verb, who it is addressed to, and its args - nothing else. The server
decides what a verb means, so new server verbs never require a client change.
act() blocks for the reply and returns its `result`; it raises ActionError if the
server answered `rejected` or `failed`.

Wire format (JSON, one request, one reply):
    request  {"request_id": str, "robot_id": "world" | "<robot>", "action": "<verb>", "value": {named args}}
    reply    {"request_id": str, "status": "done" | "failed" | "rejected", "result": {...} | null, "error": str | null}
Needs `pip install websockets`; images need numpy (+ pillow for png) and arrive already decoded as numpy RGB arrays.
This folder depends on nothing else in the project.
"""
import base64
import json
import uuid
from dataclasses import dataclass

from websockets.sync.client import connect

WORLD = 'world'
MAX_MESSAGE_BYTES = 64 * 1024 * 1024  # same ceiling as the server, or a big image arrives as a closed socket


class ActionError(RuntimeError):
    """Server rejected the request or the action failed."""

    def __init__(self, primitive, reply):
        self.reply = reply
        super().__init__(f"{primitive.robot_id}.{primitive.verb}: {reply.get('status')}: {reply.get('error')}")


@dataclass(init=False)
class Primitive:
    verb: str
    robot_id: str
    args: dict

    def __init__(self, verb, robot_id=WORLD, **args):
        self.verb = verb
        self.robot_id = robot_id
        self.args = args


class WorldController:
    def __init__(self, host='127.0.0.1', port=8988, timeout=60.0):
        self.timeout = timeout
        self._ws = connect(f'ws://{host}:{port}', max_size=MAX_MESSAGE_BYTES)

    def act(self, primitive: Primitive):
        """Send one primitive, block until its reply, return the result."""
        request_id = uuid.uuid4().hex
        self._ws.send(json.dumps({'request_id': request_id, 'robot_id': primitive.robot_id,
                                  'action': primitive.verb, 'value': primitive.args},
                                 default=lambda o: o.tolist()))   # numpy arrays go out as lists
        reply = json.loads(self._ws.recv(timeout=self.timeout))
        if reply.get('request_id') not in (request_id, None):
            raise RuntimeError(f"reply for {reply.get('request_id')}, expected {request_id}")
        if reply['status'] != 'done':
            raise ActionError(primitive, reply)
        return _decode_images(reply['result'])

    def close(self):
        self._ws.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _decode_images(value):
    """Any {"encoding", "shape", "dtype", "data"} payload in a result becomes a numpy array (RGB)."""
    if isinstance(value, list):
        return [_decode_images(v) for v in value]
    if not isinstance(value, dict):
        return value
    if 'encoding' in value and 'data' in value:
        import numpy as np
        data = base64.b64decode(value['data'])
        if value['encoding'] == 'raw':
            return np.frombuffer(data, dtype=np.dtype(value['dtype'])).reshape(value['shape'])
        import io
        from PIL import Image
        return np.array(Image.open(io.BytesIO(data)))
    return {k: _decode_images(v) for k, v in value.items()}
