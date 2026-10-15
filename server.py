"""MuJoCo scene server: build a world from a YAML file, step it, serve it.

    python server.py config/ur10e_robotiq.yaml              # headless
    mjpython server.py config/ur10e_robotiq.yaml --gui      # viewer (macOS needs mjpython)

Only the physics loop (main thread) touches MuJoCo state. The WebSocket thread
only parses JSON and queues requests; primitives.py executes them between steps.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import queue
import threading
import time

import mujoco
import mujoco.viewer
import yaml

from utils.camera_utils import RGB_ONLY_GROUP
from utils.engine_utils import apply_home, build_model, camera_index, robot_index
from primitives import Context, Task, execute


def serve_ws(commands: queue.Queue, host: str, port: int) -> None:
    """Network thread: parse JSON, hand the request to the physics loop, send its reply back."""
    from websockets.asyncio.server import serve

    async def handler(ws):
        async for raw in ws:
            try:
                req = json.loads(raw)
                if not isinstance(req, dict):
                    raise ValueError("request must be a JSON object")
            except ValueError as exc:
                reply = {"request_id": None, "status": "rejected", "result": None, "error": f"bad request: {exc}"}
            else:
                fut = asyncio.get_running_loop().create_future()
                commands.put((req, fut))
                reply = await fut
            await ws.send(json.dumps(reply))

    async def main():
        async with serve(handler, host, port):
            await asyncio.Future()

    threading.Thread(target=lambda: asyncio.run(main()), daemon=True).start()


def run_loop(ctx: Context, commands: queue.Queue, gui: bool) -> None:
    model, data = ctx.model, ctx.data

    running = []   # (Task, future) for blocking verbs still in flight

    def resolve(fut, reply):
        fut.get_loop().call_soon_threadsafe(fut.set_result, reply)

    def serve_commands():  # between two steps, so verbs never race the physics
        while not commands.empty():
            req, fut = commands.get()
            out = execute(ctx, req)
            if isinstance(out, Task):
                running.append((out, fut))
            else:
                resolve(fut, out)

    def advance_tasks():
        for item in running[:]:
            reply = item[0].step()
            if reply is not None:
                running.remove(item)
                resolve(item[1], reply)

    def step_realtime(sync=None):
        while sync is None or sync.is_running():
            serve_commands()
            t0 = time.perf_counter()
            mujoco.mj_step(model, data)
            advance_tasks()
            if sync:
                sync.sync()
            dt = model.opt.timestep - (time.perf_counter() - t0)
            if dt > 0:
                time.sleep(dt)

    if not gui:
        step_realtime()
        return
    try:
        with mujoco.viewer.launch_passive(model, data) as v:
            v.opt.geomgroup[RGB_ONLY_GROUP] = 1   # show the ArUco stickers (group 4) in the window, like the RGB camera does
            step_realtime(v)
    except RuntimeError as exc:
        raise SystemExit(f"{exc}\n(on macOS run the --gui mode with `mjpython server.py ...`)")


def main() -> None:
    p = argparse.ArgumentParser(description="MuJoCo scene server")
    p.add_argument("config", help="scene YAML, e.g. config/ur10e_robotiq.yaml")
    p.add_argument("--gui", action="store_true", help="open the MuJoCo viewer")
    p.add_argument("--host", help="override server.host")
    p.add_argument("--port", type=int, help="override server.port")
    args = p.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    model = build_model(cfg)
    data = mujoco.MjData(model)
    apply_home(model, data, cfg)

    srv = cfg.get("server", {})
    host, port = args.host or srv.get("host", "127.0.0.1"), args.port or srv.get("port", 8988)
    ctx = Context(model, data, cfg, robot_index(model, cfg), cameras=camera_index(model, cfg))
    commands = queue.Queue()
    serve_ws(commands, host, port)
    print(f"{args.config}: {model.nbody} bodies, {model.njnt} joints, {model.nu} actuators "
          f"| ws://{host}:{port}", flush=True)

    run_loop(ctx, commands, args.gui)


if __name__ == "__main__":
    main()
