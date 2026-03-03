import asyncio
import json
import os
from typing import List

import redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from ..memory import FractalMemory

app = FastAPI()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
# We use a raw client for Pub/Sub (listening), but Memory for fetching
r_sub = redis.Redis.from_url(REDIS_URL, decode_responses=True)
memory = FractalMemory(redis_url=REDIS_URL)


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


@app.get("/")
async def get():
    with open("src/fractal_agents/visualizer/templates/index.html", "r") as f:
        return HTMLResponse(f.read())


async def get_all_nodes():
    """Helper to fetch all current nodes using FractalMemory."""
    # We need to access the underlying client to scan keys,
    # but use memory.get_node_state to deserialize/decompress.
    keys = memory.client.keys("fractal:node:*")
    nodes = []
    for key in keys:
        try:
            # key is bytes from memory.client (decode_responses=False)
            node_id = key.decode("utf-8").split(":")[-1]
            node = memory.get_node_state(node_id)
            if node:
                nodes.append(node)
        except Exception:
            pass
    return nodes


async def redis_listener():
    """Listens to Redis Pub/Sub for node updates."""
    pubsub = r_sub.pubsub()
    pubsub.subscribe("fractal:updates")

    async for message in pubsub.listen():
        if message["type"] == "message":
            try:
                # Broadcast the single node update directly
                await manager.broadcast(message["data"])
            except Exception as e:
                print(f"Broadcast error: {e}")


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(redis_listener())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    # Send initial state
    try:
        nodes = await get_all_nodes()
        await websocket.send_text(json.dumps(nodes))

        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WS Error: {e}")
        manager.disconnect(websocket)
