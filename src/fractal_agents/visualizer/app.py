import os
import json
import redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any
import asyncio

app = FastAPI()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
r = redis.Redis.from_url(REDIS_URL, decode_responses=True)

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

async def redis_poller():
    """Polls Redis for node updates and broadcasts to clients."""
    last_nodes = {}
    while True:
        try:
            # Get all node keys
            keys = r.keys("fractal:node:*")
            nodes = []
            for key in keys:
                data = r.get(key)
                if data:
                    node = json.loads(data)
                    nodes.append(node)
            
            # Simple state tracking to only broadcast on change
            current_state = {n['id']: n['status'] for n in nodes}
            if current_state != last_nodes:
                await manager.broadcast(json.dumps(nodes))
                last_nodes = current_state
                
        except Exception as e:
            print(f"Poller error: {e}")
        
        await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(redis_poller())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
