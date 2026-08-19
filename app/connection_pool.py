from typing import Dict
from fastapi import WebSocket

class ConnectionPoolManager:
    """Manages active full-duplex WebSocket connections."""
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        print(f"[POOL] Client {client_id} connected. Active sessions: {len(self.active_connections)}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            print(f"[POOL] Client {client_id} disconnected. Active sessions: {len(self.active_connections)}")

    async def broadcast_text(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

pool_manager = ConnectionPoolManager()