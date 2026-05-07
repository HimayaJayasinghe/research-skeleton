"""
database/connection.py
Async MongoDB connection with a Local JSON fallback.
If Cloud MongoDB (Atlas) is unreachable, it automatically switches to a local file.
"""
import os
import json
import asyncio
import structlog
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from typing import Optional, Any, Dict, List
from datetime import datetime

log = structlog.get_logger()

# ═══════════════════════════════════════════════════════════════════════════════
#  LOCAL DATABASE MOCK (Async JSON Storage)
# ═══════════════════════════════════════════════════════════════════════════════

class LocalCursor:
    """Mock for Motor/MongoDB cursor."""
    def __init__(self, collection, query: Optional[Dict] = None):
        self._collection = collection
        self._query = query
        self._limit = None
        self._sort_key = None
        self._sort_dir = 1

    def sort(self, key: str, direction: int = 1):
        self._sort_key = key
        self._sort_dir = direction
        return self

    def limit(self, n: int):
        self._limit = n
        return self

    async def to_list(self, length: int = 100):
        data = await self._collection._get_data()
        
        # Simple filtering
        if self._query:
            filtered = []
            for doc in data:
                match = True
                for k,v in self._query.items():
                    if k.startswith("$"): continue
                    if doc.get(k) != v: match = False; break
                if match: filtered.append(doc)
            data = filtered
            
        # Sorting
        if self._sort_key:
            data.sort(key=lambda x: x.get(self._sort_key, 0), reverse=(self._sort_dir == -1))
            
        # Limit
        limit = min(length, self._limit) if self._limit else length
        return data[:limit]

class LocalCollection:
    """Mock for Motor/MongoDB collection using a shared JSON file."""
    def __init__(self, db, name: str):
        self._db = db
        self._name = name

    async def _get_data(self) -> List[Dict]:
        all_data = await self._db._read()
        return all_data.get(self._name, [])

    async def _save_data(self, data: List[Dict]):
        all_data = await self._db._read()
        all_data[self._name] = data
        await self._db._write(all_data)

    async def insert_one(self, doc: Dict):
        data = await self._get_data()
        # Convert datetime to string for JSON serialization
        doc_copy = doc.copy()
        for k, v in doc_copy.items():
            if isinstance(v, datetime): doc_copy[k] = v.isoformat()
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, datetime): v[k2] = v2.isoformat()
        
        data.append(doc_copy)
        await self._save_data(data)
        return type('Result', (), {'inserted_id': 'local_id'})()

    async def find_one(self, query: Dict, projection: Optional[Dict] = None):
        cursor = LocalCursor(self, query)
        results = await cursor.to_list(length=1)
        return results[0] if results else None

    def find(self, query: Dict = None, projection: Optional[Dict] = None):
        return LocalCursor(self, query)

    async def delete_many(self, query: Dict):
        """Mock delete_many."""
        if not query:
            await self._save_data([])
            return type('Result', (), {'deleted_count': 1})()
        
        data = await self._get_data()
        # Simple match-based deletion
        new_data = []
        deleted = 0
        for doc in data:
            match = True
            for k, v in query.items():
                if doc.get(k) != v: match = False; break
            if match: deleted += 1
            else: new_data.append(doc)
        
        await self._save_data(new_data)
        return type('Result', (), {'deleted_count': deleted})()

    async def delete_one(self, query: Dict):
        """Mock delete_one."""
        data = await self._get_data()
        new_data = []
        deleted = 0
        for doc in data:
            if deleted == 0:
                match = True
                for k, v in query.items():
                    if doc.get(k) != v: match = False; break
                if match: 
                    deleted = 1
                    continue
            new_data.append(doc)
        
        await self._save_data(new_data)
        return type('Result', (), {'deleted_count': deleted})()

    async def update_one(self, query: Dict, update: Dict):
        data = await self._get_data()
        for i, doc in enumerate(data):
            match = True
            for k, v in query.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                if "$set" in update:
                    for k, v in update["$set"].items():
                        if isinstance(v, datetime): v = v.isoformat()
                        # Handle nested updates (simple version)
                        if "." in k:
                            parts = k.split(".")
                            target = doc
                            for p in parts[:-1]:
                                if p not in target: target[p] = {}
                                target = target[p]
                            target[parts[-1]] = v
                        else:
                            doc[k] = v
                if "$inc" in update:
                    for k, v in update["$inc"].items():
                        doc[k] = doc.get(k, 0) + v
                data[i] = doc
                await self._save_data(data)
                return type('Result', (), {'modified_count': 1})()
        return type('Result', (), {'modified_count': 0})()

    async def update_many(self, query: Dict, update: Dict):
        # Simplified: just runs logic on all
        data = await self._get_data()
        modified = 0
        for i, doc in enumerate(data):
            match = True
            for k, v in query.items():
                if doc.get(k) != v: match = False; break
            if match:
                if "$set" in update:
                    for k, v in update["$set"].items(): doc[k] = v
                data[i] = doc
                modified += 1
        await self._save_data(data)



    async def count_documents(self, query: Dict):
        data = await self._get_data()
        return len(data)

    async def create_index(self, keys, unique=False):
        pass # Not needed for local file

    def aggregate(self, pipeline: List[Dict]):
        # Mock aggregate: returns recent data for stats
        return LocalCursor(self)

class LocalDatabase:
    """Mock for Motor Database using a JSON file."""
    def __init__(self, path: str):
        self._path = Path(path)
        self._lock = asyncio.Lock()
        if not self._path.parent.exists():
            self._path.parent.mkdir(parents=True)
        if not self._path.exists():
            with open(self._path, "w") as f:
                json.dump({}, f)

    async def _read(self) -> Dict:
        async with self._lock:
            with open(self._path, "r") as f:
                return json.load(f)

    async def _write(self, data: Dict):
        async with self._lock:
            with open(self._path, "w") as f:
                json.dump(data, f, indent=2)

    def __getitem__(self, name: str):
        return LocalCollection(self, name)

# ═══════════════════════════════════════════════════════════════════════════════
#  MONGODB CONNECTION MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class MongoDB:
    """Async MongoDB connection manager with Local Fallback."""

    _client: Optional[Any] = None
    _db: Optional[Any] = None
    _is_local: bool = False

    @classmethod
    async def connect(cls, uri: Optional[str] = None, db_name: Optional[str] = None):
        """Initialize Connection. Fallback to local if URI fails."""
        from config import settings
        
        uri = uri or settings.mongodb_uri
        db_name = db_name or settings.mongodb_db
        
        if settings.use_local_db:
            log.info("using_local_db_forced", path=settings.local_db_path)
            cls._is_local = True
            cls._db = LocalDatabase(settings.local_db_path)
            return

        try:
            log.info("connecting_to_mongodb", uri=uri.split("@")[-1]) # Hide credentials
            cls._client = AsyncIOMotorClient(
                uri,
                serverSelectionTimeoutMS=3000,
                connectTimeoutMS=3000,
            )
            # Verify connection
            await cls._client.admin.command("ping")
            cls._db = cls._client[db_name]
            cls._is_local = False
            log.info("mongodb_connected_atlas", db=db_name)
            await cls._create_indexes()
        except Exception as e:
            log.warning("mongodb_atlas_failed_switching_to_local", error=str(e))
            cls._is_local = True
            cls._db = LocalDatabase(settings.local_db_path)
            log.info("local_db_initialized", path=settings.local_db_path)

    @classmethod
    async def _create_indexes(cls):
        """Create required indexes (Cloud only)."""
        if cls._is_local: return
        db = cls.get_db()
        await db.users.create_index("user_id", unique=True)
        await db.feature_profiles.create_index("user_id", unique=True)
        await db.identification_logs.create_index("timestamp")
        await db.trained_models.create_index([("model_type", 1), ("is_active", 1)])
        log.info("mongodb_indexes_created")

    @classmethod
    def get_db(cls):
        if cls._db is None:
            raise RuntimeError("Database not connected.")
        return cls._db

    @classmethod
    def get_collection(cls, name: str):
        return cls.get_db()[name]

    @classmethod
    async def close(cls):
        if cls._client and not cls._is_local:
            cls._client.close()
        cls._client = None
        cls._db = None
        log.info("database_connection_closed")

    @classmethod
    async def is_connected(cls) -> bool:
        if cls._is_local: return True
        try:
            if cls._client:
                await cls._client.admin.command("ping")
                return True
        except: pass
        return False
