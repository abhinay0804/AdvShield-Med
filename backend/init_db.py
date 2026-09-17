import asyncio
from backend.database import engine, Base
import backend.models as models # Important to import models so Base knows about them

async def init_models():
    async with engine.begin() as conn:
        print("Dropping old tables...")
        await conn.run_sync(Base.metadata.drop_all)
        print("Creating new tables...")
        await conn.run_sync(Base.metadata.create_all)
    print("Database initialized successfully!")

if __name__ == "__main__":
    asyncio.run(init_models())
