from app.models import Base, engine

print("Initializing PostgreSQL database schemas via SQLAlchemy metadata tracker...")
Base.metadata.create_all(engine)
print("Initialization successful! Table 'predictions' materialized.")