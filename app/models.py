import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import Column, Integer, String, Float, DateTime, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Load environmental configs from .env
load_dotenv()

Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"
    
    # Bookkeeping columns (Auto-managed by database)
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Payload metrics columns 
    prediction = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    model_version = Column(String, nullable=False)

# Safely extract secret strings without variable chaining inside systemd configs
DB_PASSWORD = os.getenv("DB_PASSWORD")
DATABASE_URL = f"postgresql://pixelwise:{DB_PASSWORD}@localhost/pixelwise"

# Initialize lazy-loading connection components
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)