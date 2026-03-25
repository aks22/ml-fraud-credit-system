"""
src/api/database.py
--------------------
PostgreSQL database setup using SQLAlchemy.

Tables:
  - fraud_predictions: One row per fraud detection request
  - credit_predictions: One row per credit risk request

All predictions are logged to enable:
  1. Model monitoring (drift detection over time)
  2. Audit trail (regulatory compliance)
  3. Performance dashboards (Grafana integration)

Usage:
    from src.api.database import engine, SessionLocal, create_tables
    create_tables()
    with SessionLocal() as session:
        session.add(prediction_log)
        session.commit()
"""

from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Boolean, DateTime, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from config.settings import DATABASE_URL
from src.utils.logger import logger

# SQLAlchemy engine and session factory
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,        # Test connection before using from pool
    pool_size=10,              # Number of persistent connections
    max_overflow=20,           # Extra connections allowed beyond pool_size
    echo=False,                # Set True to log all SQL queries (debug only)
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── ORM Models ────────────────────────────────────────────────────────────────

class FraudPredictionLog(Base):
    """Logs every fraud detection prediction made by the API."""
    __tablename__ = "fraud_predictions"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(36), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Input features (subset — log what's needed for monitoring)
    step = Column(Integer)
    transaction_type = Column(String(20))
    amount = Column(Float)
    oldbalance_org = Column(Float)
    newbalance_orig = Column(Float)
    oldbalance_dest = Column(Float)
    newbalance_dest = Column(Float)

    # Prediction output
    fraud_probability = Column(Float, nullable=False)
    is_fraud = Column(Boolean, nullable=False)
    threshold = Column(Float, default=0.5)
    model_version = Column(String(50))


class CreditPredictionLog(Base):
    """Logs every credit risk scoring prediction made by the API."""
    __tablename__ = "credit_predictions"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(36), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Input features
    age = Column(Integer)
    duration_months = Column(Integer)
    credit_amount = Column(Float)
    employment = Column(String(50))
    purpose = Column(String(50))

    # Prediction output
    credit_tier = Column(String(20), nullable=False)
    confidence_good = Column(Float)
    confidence_moderate = Column(Float)
    confidence_poor = Column(Float)
    model_version = Column(String(50))


def create_tables() -> None:
    """
    Create all database tables if they don't exist.
    Safe to call multiple times (CREATE TABLE IF NOT EXISTS semantics).
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created/verified successfully")
    except SQLAlchemyError as e:
        logger.error(f"Database table creation failed: {e}")
        logger.warning("Prediction logging will be disabled. Check DATABASE_URL in .env")


def get_db():
    """
    FastAPI dependency that yields a database session.
    Ensures the session is always closed after the request completes.

    Usage in route:
        @router.post("/predict/fraud")
        async def predict(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
