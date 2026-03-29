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
    create_engine, Column, Integer, Float, String, Boolean, DateTime
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from config.settings import DATABASE_URL
from src.utils.logger import logger

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


def _build_engine():
    """
    Build SQLAlchemy engine. Uses PostgreSQL if available, otherwise falls back
    to SQLite (for testing / environments without PostgreSQL).
    """
    db_url = DATABASE_URL
    is_postgres = db_url.startswith("postgresql")

    try:
        if is_postgres:
            eng = create_engine(
                db_url,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                echo=False,
            )
        else:
            eng = create_engine(db_url, echo=False)
        return eng
    except Exception as e:
        logger.warning(
            f"Could not create database engine with '{db_url}': {e}. "
            "Falling back to in-memory SQLite — prediction logging will not persist."
        )
        return create_engine("sqlite:///:memory:", echo=False)


# Lazy engine: built on first access so import does not fail when psycopg2 is absent.
_engine = None
_SessionLocal = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


def _get_session_local():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_get_engine())
    return _SessionLocal


# Convenience alias: engine is the lazy-loaded engine instance
def get_engine():
    """Return the SQLAlchemy engine, building it lazily on first call."""
    return _get_engine()

SessionLocal = None  # Kept for backwards compat; use _get_session_local() internally


def create_tables() -> None:
    """
    Create all database tables if they don't exist.
    Safe to call multiple times (CREATE TABLE IF NOT EXISTS semantics).
    """
    try:
        Base.metadata.create_all(bind=_get_engine())
        logger.info("Database tables created/verified successfully")
    except SQLAlchemyError as e:
        logger.error(f"Database table creation failed: {e}")
        logger.warning("Prediction logging will be disabled. Check DATABASE_URL in .env")
    except Exception as e:
        logger.warning(f"Database unavailable: {e}. Prediction logging disabled.")


def get_db():
    """
    FastAPI dependency that yields a database session.
    Ensures the session is always closed after the request completes.

    Usage in route:
        @router.post("/predict/fraud")
        async def predict(db: Session = Depends(get_db)):
            ...
    """
    SessionFactory = _get_session_local()
    db = SessionFactory()
    try:
        yield db
    finally:
        db.close()
