from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from fusion.config import get_settings

settings = get_settings()
engine = create_engine(settings.database_url, echo=settings.database_echo)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    import fusion.models.orm  # noqa: F401
    Base.metadata.create_all(bind=engine)
