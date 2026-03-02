from sqlalchemy import create_engine, Column, Integer, String, JSON, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, UTC

DATABASE_URL='postgresql://postgres:2003@localhost:5432/chat_summarizer'

engine=create_engine(DATABASE_URL)
SessionLocal=sessionmaker(bind=engine)

Base=declarative_base()

class ChatSummary(Base):
    __tablename__='chat_summaries'

    id=Column(Integer,primary_key=True,index=True)
    chat_id=Column(String,index=True)
    raw_chat=Column(JSON)
    generated_summary=Column(JSON)
    token_usage=Column(Integer)
    processing_time=Column(Float)
    created_at=Column(DateTime,default=datetime.now(UTC))

def init_db():
    Base.metadata.create_all(bind=engine)