from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Float
from sqlalchemy.orm import relationship
import datetime
from .database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="Doctor")
    
    scans = relationship("Scan", back_populates="doctor")

class Scan(Base):
    __tablename__ = "scans"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    patient_name = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    original_image_path = Column(String, nullable=False)
    heatmap_image_path = Column(String, nullable=False)
    
    gatekeeper_flag = Column(Boolean, default=False)
    gatekeeper_confidence = Column(Float, nullable=True)
    
    diagnosis = Column(String, nullable=False) # "Normal" or "Pneumonia"
    
    doctor = relationship("User", back_populates="scans")
