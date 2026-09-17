from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import os
import sys
import uuid
import shutil
from datetime import timedelta
import torch
import torchvision.transforms as transforms
from PIL import Image

# Setup paths for ML models
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from backend.database import get_db, Base, engine
from backend.models import User, Scan
from backend.schemas import UserCreate, UserResponse, Token, ScanResponse
from backend.auth import get_password_hash, verify_password, create_access_token, get_current_user, ACCESS_TOKEN_EXPIRE_MINUTES

# Import ML Pipeline Components
from ml.src.models import build_diagnostic_model, load_model, freeze_model
from ml.src.purifier import PurifierCNN
from ml.src.pipeline import AdvShieldPipeline

app = FastAPI(title="AdvShield-Med API")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global ML State ---
ml_pipeline = None
device = None
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.on_event("startup")
async def startup_event():
    global ml_pipeline, device
    print("Loading PyTorch Models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    diag_model = load_model(build_diagnostic_model, os.path.join(PROJECT_ROOT, 'ml', 'checkpoints', 'diagnostic_resnet18.pth'), device)
    diag_model = freeze_model(diag_model)
    
    gatekeeper = load_model(build_diagnostic_model, os.path.join(PROJECT_ROOT, 'ml', 'checkpoints', 'gatekeeper_resnet18.pth'), device)
    gatekeeper = freeze_model(gatekeeper)
    
    purifier = PurifierCNN().to(device)
    purifier.load_state_dict(torch.load(os.path.join(PROJECT_ROOT, 'ml', 'checkpoints', 'purifier.pth')))
    purifier.eval()
    
    ml_pipeline = AdvShieldPipeline(
        gatekeeper=gatekeeper,
        diagnostic=diag_model,
        purifier=purifier,
        device=device
    )
    print("AdvShieldPipeline successfully loaded and ready for inference!")

# --- Auth Routes ---
@app.post("/auth/register", response_model=UserResponse)
async def register(user: UserCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).filter(User.username == user.username))
    if result.scalars().first():
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = get_password_hash(user.password)
    db_user = User(username=user.username, password_hash=hashed_password)
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user

@app.post("/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).filter(User.username == form_data.username))
    user = result.scalars().first()
    
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/auth/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

# --- Core ML Route ---
@app.post("/api/scan", response_model=ScanResponse)
async def process_scan(patient_name: str, file: UploadFile = File(...), current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    # 1. Save uploaded image
    file_ext = file.filename.split('.')[-1]
    unique_id = str(uuid.uuid4())
    orig_filename = f"{unique_id}_orig.{file_ext}"
    orig_path = os.path.join(PROJECT_ROOT, "uploads", "originals", orig_filename)
    
    with open(orig_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 2. Preprocess image
    image = Image.open(orig_path).convert('RGB')
    tensor_img = transform(image).unsqueeze(0).to(device)
    
    # 3. Run Pipeline Inference
    with torch.no_grad():
        result = ml_pipeline.classify(tensor_img)
        
    decision = result['decision']
    confidence = result['gate_confidence']
    diagnosis_idx = result['diagnosis']
    diagnosis_label = "Pneumonia" if diagnosis_idx == 1 else "Normal"
    
    # For now, we mock the heatmap path since Grad-CAM needs an active backward pass,
    # but the pipeline runs in eval/no_grad mode here.
    # We will implement real Grad-CAM extraction in Phase 7.2!
    heatmap_filename = f"{unique_id}_heatmap.png"
    heatmap_path = os.path.join(PROJECT_ROOT, "uploads", "heatmaps", heatmap_filename)
    
    # Just copy the original as a placeholder for the heatmap for now
    shutil.copy(orig_path, heatmap_path)

    # 4. Save to Database
    db_scan = Scan(
        user_id=current_user.id,
        patient_name=patient_name,
        original_image_path=f"/uploads/originals/{orig_filename}",
        heatmap_image_path=f"/uploads/heatmaps/{heatmap_filename}",
        gatekeeper_flag=(decision == "adversarial"),
        gatekeeper_confidence=confidence,
        diagnosis=diagnosis_label
    )
    
    db.add(db_scan)
    await db.commit()
    await db.refresh(db_scan)
    
    return db_scan

@app.get("/api/scans/history", response_model=list[ScanResponse])
async def get_history(current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Scan).filter(Scan.user_id == current_user.id).order_by(Scan.timestamp.desc()))
    return result.scalars().all()
