#!/bin/bash
echo "🚀 Initializing AdvShield-Med Full Stack Architecture (Metis-Style)..."

# 1. Reorganize existing ML code to match Metis structure (moving into ml/ folder)
echo "📁 Reorganizing ML directories..."
mkdir -p ml
# Move existing ML-related folders into ml/ if they aren't already there
[ -d "src" ] && mv src ml/
[ -d "notebooks" ] && mv notebooks ml/
[ -d "checkpoints" ] && mv checkpoints ml/
[ -d "training_plots" ] && mv training_plots ml/
[ -d "results" ] && mv results ml/
echo "✅ ML directories moved."

# 2. Setup Backend (backend/)
echo "⚙️ Setting up FastAPI Backend..."
mkdir -p backend
mv server/requirements.txt backend/ 2>/dev/null || true
rm -rf server/

# Install backend dependencies (assumes venv is active)
echo "📦 Installing Python dependencies..."
pip install -r backend/requirements.txt
echo "✅ Backend setup complete."

# 3. Setup Frontend (frontend/)
echo "🎨 Scaffolding React Frontend (Vite + Tailwind)..."
# Remove existing frontend if it failed previously
rm -rf frontend/
npm create vite@latest frontend -- --template react-ts

cd frontend
echo "📦 Installing npm dependencies..."
npm install
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
npm install clsx tailwind-merge lucide-react framer-motion react-router-dom
cd ..
echo "✅ Frontend scaffolding complete."

# 4. Start Database
echo "🗄️ Starting PostgreSQL Database..."
docker-compose up -d
echo "✅ Database started."

echo "🎉 Architecture Initialization Complete!"
