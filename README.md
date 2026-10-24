# AI Dataset Negotiator

**Intelligent Dataset Recommendation for Efficient Machine Learning Innovation**  
A full-stack AIML college project that ranks datasets for a machine-learning task using a transparent weighted scoring model.

## Features
- FastAPI REST API for dataset discovery and recommendations
- React + TypeScript + Vite web interface
- Explainable scoring: relevance (35%), data quality (20%), diversity (15%), low redundancy (10%), size fit (10%), compute fit (10%)
- Demo catalog of classification, regression, image and text datasets
- Filters for dataset size and compute budget
- Health endpoint for hosting checks

## Repository structure
- `backend/app` — API, demo catalog and ranking logic
- `backend/tests` — basic recommender tests
- `frontend` — responsive React UI
- `render.yaml` — Render backend service configuration
- `vercel.json` — Vercel frontend build configuration

## Run locally
### Backend
```bash
cd backend
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
API docs: http://localhost:8000/docs

### Frontend
In a second terminal:
```bash
cd frontend
npm install
# Set VITE_API_URL=http://localhost:8000 if needed
npm run dev
```
Open the URL printed by Vite.

## Deploy
### Backend on Render
1. Create a new **Web Service** from this GitHub repository.
2. Set the root directory to `backend`, runtime to Docker, and health-check path to `/health`.
3. Deploy. The API root and `/docs` are available at the Render service URL.

### Frontend on Vercel
1. Import this repository in Vercel.
2. Set the root directory to `frontend`.
3. Build command: `npm run build`; output directory: `dist`.
4. Add environment variable `VITE_API_URL` with the deployed Render URL (no trailing slash), then redeploy.

## Important scope note
This MVP uses a curated demo catalog and heuristic scores. It does not yet crawl external dataset hubs or validate live dataset download links. Those are appropriate future extensions.

## Testing
From the backend folder, run `pytest`.
