# AI Dataset Negotiator

**Intelligent Dataset Recommendation for Efficient Machine Learning Innovation**  
A full-stack AIML college project that ranks datasets for a machine-learning task using a transparent weighted scoring model.

## Features
- FastAPI REST API for dataset discovery and recommendations
- React + TypeScript + Vite web interface
- Explainable scoring: relevance (35%), data quality (20%), diversity (15%), low redundancy (10%), size fit (10%), compute fit (10%)
- Built-in demo catalog plus live metadata search from Hugging Face and UCI
- Explainable scoring across built-in and externally discovered datasets
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
4. If the frontend and backend are deployed separately, set `VITE_API_URL` to the deployed backend URL (no trailing slash), then redeploy.

## External dataset sources
- Hugging Face public dataset metadata API: queried at recommendation time; public metadata search does not require a token.
- UCI Machine Learning Repository public catalogue: queried at recommendation time where the public catalogue API responds in the expected format.
- Kaggle is not connected yet because its API requires a user-generated token. Never commit API tokens to GitHub; configure them as host environment variables when adding this provider.

External metadata may not include reliable row counts, feature counts, licences, or quality statistics. Missing values are treated as unknown/estimated for ranking, not as verified measurements. The app recommends dataset pages; it does not automatically download or audit dataset contents. If an external source is unavailable, the built-in catalogue still works.

## Testing
From the backend folder, run `pytest`.
