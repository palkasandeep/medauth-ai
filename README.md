# MedAuth AI — MedCore Prior Authorization Intelligence System

Full-stack agentic prior-authorization (PA) adjudication and clinical intelligence platform for **MedCore** using **Python (FastAPI + LangGraph + Pydantic + google-genai)** and **React + Tailwind CSS**.

## Local Setup
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
npm --prefix client install
cp .env.example .env
pytest -v
uvicorn server.main:app --reload --port 4000
npm --prefix client run dev
```
