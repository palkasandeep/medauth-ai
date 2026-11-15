FROM node:22-slim AS frontend
WORKDIR /app/client
COPY client/package.json client/package-lock.json* ./
RUN npm install
COPY client/ ./
RUN npm run build

FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY server/ ./server/
COPY --from=frontend /app/client/dist ./client/dist
ENV PYTHONUNBUFFERED=1
ENV PORT=4000
EXPOSE 4000
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "4000"]
