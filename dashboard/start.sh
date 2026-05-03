#!/usr/bin/env bash
# FinRL Dashboard — start backend + frontend together
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "▶  Starting FastAPI backend on http://localhost:8000 …"
cd "$ROOT/backend"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

echo "▶  Starting Vite dev server on http://localhost:5173 …"
cd "$ROOT/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "  Dashboard: http://localhost:5173"
echo "  API docs:  http://localhost:8000/docs"
echo ""
echo "  Press Ctrl+C to stop both servers."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
wait
