.PHONY: install backend frontend test lint docker-up docker-down demo

install:
	cd backend && pip install -r requirements.txt --break-system-packages
	cd frontend && npm install

backend:
	cd backend && PYTHONPATH=.. uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm run dev

test:
	cd backend && PYTHONPATH=.. pytest -v
	PYTHONPATH=. python -m pytest ml/tests -v
	cd frontend && npm run test

lint:
	cd backend && python -m py_compile $$(find app -name '*.py')
	cd frontend && npm run lint

docker-up:
	docker compose up --build

docker-down:
	docker compose down

demo:
	cd backend && PYTHONPATH=.. python ../scripts/generate_demo_data.py
