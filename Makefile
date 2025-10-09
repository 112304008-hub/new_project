# Makefile — 常用開發/維運指令（Windows 可用 `make <target>`，若無 make 請參考命令內容）

PY?=python
UVICORN?=uvicorn
PORT?=8000
HOST?=0.0.0.0

.PHONY: help install dev run api docs lint test cov format build train-rf train-all ddns summary \
	docker-build docker-run docker-stop docker-logs docker-shell docker-health

help:
	@echo "可用目標:"
	@echo "  install     安裝 requirements.txt"
	@echo "  dev         啟動開發伺服器 (reload)"
	@echo "  run         啟動正式伺服器 (無 reload)"
	@echo "  api         用瀏覽器開啟 docs"
	@echo "  train-rf    訓練隨機森林模型"
	@echo "  train-all   訓練 rf + lr"
	@echo "  bulk-sp500  批次建置 S&P500 前 50"
	@echo "  tests       執行測試"
	@echo "  cov         產生覆蓋率報告 htmlcov/"
	@echo "  summary     顯示 models/ 與 data/ 檔案概況"
	@echo "  docker-build  建置 Docker 映像 (tag: new_project:latest)"
	@echo "  docker-run    以容器執行並映射 8000 連接埠"
	@echo "  docker-stop   停止名為 new_project 的容器"
	@echo "  docker-logs   追蹤容器日誌"
	@echo "  docker-shell  進入容器 sh"
	@echo "  docker-health 用容器內 /health 檢查健康狀態"

install:
	$(PY) -m pip install -r requirements.txt

dev:
	$(UVICORN) main:app --reload --host $(HOST) --port $(PORT)

run:
	$(UVICORN) main:app --host $(HOST) --port $(PORT)

api:
	start http://localhost:$(PORT)/docs

train-rf:
	$(PY) stock.py --train --model rf

train-all:
	$(PY) stock.py --train --model all

bulk-sp500:
	$(PY) -m scripts.batch.fetch_sp500_github

tests:
	$(PY) -m pytest -q tests

cov:
	$(PY) -m pytest --cov=. --cov-report=html tests
	@echo "HTML 覆蓋率報告: htmlcov/index.html"

summary:
	@echo "=== models/ ===" && dir models 2> NUL || echo "(無 models 目錄)"
	@echo "=== data/ (top 10) ===" && dir data | find /V "<DIR>" | head -n 10

# ==== Docker helpers ====
docker-build:
	docker build -t new_project:latest .

docker-run: docker-stop
	docker run --rm -d -p 8000:8000 --name new_project new_project:latest

docker-stop:
	-@docker stop new_project 2> NUL || exit 0

docker-logs:
	docker logs -f new_project

docker-shell:
	docker run --rm -it --entrypoint /bin/sh new_project:latest

docker-health:
	powershell -NoProfile -Command "try { $r = iwr http://localhost:8000/health -TimeoutSec 3; if ($r.StatusCode -eq 200) { 'OK'; exit 0 } else { 'BAD'; exit 1 } } catch { 'UNREACHABLE'; exit 1 }"
