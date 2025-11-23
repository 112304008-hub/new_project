"""首頁路由 — 回傳 template2.html
"""
from fastapi import APIRouter
from fastapi.responses import FileResponse
from pathlib import Path

router = APIRouter(tags=["home"])

# 路徑配置
ROOT = Path(__file__).parent.parent
HTML = ROOT / "template2.html"


@router.get("/")
def home():
    """首頁：直接回傳 `template2.html`（籤筒網頁）。"""
    return FileResponse(HTML)
