"""預測 API — 股價籤筒抽籤

本模組處理：
- GET /api/draw — 完整預測（含信心度、門檻）
- GET /api/predict — 精簡預測（僅 label、proba）
"""
import logging
from enum import Enum
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api", tags=["predict"])

# 在模組內定義 ModelName，避免循環導入
class ModelName(str, Enum):
    rf = "rf"
    lr = "lr"

@router.get("/draw")
def draw(model: ModelName = ModelName.rf, symbol: str | None = None):
    """執行單次預測（便利版）。
    - 需要提供 symbol。
    - 若 CSV 不存在：依 ENABLE_AUTO_BUILD_PREDICT 環境變數決定是否自動建置（預設啟用）。
    - 回傳：{model, label, proba, symbol, threshold, confidence}
    """
    # 延遲導入避免循環依賴
    import main
    
    # 必須指定 symbol（僅使用個股 CSV）
    if not symbol:
        raise HTTPException(status_code=400, detail="請提供 symbol 參數，例如 ?symbol=AAPL")

    # 先解析 CSV，再帶入預測
    csvp = main._resolve_csv_for(symbol, auto_build_csv=main.ENABLE_AUTO_BUILD_PREDICT)
    res = main._run_predict(csvp, model.value, symbol)

    # 讀取對應模型的 threshold 並計算信心度
    thr = main._load_threshold(model.value)
    conf = None
    try:
        proba = res.get("proba")
        if thr is not None and proba is not None:
            conf = abs(float(proba) - float(thr))
    except Exception:
        conf = None

    # 選擇性記錄
    logging.info(
        "[draw] model=%s symbol=%s proba=%s threshold=%s confidence=%s",
        res.get("model"), symbol, res.get("proba"), thr, conf,
    )

    res.update({"threshold": thr, "confidence": conf})
    return res


@router.get("/predict")
def predict_min(model: ModelName = ModelName.rf, symbol: str | None = None):
    """簡化版單次預測。

    設計目標：
    - 僅回傳最小必需欄位：label、proba（以及 model 供除錯）。
    - 不自動建置缺少的 CSV；若無資料則回 404，請先呼叫 /api/build_symbol。
    - 以標準 HTTP 例外回應錯誤碼，不含額外文案。
    """
    # 延遲導入避免循環依賴
    import main
    
    if not symbol:
        raise HTTPException(status_code=400, detail="請提供 symbol 參數，例如 ?symbol=AAPL")

    # 僅使用既有 CSV，不嘗試自動建置
    csvp = main._resolve_csv_for(symbol, auto_build_csv=False)
    res = main._run_predict(csvp, model.value, symbol)
    # 與 /api/draw 輕微不同：最小欄位需求（label, proba, model）
    return {"label": res["label"], "proba": res["proba"], "model": res["model"]}
