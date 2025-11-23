"""
分批建置股票CSV - 智能處理 API 限制

這個腳本會：
1. 自動檢測哪些股票還沒建置
2. 分批處理（預設每批 5 支）
3. 遇到 API 限制時自動等待
4. 可以隨時中斷並從上次進度繼續

用法：
    # 更新現有股票（掃描 data/ 目錄）
    python batch_build.py
    
    # 指定要建置的股票
    python batch_build.py --symbols AAPL,MSFT,NVDA,GOOGL
    
    # 續建未完成的股票
    python batch_build.py --symbols AAPL,MSFT,NVDA --resume
    
    # 自訂批次大小與等待時間
    python batch_build.py --symbols AAPL,MSFT --batch-size 3 --wait-time 70
"""

import argparse
from pathlib import Path
import asyncio
from datetime import datetime
import sys
import time

from stock import _ensure_td, _build_from_td

DATA_DIR = Path(__file__).parent / "data"


def get_all_us_symbols() -> list:
    """取得所有已存在的美股代碼（從 data/ 目錄掃描）"""
    symbols = set()
    try:
        for csv_file in DATA_DIR.glob("*_short_term_with_lag3.csv"):
            symbol = csv_file.stem.replace("_short_term_with_lag3", "")
            # 只保留美股（排除純數字的台股代碼）
            if not symbol.isdigit():
                symbols.add(symbol.upper())
    except Exception as e:
        print(f"⚠️  掃描目錄失敗：{e}")
    
    return sorted(list(symbols))


def get_missing_symbols(all_symbols: list) -> list:
    """找出還沒建置的股票"""
    missing = []
    for symbol in all_symbols:
        csv_file = DATA_DIR / f"{symbol}_short_term_with_lag3.csv"
        if not csv_file.exists():
            missing.append(symbol)
    return missing


def filter_us_stocks(symbols: list) -> list:
    """過濾出美股代碼（排除純數字的台股代碼）"""
    us_stocks = [s for s in symbols if not s.isdigit()]
    tw_stocks = [s for s in symbols if s.isdigit()]
    
    if tw_stocks:
        print(f"⚠️  跳過 {len(tw_stocks)} 支台股代碼：{', '.join(tw_stocks)}")
    
    return us_stocks


async def build_one_symbol(symbol: str, start_date: str, semaphore: asyncio.Semaphore):
    """建置單一股票的CSV"""
    async with semaphore:
        csv_path = DATA_DIR / f"{symbol}_short_term_with_lag3.csv"
        try:
            # 在執行緒中執行阻塞的建置操作
            await asyncio.to_thread(
                lambda: _build_from_td(symbol=symbol, out_csv=csv_path, start=start_date)
            )
            
            # 記錄更新時間
            update_file = DATA_DIR / f"{symbol}_last_update.txt"
            update_file.write_text(datetime.now().isoformat(), encoding="utf-8")
            
            # 取得檔案大小
            size_kb = csv_path.stat().st_size / 1024
            
            return {
                "symbol": symbol,
                "status": "success",
                "path": str(csv_path),
                "size_kb": size_kb
            }
        except Exception as e:
            return {"symbol": symbol, "status": "error", "error": str(e)}


async def build_batch(symbols: list, start_date: str, batch_num: int, total_batches: int):
    """建置一批股票"""
    semaphore = asyncio.Semaphore(len(symbols))  # 同時處理整批
    
    print(f"\n{'='*70}")
    print(f"📦 批次 {batch_num}/{total_batches} - 建置 {len(symbols)} 支股票")
    print(f"{'='*70}")
    
    tasks = [
        build_one_symbol(symbol, start_date, semaphore)
        for symbol in symbols
    ]
    
    results = []
    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        result = await coro
        status_icon = "✓" if result["status"] == "success" else "✗"
        
        print(f"[{i:2d}/{len(symbols)}] {status_icon} {result['symbol']:8s}", end="")
        
        if result["status"] == "error":
            error_msg = result['error'][:60]
            print(f" {error_msg}")
        else:
            size = result.get('size_kb', 0)
            print(f" {size:6.1f} KB")
        
        results.append(result)
    
    success_count = sum(1 for r in results if r["status"] == "success")
    print(f"\n批次完成：成功 {success_count}/{len(symbols)} 支")
    
    return results


async def build_all_batches(symbols: list, batch_size: int, wait_time: int, start_date: str):
    """分批建置所有股票"""
    # 初始化 API
    print("🔧 初始化 Twelve Data API...")
    try:
        await asyncio.to_thread(_ensure_td)
        print("✓ API 已就緒\n")
    except Exception as e:
        print(f"✗ 初始化失敗：{e}")
        return []
    
    # 分批
    batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
    total_batches = len(batches)
    
    print(f"📊 總計 {len(symbols)} 支股票，分為 {total_batches} 批")
    print(f"   每批 {batch_size} 支，批次間等待 {wait_time} 秒")
    
    all_results = []
    
    for batch_num, batch_symbols in enumerate(batches, 1):
        batch_results = await build_batch(
            batch_symbols, 
            start_date, 
            batch_num, 
            total_batches
        )
        all_results.extend(batch_results)
        
        # 檢查是否有 API 限制錯誤
        has_rate_limit = any(
            "run out of API credits" in r.get("error", "")
            for r in batch_results
            if r["status"] == "error"
        )
        
        # 如果不是最後一批，則等待
        if batch_num < total_batches:
            if has_rate_limit:
                print(f"\n⏳ 遇到 API 限制，等待 {wait_time} 秒後繼續...")
            else:
                print(f"\n⏳ 等待 {wait_time} 秒後處理下一批...")
            
            # 倒數計時
            for remaining in range(wait_time, 0, -5):
                print(f"   還剩 {remaining} 秒...", end="\r")
                await asyncio.sleep(min(5, remaining))
            print(" " * 30, end="\r")  # 清除倒數顯示
    
    return all_results


def print_summary(results: list, start_time: datetime):
    """列印最終結果"""
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = len(results) - success_count
    total_time = (datetime.now() - start_time).total_seconds()
    
    total_size_kb = sum(r.get('size_kb', 0) for r in results if r["status"] == "success")
    
    print(f"\n{'='*70}")
    print("🎯 全部完成！")
    print(f"{'='*70}")
    print(f"✓ 成功：{success_count:3d} 支")
    print(f"✗ 失敗：{error_count:3d} 支")
    print(f"📦 總大小：{total_size_kb/1024:.2f} MB")
    print(f"⏱️  總耗時：{int(total_time/60)}:{int(total_time%60):02d}")
    
    if error_count > 0:
        print(f"\n❌ 失敗的股票：")
        for r in results:
            if r["status"] == "error":
                error_msg = r['error'][:60]
                print(f"   {r['symbol']:8s}: {error_msg}")
    
    print(f"\n📁 檔案位置：{DATA_DIR.resolve()}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="分批建置股票CSV")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="每批建置幾支股票（預設：5）"
    )
    parser.add_argument(
        "--wait-time",
        type=int,
        default=65,
        help="每批之間等待秒數（預設：65）"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01",
        help="資料起始日期（預設：2020-01-01）"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="只建置尚未完成的股票"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="指定股票代碼（逗號分隔），例如：AAPL,MSFT,NVDA"
    )
    
    args = parser.parse_args()
    
    # 取得股票列表
    if args.symbols:
        # 使用者指定的股票
        all_symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    else:
        # 從現有 data/ 目錄掃描
        all_symbols = get_all_us_symbols()
        if not all_symbols:
            print("❌ data/ 目錄中沒有找到任何股票資料")
            print("💡 請使用 --symbols 參數指定要建置的股票，例如：")
            print("   python batch_build.py --symbols AAPL,MSFT,NVDA")
            sys.exit(1)
    
    # 過濾美股
    us_symbols = filter_us_stocks(all_symbols)
    
    # 如果是續建模式，只建置缺少的
    if args.resume:
        missing = get_missing_symbols(us_symbols)
        if not missing:
            print("✓ 所有股票都已建置完成！")
            sys.exit(0)
        print(f"📋 找到 {len(missing)} 支尚未建置的股票")
        us_symbols = missing
    
    print(f"\n{'='*70}")
    print(f"📊 準備建置 {len(us_symbols)} 支美股")
    print(f"{'='*70}")
    print(f"每批：{args.batch_size} 支")
    print(f"等待時間：{args.wait_time} 秒")
    print(f"預估總時間：約 {len(us_symbols) // args.batch_size * args.wait_time / 60:.0f} 分鐘")
    print(f"{'='*70}\n")
    
    # 確認
    try:
        response = input("確定要開始嗎？(y/N): ")
        if response.lower() != 'y':
            print("取消執行")
            sys.exit(0)
    except (KeyboardInterrupt, EOFError):
        print("\n取消執行")
        sys.exit(0)
    
    # 執行
    start_time = datetime.now()
    try:
        results = asyncio.run(
            build_all_batches(
                us_symbols,
                args.batch_size,
                args.wait_time,
                args.start_date
            )
        )
        print_summary(results, start_time)
        
        # 返回狀態
        error_count = sum(1 for r in results if r["status"] == "error")
        sys.exit(0 if error_count == 0 else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  使用者中斷執行")
        print(f"💡 提示：下次可以使用 --resume 參數繼續建置未完成的股票")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ 發生錯誤：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
