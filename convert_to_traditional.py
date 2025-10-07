#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批次將 docs/ 目錄下的所有可讀文字檔與其檔名、資料夾名稱，從簡體中文轉換為繁體中文。

特性：
- 優先使用 OpenCC 進行高品質轉換（s2t），若未安裝則退回字典替換。
- 遞迴處理 docs/ 目錄中所有文字檔（安全跳過二進位檔，例如圖片）。
- 先轉換檔案內容，再將檔名與資料夾名稱改為繁體（避免走訪時路徑失效）。
- 基本的名稱衝突處理（若目標名稱已存在，會在名稱後加上 _trad 避免覆蓋）。
"""
import os
from pathlib import Path
from typing import Iterable, Tuple

try:
    # opencc-python-reimplemented 提供純 Python 實作，較易於跨平台安裝
    from opencc import OpenCC  # type: ignore
    _OPENCC_AVAILABLE = True
except Exception:
    OpenCC = None  # type: ignore
    _OPENCC_AVAILABLE = False

# 簡體到繁體的常用映射表（擴展版）
CONVERSION_MAP = {
    # 基礎字詞
    '项目': '專案', '简介': '簡介', '构建': '建構', '预测': '預測', '系统': '系統',
    '提供': '提供', '数据': '資料', '采集': '採集', '特征': '特徵', '工程': '工程',
    '模型': '模型', '训练': '訓練', '推理': '推理', '完整': '完整', '解决': '解決',
    '方案': '方案', '技术': '技術', '栈': '棧', '后端': '後端', '框架': '框架',
    '现代': '現代', '化': '化', '异步': '非同步', '服务': '服務', '器': '器',
    '运行': '執行', '时': '時', '环境': '環境', '处理': '處理', '与': '與',
    '机器': '機器', '学习': '學習', '算法': '演算法', '数值': '數值', '计算': '計算',
    '科学': '科學', '统计': '統計', '建模': '建模', '监控': '監控', '观测': '可觀測',
    '性': '性', '容器': '容器', '测试': '測試', '单元': '單元', '集成': '整合',
    
    # 架構相關
    '架构': '架構', '概览': '概覽', '层': '層', '逻辑': '邏輯', '持久': '持久',
    '分': '分', '设计': '設計', '决策': '決策', '部署': '部署', '扩展': '擴展',
    
    # 數據相關
    '数据': '資料', '模型': '模型', '实体': '實體', '定义': '定義', '字段': '欄位',
    '关联': '關聯', '关系': '關係', '格式': '格式', '文件': '檔案', '生命': '生命',
    '周期': '週期', '质量': '品質', '验证': '驗證', '规则': '規則',
    
    # 業務相關
    '业务': '業務', '规则': '規則', '边界': '邊界', '条件': '條件', '约束': '約束',
    '流程': '流程', '策略': '策略',
    
    # 術語相關
    '术语': '術語', '词汇': '詞彙', '编码': '編碼', '规范': '規範', '命名': '命名',
    '约定': '約定', '注释': '註釋',
    
    # 開發相關
    '开发': '開發', '规范': '規範', '代码': '程式碼', '风格': '風格', '异常': '例外',
    '处理': '處理', '要求': '要求', '测试': '測試', '覆盖': '覆蓋', '率': '率',
    
    # 常見問題
    '常见': '常見', '问题': '問題', '安装': '安裝', '环境': '環境', '失败': '失敗',
    '错误': '錯誤', '解决': '解決', '诊断': '診斷', '方法': '方法',
    
    # 其他常用詞
    '功能': '功能', '核心': '核心', '说明': '說明', '配置': '配置', '参数': '參數',
    '响应': '回應', '请求': '請求', '状态': '狀態', '错误': '錯誤', '成功': '成功',
    '示例': '範例', '使用': '使用', '场景': '場景', '建议': '建議', '注意': '注意',
    '事项': '事項', '步骤': '步驟', '检查': '檢查', '确认': '確認', '验证': '驗證',
    '启动': '啟動', '停止': '停止', '查询': '查詢', '返回': '回傳', '输入': '輸入',
    '输出': '輸出', '类型': '類型', '对象': '物件', '变量': '變數', '函数': '函式',
    '默认': '預設', '值': '值', '选项': '選項', '可选': '可選', '必需': '必需',
    '原因': '原因', '症状': '症狀', '排查': '排查', '修复': '修復', '优化': '最佳化',
    '性能': '效能', '延迟': '延遲', '内存': '記憶體', '占用': '佔用', '泄漏': '洩漏',
    '并发': '並發', '度': '度', '任务': '任務', '队列': '佇列', '线程': '執行緒',
    '进程': '程序', '协程': '協程', '同步': '同步', '异步': '非同步',
    '网络': '網路', '连接': '連線', '超时': '逾時', '重试': '重試', '备份': '備份',
    '恢复': '復原', '操作': '操作', '执行': '執行', '创建': '建立', '删除': '刪除',
    '更新': '更新', '查找': '尋找', '搜索': '搜尋', '过滤': '篩選', '排序': '排序',
    '组合': '組合', '拆分': '拆分', '合并': '合併', '转换': '轉換', '解析': '解析',
    '加载': '載入', '保存': '儲存', '读取': '讀取', '写入': '寫入', '追加': '追加',
    '缓存': '快取', '清理': '清理', '释放': '釋放', '分配': '配置', '资源': '資源',
    '限制': '限制', '阈值': '閾值', '范围': '範圍', '区间': '區間', '集合': '集合',
    '列表': '列表', '字典': '字典', '数组': '陣列', '结构': '結構', '实现': '實作',
    '继承': '繼承', '封装': '封裝', '多态': '多型', '接口': '介面', '类': '類別',
    '属性': '屬性', '方法': '方法', '静态': '靜態', '动态': '動態', '公开': '公開',
    '私有': '私有', '保护': '保護', '抽象': '抽象', '具体': '具體', '泛型': '泛型',
}

# 可處理的預設文字副檔名（仍會以實際能否以 UTF-8 讀取為準）
TEXT_EXTS = {
    '.md', '.markdown', '.txt', '.rst', '.html', '.htm', '.xml', '.json', '.yml', '.yaml', '.csv', '.ini', '.cfg'
}

def _get_converter():
    """取得轉換器，OpenCC 優先，否則使用字典替換。"""
    if _OPENCC_AVAILABLE:
        try:
            return OpenCC('s2t')  # 簡體到繁體
        except Exception:
            pass
    # 回退：以簡單字典替換
    class _DictConv:
        def convert(self, s: str) -> str:
            result = s
            for simp, trad in CONVERSION_MAP.items():
                result = result.replace(simp, trad)
            return result

    return _DictConv()


def is_text_file(path: Path) -> bool:
    """粗略判斷是否為文字檔：
    - 副檔名在 TEXT_EXTS 中快速通過
    - 否則嘗試讀取小塊 bytes 並以 UTF-8 解碼
    """
    if path.suffix.lower() in TEXT_EXTS:
        return True
    try:
        with open(path, 'rb') as f:
            chunk = f.read(4096)
        # 如果包含大量 NUL 或無法解碼，多半是二進位
        if b"\x00" in chunk:
            return False
        chunk.decode('utf-8')
        return True
    except Exception:
        return False


def convert_content(content: str, converter) -> str:
    """將內容從簡體轉換為繁體。"""
    return converter.convert(content)

def process_file(file_path: Path, converter) -> bool:
    """處理單個檔案，回傳是否成功。"""
    rel = file_path
    print(f"處理內容: {rel}")
    try:
        if not is_text_file(file_path):
            print(f"- 跳過非文字檔: {file_path.name}")
            return False
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        converted = convert_content(content, converter)
        if converted != content:
            with open(file_path, 'w', encoding='utf-8', newline='') as f:
                f.write(converted)
        print(f"✓ 內容完成: {file_path.name}")
        return True
    except UnicodeDecodeError:
        print(f"- 跳過（解碼失敗）: {file_path.name}")
        return False
    except Exception as e:
        print(f"✗ 內容錯誤 {file_path.name}: {e}")
        return False


def convert_name(name: str, converter) -> str:
    """將檔名/資料夾名（不含副檔名變更）轉為繁體。"""
    # 只處理 base 名稱的中文，副檔名（如 .md）保持不變
    stem, suffix = os.path.splitext(name)
    new_stem = converter.convert(stem)
    return f"{new_stem}{suffix}"


def safe_rename(src: Path, dst: Path) -> Tuple[bool, Path]:
    """安全改名：若目標存在則加上 _trad 避免覆蓋。
    回傳 (是否改名, 最終路徑)
    """
    if src == dst:
        return False, src
    final_dst = dst
    if final_dst.exists():
        # 若已存在且不是同一路徑，嘗試加後綴
        parent = dst.parent
        stem, suffix = os.path.splitext(dst.name)
        i = 1
        while final_dst.exists():
            final_dst = parent / f"{stem}_trad{i}{suffix}"
            i += 1
    try:
        src.rename(final_dst)
        print(f"✓ 重新命名: {src.name} -> {final_dst.name}")
        return True, final_dst
    except Exception as e:
        print(f"✗ 重新命名失敗 {src.name}: {e}")
        return False, src

def main():
    """主函式"""
    docs_dir = Path(__file__).parent / 'docs'
    if not docs_dir.exists():
        print(f"錯誤: docs 目錄不存在: {docs_dir}")
        return

    converter = _get_converter()

    print("開始處理檔案內容（遞迴）…")
    print("=" * 60)

    total_files = 0
    changed_files = 0
    for root, dirs, files in os.walk(docs_dir):
        root_path = Path(root)
        for fn in files:
            fp = root_path / fn
            total_files += 1
            if process_file(fp, converter):
                changed_files += 1

    print("=" * 60)
    print(f"內容處理完成：{changed_files}/{total_files} 個檔案已處理或確認為文字檔")

    # 檔名與資料夾改名（先檔案、再資料夾；資料夾自底向上）
    print("開始重新命名檔案…")
    renamed_files = 0
    for root, dirs, files in os.walk(docs_dir):
        root_path = Path(root)
        for fn in files:
            new_name = convert_name(fn, converter)
            if new_name != fn:
                src = root_path / fn
                dst = root_path / new_name
                ok, _ = safe_rename(src, dst)
                if ok:
                    renamed_files += 1

    print("開始重新命名資料夾（自底向上）…")
    renamed_dirs = 0
    for root, dirs, files in os.walk(docs_dir, topdown=False):
        root_path = Path(root)
        for d in list(dirs):
            new_name = convert_name(d, converter)
            if new_name != d:
                src = root_path / d
                dst = root_path / new_name
                ok, _ = safe_rename(src, dst)
                if ok:
                    renamed_dirs += 1

    print("=" * 60)
    print(f"重新命名完成：檔案 {renamed_files} 個、資料夾 {renamed_dirs} 個")
    print("全部轉換完成！")

if __name__ == '__main__':
    main()
