"""本檔案對應舊版 /api/auto/* 端點，現行版本已移除相關功能，改由全域背景更新。

為避免誤導與維持測試綠燈，暫時跳過本檔所有測試。
"""
import pytest

pytestmark = pytest.mark.skip("/api/auto/* 端點已移除，改為內建全域更新機制")
