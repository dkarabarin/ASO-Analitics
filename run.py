#!/usr/bin/env python
"""
Запуск ASO Analytics Service v4.0
"""

import os
import sys
from pathlib import Path

# =====================================================
# 🆕 УСТАНОВКА РАБОЧЕЙ ДИРЕКТОРИИ В КОРЕНЬ ПРОЕКТА
# =====================================================
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Проверка
data_path = PROJECT_ROOT / "data"
clean_pkl = data_path / "clean_data.pkl"

print("=" * 60)
print(f"📂 Working directory: {Path.cwd()}")
print(f"📂 Project root:      {PROJECT_ROOT}")
print(f"📂 Data folder:       {data_path}")
print(f"📂 Data exists:       {data_path.exists()}")
print(f"📂 clean_data.pkl:    {clean_pkl.exists()}")
if clean_pkl.exists():
    size_mb = clean_pkl.stat().st_size / (1024 * 1024)
    print(f"📂 Размер:            {size_mb:.2f} MB")
print("=" * 60)

import uvicorn

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🚀 ASO Analytics Service v4.0                             ║
║                                                              ║
║   API документация: http://localhost:8000/docs              ║
║   Главная страница:  http://localhost:8000                  ║
║   ML страница:        http://localhost:8000/ml              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )