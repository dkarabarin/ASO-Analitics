"""
FastAPI приложение для ASO Analytics Service v4.1
Главный файл с эндпоинтами и маршрутами
Обновлён: абсолютные пути, диагностика, пересборка отчёта с событиями
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# =====================================================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================================================
# КОНФИГУРАЦИЯ (АБСОЛЮТНЫЕ ПУТИ)
# =====================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)

DATA_PATH = PROJECT_ROOT / "data"
PLOTS_PATH = DATA_PATH / "plots"
MODELS_PATH = DATA_PATH / "saved_models"
STATIC_PATH = PROJECT_ROOT / "app" / "static"

DATA_PATH.mkdir(exist_ok=True)
PLOTS_PATH.mkdir(exist_ok=True)
MODELS_PATH.mkdir(exist_ok=True)
STATIC_PATH.mkdir(parents=True, exist_ok=True)

logger.info("=" * 60)
logger.info("📁 ПУТИ ПРОЕКТА:")
logger.info(f"   PROJECT_ROOT: {PROJECT_ROOT}")
logger.info(f"   CWD:          {Path.cwd()}")
logger.info(f"   DATA_PATH:    {DATA_PATH}")
logger.info(f"   PLOTS_PATH:   {PLOTS_PATH}")
logger.info(f"   MODELS_PATH:  {MODELS_PATH}")
logger.info(f"   STATIC_PATH:  {STATIC_PATH}")
logger.info("=" * 60)

# =====================================================
# ИМПОРТ МОДУЛЕЙ ЯДРА
# =====================================================
from app.core.data_loader import DataLoader
from app.core.eda import EDAAnalyzer
from app.core.clustering import ClusterAnalyzer
from app.core.ml_model import MLModeler
from app.core.statistical_tests import StatisticalTester
from app.core.reporting import ReportGenerator
from app.core.motiv_calculator import MotivCalculator
from app.core.events_analyzer import EventsAnalyzer
from app.llm.client import LLMClient

# =====================================================
# ИМПОРТ СХЕМ
# =====================================================
from app.api.schemas import (
    StatusResponse,
    SuccessResponse,
    DataLoadResponse,
    DataStatusResponse,
    TopKeywordsResponse,
    PlotsListResponse,
    PlotsByStepResponse,
    PlotsGroupedResponse,
    PlotItem,
    ClusteringRunResponse,
    ClustersResponse,
    ClusterInfo,
    AnchorWordsResponse,
    AnchorWord,
    MLTrainResponse,
    MLResultsResponse,
    ModelResult,
    FeatureImportanceResponse,
    FeatureImportanceItem,
    PredictRequest,
    PredictResponse,
    TestsResultsResponse,
    ReportGenerateResponse,
    LLMAnalyzeRequest,
    LLMChatRequest,
    LLMResponse,
    LLMStatusResponse,
    MethodologyResponse,
    ExportRequest,
    ExplainPlotRequest
)

# =====================================================
# СОЗДАНИЕ ПРИЛОЖЕНИЯ
# =====================================================
app = FastAPI(
    title="ASO Analytics Service",
    description="Сервис для анализа ASO данных приложения Sound Amplifier (v4.1)",
    version="4.1",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/plots", StaticFiles(directory=str(PLOTS_PATH)), name="plots")
app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")


# =====================================================
# СОСТОЯНИЕ ПРИЛОЖЕНИЯ
# =====================================================
class AppState:
    def __init__(self):
        self.df = None
        self.data_loaded = False
        self.eda_completed = False
        self.eda_results = None
        self.eda_analyzer = None
        self.clustering_completed = False
        self.clusterer = None
        self.ml_completed = False
        self.ml_model = None
        self.ml_results = None
        self.tests_completed = False
        self.tester = None
        self.report_generated = False
        self.current_report = None
        self.motiv_calculator = None
        self.events_analyzer = None
        self.llm_client = LLMClient()

state = AppState()


# =====================================================
# ФУНКЦИЯ ЗАГРУЗКИ ДАННЫХ (С ЗАЩИТОЙ ОТ ОШИБОК PICKLE)
# =====================================================
def load_dataframe() -> Optional[pd.DataFrame]:
    """
    Универсальная загрузка данных: сначала пробуем pickle, потом CSV.
    """
    pkl_path = DATA_PATH / "clean_data.pkl"
    csv_path = DATA_PATH / "clean_data.csv"
    
    # Пробуем pickle
    if pkl_path.exists():
        try:
            df = pd.read_pickle(pkl_path)
            logger.info(f"✅ Данные загружены из pickle: {len(df):,} записей, {len(df.columns)} признаков")
            return df
        except Exception as e:
            logger.warning(f"⚠️ Ошибка загрузки pickle: {e}")
            logger.info("   Пробуем загрузить из CSV...")
    
    # Fallback на CSV
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, parse_dates=['date'])
            logger.info(f"✅ Данные загружены из CSV: {len(df):,} записей, {len(df.columns)} признаков")
            return df
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки CSV: {e}")
    
    logger.error("❌ Не удалось загрузить данные")
    return None


# =====================================================
# ВОССТАНОВЛЕНИЕ EDA ИЗ DF
# =====================================================
def build_eda_results_from_df(df: pd.DataFrame) -> dict:
    """Восстанавливает базовые результаты EDA из DataFrame."""
    try:
        valid_pos = df[df['position'] != -1]['position']
        motiv_pos = df[df['motiv'] > 0]['motiv']
        
        general_stats = {
            'total_records': int(len(df)),
            'unique_keywords': int(df['keyword'].nunique()),
            'unique_dates': int(df['date'].nunique()),
            'years': [int(y) for y in sorted(df['year'].unique().tolist())] if 'year' in df.columns else [],
            'total_features': int(len(df.columns)),
            'period_start': df['date'].min().strftime('%Y-%m-%d'),
            'period_end': df['date'].max().strftime('%Y-%m-%d'),
            'has_events': bool('is_freeze' in df.columns and 'is_ios_update_day' in df.columns),
            'avg_motiv': float(df['motiv'].mean()),
            'avg_position': float(valid_pos.mean()) if len(valid_pos) > 0 else 0.0
        }
        
        insights = {
            'text': f"Всего записей: {len(df):,}, ключей: {df['keyword'].nunique()}",
            'top_keyword': str(df['keyword'].value_counts().index[0]) if len(df) > 0 else 'N/A',
            'top_keyword_count': int(df['keyword'].value_counts().iloc[0]) if len(df) > 0 else 0,
            'recommendations': [
                'Использовать расчёт требуемого мотива',
                'Учитывать фризы и iOS обновления',
                'Использовать прогноз на 1 день'
            ]
        }
        
        distribution = {
            'motiv_positive_count': int(len(motiv_pos)),
            'motiv_positive_pct': float(len(motiv_pos)/len(df)*100) if len(df) > 0 else 0,
            'motiv_mean_all': float(df['motiv'].mean()),
            'motiv_mean_positive': float(motiv_pos.mean()) if len(motiv_pos) > 0 else 0,
            'position_mean': float(valid_pos.mean()) if len(valid_pos) > 0 else 0,
            'position_median': float(valid_pos.median()) if len(valid_pos) > 0 else 0
        }
        
        return {
            'general_stats': general_stats,
            'insights': insights,
            'distribution': distribution,
            '_restored_from_df': True
        }
    except Exception as e:
        logger.warning(f"Ошибка восстановления EDA из df: {e}")
        return {'general_stats': {}, 'insights': {}, 'distribution': {}}


# =====================================================
# ЗАГРУЗКА СУЩЕСТВУЮЩИХ РЕЗУЛЬТАТОВ (С ДИАГНОСТИКОЙ И ПЕРЕСБОРКОЙ ОТЧЁТА)
# =====================================================
def load_existing_results():
    """Загрузка существующих результатов из файлов при старте сервиса."""
    logger.info("=" * 60)
    logger.info("🔍 Проверка существующих результатов...")
    logger.info("=" * 60)
    
    # Диагностика путей
    logger.info("")
    logger.info(f"📂 CWD:        {Path.cwd()}")
    logger.info(f"📂 DATA_PATH:  {DATA_PATH.resolve()}")
    logger.info(f"📂 PLOTS_PATH: {PLOTS_PATH.resolve()}")
    logger.info("")
    logger.info("📋 Проверка файлов в data/:")
    
    expected_files = [
        ("clean_data.pkl", "Данные (Шаг 1)"),
        ("clean_data.csv", "Данные CSV (Шаг 1)"),
        ("motiv_efficiency.csv", "Эффективность мотива (Шаг 3)"),
        ("model_results_1day.csv", "Результаты ML (Шаг 4)"),
        ("feature_importance_1day.csv", "Важность признаков (Шаг 4)"),
        ("hypotheses_summary_1day.csv", "Гипотезы (Шаг 5)"),
        ("rules_summary_1day.csv", "Правила (Шаг 6)"),
        ("final_report_1day.json", "Отчёт (Шаг 7)"),
        ("freeze_periods.csv", "Периоды фризов"),
        ("ios_updates.csv", "iOS обновления"),
        ("best_model_xgboost_1day.pkl", "Модель"),
        ("scaler_1day.pkl", "Scaler"),
        ("le_keyword.pkl", "Кодировщик"),
    ]
    
    for fname, description in expected_files:
        fpath = DATA_PATH / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            logger.info(f"   ✅ {fname:<35s} {description:<30s} ({size_kb:.1f} KB)")
        else:
            logger.info(f"   ❌ {fname:<35s} {description:<30s} НЕ НАЙДЕН")
    
    # Графики
    if PLOTS_PATH.exists():
        plots = list(PLOTS_PATH.glob("*.png"))
        logger.info("")
        logger.info(f"📊 Графиков в plots/: {len(plots)}")
        for p in sorted(plots)[:20]:
            logger.info(f"   ✅ {p.name}")
    else:
        logger.info(f"📊 Папка plots/ не существует")
    
    logger.info("")
    
    loaded_items = []
    warnings_list = []
    
    try:
        # =====================================================
        # 1. ОСНОВНЫЕ ДАННЫЕ
        # =====================================================
        if (DATA_PATH / "clean_data.pkl").exists() or (DATA_PATH / "clean_data.csv").exists():
            state.df = load_dataframe()
            
            if state.df is not None:
                state.data_loaded = True
                n_records = len(state.df)
                n_features = len(state.df.columns)
                
                loaded_items.append(f"✅ Данные: {n_records:,} записей, {n_features} признаков")
                
                if n_features < 100:
                    warnings_list.append(
                        f"❌ МАЛО ПРИЗНАКОВ: {n_features} (ожидается ~121). "
                        f"Перезапустите Шаг 1 в ноутбуке!"
                    )
                
                required_features = [
                    'position_delta_1d', 'is_freeze', 'is_ios_update_day',
                    'is_week_after_ios_update', 'days_after_freeze', 'is_holiday_season'
                ]
                missing = [f for f in required_features if f not in state.df.columns]
                
                if missing:
                    warnings_list.append(f"❌ Отсутствуют признаки: {', '.join(missing)}")
                else:
                    loaded_items.append(f"   ✓ Все ключевые признаки найдены")
                
                if 'cluster' in state.df.columns:
                    n_clusters = state.df['cluster'].nunique()
                    state.clustering_completed = True
                    loaded_items.append(f"   ✓ Кластеры: {n_clusters}")
        else:
            warnings_list.append("❌ clean_data.pkl / clean_data.csv НЕ НАЙДЕНЫ")
        
        # =====================================================
        # 2. EDA
        # =====================================================
        if PLOTS_PATH.exists():
            plots = list(PLOTS_PATH.glob("*.png"))
            if len(plots) > 0:
                state.eda_completed = True
                loaded_items.append(f"✅ EDA: {len(plots)} графиков")
                
                if state.df is not None:
                    state.eda_results = build_eda_results_from_df(state.df)
                    loaded_items.append(f"   ✓ EDA результаты восстановлены из df")
                
                if (PLOTS_PATH / "freeze_and_updates.png").exists():
                    loaded_items.append(f"   ✓ График фризов и iOS")
        
        # =====================================================
        # 3. ML МОДЕЛЬ
        # =====================================================
        model_new = DATA_PATH / "model_results_1day.csv"
        if model_new.exists():
            state.ml_completed = True
            try:
                model_df = pd.read_csv(model_new)
                loaded_items.append(f"✅ ML (прогноз 1 день): {len(model_df)} моделей")
                if len(model_df) > 0:
                    best_idx = model_df['MAE'].idxmin()
                    best = model_df.loc[best_idx]
                    loaded_items.append(f"   ✓ Лучшая: {best['Model']} (MAE={best['MAE']:.4f})")
            except Exception as e:
                warnings_list.append(f"⚠️ Ошибка чтения {model_new.name}: {e}")
        
        # =====================================================
        # 4. ТЕСТЫ
        # =====================================================
        hyp_new = DATA_PATH / "hypotheses_summary_1day.csv"
        if hyp_new.exists():
            state.tests_completed = True
            try:
                hyp_df = pd.read_csv(hyp_new)
                loaded_items.append(f"✅ Гипотезы: {len(hyp_df)} тестов")
                if 'Результат' in hyp_df.columns:
                    confirmed = (hyp_df['Результат'].str.contains('✅', na=False)).sum()
                    loaded_items.append(f"   ✓ Подтверждено: {confirmed}/{len(hyp_df)}")
            except Exception as e:
                warnings_list.append(f"⚠️ Ошибка чтения {hyp_new.name}: {e}")
        
        # =====================================================
        # 5. ОТЧЁТ (ПЕРЕСОБИРАЕМ С АКТУАЛЬНЫМИ СОБЫТИЯМИ)
        # =====================================================
        report_new = DATA_PATH / "final_report_1day.json"
        
        if report_new.exists() and state.df is not None:
            state.report_generated = True
            try:
                # 🆕 Пересобираем отчёт, чтобы подтянуть актуальные события
                logger.info("")
                logger.info("🔄 Пересборка отчёта с актуальными событиями...")
                reporter = ReportGenerator(DATA_PATH)
                state.current_report = reporter.generate_report()
                loaded_items.append(f"✅ Отчёт (1 день) — пересобран с событиями")
            except Exception as e:
                warnings_list.append(f"⚠️ Ошибка пересборки отчёта: {e}")
                # Fallback: загружаем как есть
                try:
                    with open(report_new, 'r', encoding='utf-8') as f:
                        state.current_report = json.load(f)
                    loaded_items.append(f"✅ Отчёт (загружен как есть)")
                except Exception as e2:
                    warnings_list.append(f"⚠️ Ошибка чтения отчёта: {e2}")
        elif report_new.exists():
            state.report_generated = True
            try:
                with open(report_new, 'r', encoding='utf-8') as f:
                    state.current_report = json.load(f)
                loaded_items.append(f"✅ Отчёт (1 день)")
            except Exception as e:
                warnings_list.append(f"⚠️ Ошибка чтения отчёта: {e}")
        
        # =====================================================
        # 6. ЭФФЕКТИВНОСТЬ МОТИВА
        # =====================================================
        eff_path = DATA_PATH / "motiv_efficiency.csv"
        if eff_path.exists():
            try:
                eff_df = pd.read_csv(eff_path)
                loaded_items.append(f"✅ Эффективность мотива: {len(eff_df)} ключей")
            except Exception as e:
                warnings_list.append(f"⚠️ Ошибка чтения motiv_efficiency.csv: {e}")
        
        # =====================================================
        # 7. MOTIV CALCULATOR
        # =====================================================
        if state.df is not None:
            try:
                state.motiv_calculator = MotivCalculator(DATA_PATH, state.df)
                loaded_items.append(f"✅ MotivCalculator инициализирован")
            except Exception as e:
                warnings_list.append(f"⚠️ Ошибка MotivCalculator: {e}")
        
        # =====================================================
        # 8. EVENTS ANALYZER
        # =====================================================
        try:
            state.events_analyzer = EventsAnalyzer(DATA_PATH, state.df)
            loaded_items.append(f"✅ EventsAnalyzer инициализирован")
        except Exception as e:
            warnings_list.append(f"⚠️ Ошибка EventsAnalyzer: {e}")
        
        # =====================================================
        # 9. LLM
        # =====================================================
        if state.llm_client.is_available:
            loaded_items.append(f"✅ LLM доступен ({state.llm_client.model})")
        else:
            warnings_list.append("⚠️ LLM недоступен")
        
        # =====================================================
        # 10. ИТОГИ
        # =====================================================
        logger.info("")
        logger.info("📊 ЗАГРУЖЕННЫЕ РЕЗУЛЬТАТЫ:")
        for item in loaded_items:
            logger.info(f"  {item}")
        
        if warnings_list:
            logger.info("")
            logger.info("⚠️ ПРЕДУПРЕЖДЕНИЯ:")
            for w in warnings_list:
                logger.info(f"  {w}")
        
        completed_count = sum([
            state.data_loaded, state.eda_completed,
            state.ml_completed, state.tests_completed, state.report_generated
        ])
        
        logger.info("")
        logger.info(f"📈 Загружено шагов: {completed_count}/5")
        
        if completed_count == 5:
            logger.info("✅ Все результаты загружены!")
        elif completed_count > 0:
            logger.info("⚠️ Часть результатов отсутствует")
        else:
            logger.info("❌ Результаты не найдены. Проверьте пути выше!")
        
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка при загрузке: {e}")
        import traceback
        traceback.print_exc()


@app.on_event("startup")
async def startup_event():
    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║" + " " * 15 + "🚀 ASO Analytics Service v4.1" + " " * 16 + "║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info("")
    load_existing_results()
    logger.info("")
    logger.info("🌐 Сервис запущен на http://localhost:8000")
    logger.info("📚 API документация: http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("👋 Сервис остановлен")


# =====================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =====================================================
def get_data_summary() -> dict:
    if state.df is None:
        return {}
    valid_pos = state.df[state.df['position'] != -1]
    return {
        'total_records': len(state.df),
        'unique_keywords': state.df['keyword'].nunique(),
        'unique_clusters': state.df['cluster'].nunique() if 'cluster' in state.df.columns else 0,
        'period_start': state.df['date'].min().strftime('%Y-%m-%d'),
        'period_end': state.df['date'].max().strftime('%Y-%m-%d'),
        'avg_motiv': float(state.df['motiv'].mean()),
        'avg_position': float(valid_pos['position'].mean()) if len(valid_pos) > 0 else 0
    }


def clean_float_values(obj):
    if isinstance(obj, dict):
        return {k: clean_float_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_float_values(v) for v in obj]
    elif isinstance(obj, float):
        if pd.isna(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.integer, np.floating)):
        val = float(obj)
        if pd.isna(val) or np.isinf(val):
            return None
        return val
    else:
        return obj


# =====================================================
# ЭНДПОИНТЫ: ЗАГРУЗКА ДАННЫХ
# =====================================================

@app.post("/api/data/load", response_model=DataLoadResponse, tags=["Данные"])
async def load_data():
    """Загрузка данных из Excel файлов"""
    try:
        logger.info("Начало загрузки данных из Excel...")
        loader = DataLoader(DATA_PATH)
        state.df = loader.load_and_preprocess()
        state.data_loaded = True
        
        state.motiv_calculator = MotivCalculator(DATA_PATH, state.df)
        state.events_analyzer = EventsAnalyzer(DATA_PATH, state.df)
        
        state.eda_completed = False
        state.eda_results = None
        state.ml_completed = False
        state.tests_completed = False
        state.report_generated = False
        
        logger.info(f"Данные загружены: {len(state.df)} записей")
        
        return DataLoadResponse(
            status="success",
            message="Данные успешно загружены",
            records=len(state.df),
            keywords=state.df['keyword'].nunique(),
            period_start=state.df['date'].min().strftime('%Y-%m-%d'),
            period_end=state.df['date'].max().strftime('%Y-%m-%d')
        )
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/status", response_model=DataStatusResponse, tags=["Данные"])
async def get_data_status():
    if not state.data_loaded or state.df is None:
        return DataStatusResponse(loaded=False)
    return DataStatusResponse(
        loaded=True,
        records=len(state.df),
        keywords=state.df['keyword'].nunique(),
        clusters=state.df['cluster'].nunique() if 'cluster' in state.df.columns else 0,
        period_start=state.df['date'].min().strftime('%Y-%m-%d'),
        period_end=state.df['date'].max().strftime('%Y-%m-%d')
    )


@app.post("/api/data/reload", response_model=SuccessResponse, tags=["Данные"])
async def reload_data():
    """Перезагрузка данных из clean_data.pkl БЕЗ парсинга Excel"""
    try:
        df = load_dataframe()
        if df is None:
            raise HTTPException(status_code=404, detail="Файл данных не найден")
        
        state.df = df
        state.data_loaded = True
        state.motiv_calculator = MotivCalculator(DATA_PATH, state.df)
        state.events_analyzer = EventsAnalyzer(DATA_PATH, state.df)
        
        n_features = len(state.df.columns)
        message = f"Данные перезагружены: {len(state.df):,} записей, {n_features} признаков"
        
        return SuccessResponse(
            status="success",
            message=message,
            data={"records": len(state.df), "features": n_features}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data/reset", response_model=SuccessResponse, tags=["Данные"])
async def reset_data():
    state.df = None
    state.data_loaded = False
    state.eda_completed = False
    state.eda_results = None
    state.clustering_completed = False
    state.ml_completed = False
    state.tests_completed = False
    state.report_generated = False
    state.current_report = None
    state.motiv_calculator = None
    state.events_analyzer = None
    return SuccessResponse(status="success", message="Состояние сброшено", data={"reset": True})


# =====================================================
# ЭНДПОИНТЫ: EDA
# =====================================================

@app.post("/api/eda/run", response_model=SuccessResponse, tags=["EDA"])
async def run_eda():
    if not state.data_loaded:
        raise HTTPException(status_code=400, detail="Данные не загружены")
    try:
        logger.info("Начало EDA...")
        eda = EDAAnalyzer(DATA_PATH, state.df)
        results = eda.run_eda()
        state.eda_completed = True
        state.eda_results = results
        state.eda_analyzer = eda
        
        plots_count = len(list(PLOTS_PATH.glob("*.png")))
        logger.info(f"EDA завершён. Графиков: {plots_count}")
        
        return SuccessResponse(
            status="success",
            message="EDA завершён успешно",
            data={
                "plots_count": plots_count,
                "results": {
                    "total_records": results.get('general_stats', {}).get('total_records', 0),
                    "unique_keywords": results.get('general_stats', {}).get('unique_keywords', 0)
                }
            }
        )
    except Exception as e:
        logger.error(f"Ошибка EDA: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка EDA: {str(e)}")


@app.get("/api/eda/summary", response_model=Dict[str, Any], tags=["EDA"])
async def get_eda_summary():
    if not state.eda_completed:
        raise HTTPException(status_code=400, detail="EDA не выполнен")
    
    if state.eda_results is not None:
        return clean_float_values(state.eda_results.get('general_stats', {}))
    
    if state.df is not None:
        restored = build_eda_results_from_df(state.df)
        state.eda_results = restored
        logger.info("EDA результаты восстановлены из df")
        return clean_float_values(restored.get('general_stats', {}))
    
    return {}


@app.get("/api/eda/top_keywords", response_model=TopKeywordsResponse, tags=["EDA"])
async def get_top_keywords(limit: int = Query(20, ge=1, le=100)):
    if not state.data_loaded:
        raise HTTPException(status_code=400, detail="Данные не загружены")
    top = state.df['keyword'].value_counts().head(limit)
    return TopKeywordsResponse(top_keywords=top.to_dict())


@app.get("/api/eda/plots", response_model=PlotsListResponse, tags=["EDA"])
async def get_plots_list():
    plots = []
    for f in PLOTS_PATH.glob("*.png"):
        plots.append(PlotItem(
            name=f.name, url=f"/plots/{f.name}",
            size=f.stat().st_size, modified=f.stat().st_mtime
        ))
    return PlotsListResponse(plots=plots)


@app.get("/api/eda/insights", response_model=Dict[str, Any], tags=["EDA"])
async def get_eda_insights():
    if not state.eda_completed:
        raise HTTPException(status_code=400, detail="EDA не выполнен")
    
    if state.eda_results is not None:
        return clean_float_values(state.eda_results.get('insights', {}))
    
    if state.df is not None:
        restored = build_eda_results_from_df(state.df)
        state.eda_results = restored
        return clean_float_values(restored.get('insights', {}))
    
    return {}


# =====================================================
# ГРАФИКИ ПО ШАГАМ
# =====================================================

@app.get("/api/plots/by_step/{step}", response_model=PlotsByStepResponse, tags=["Графики"])
async def get_plots_by_step(step: int):
    step_names = {
        1: "Загрузка данных", 2: "EDA", 3: "Расчёт мотива",
        4: "ML", 5: "Статистические тесты", 6: "Проверка правил"
    }
    step_patterns = {
        1: ['clean_data'],
        2: ['eda_overview', 'correlation_heatmap', 'timeseries_top5',
            'monthly_analysis', 'motiv_distribution', 'position_distribution',
            'top20_keywords', 'motiv_vs_position', 'freeze_and_updates'],
        3: ['motiv_efficiency'],
        4: ['model_comparison_1day', 'feature_importance_1day',
            'predictions_vs_actual_1day', 'events_importance'],
        5: ['hypotheses_1day'],
        6: ['rules_1day']
    }
    patterns = step_patterns.get(step, [])
    plots = []
    for f in PLOTS_PATH.glob("*.png"):
        for pattern in patterns:
            if pattern in f.name:
                plots.append(PlotItem(
                    name=f.name, url=f"/plots/{f.name}",
                    size=f.stat().st_size, modified=f.stat().st_mtime,
                    step=step, step_name=step_names.get(step, f"Шаг {step}")
                ))
                break
    return PlotsByStepResponse(
        step=step, step_name=step_names.get(step, f"Шаг {step}"),
        plots=plots, total=len(plots)
    )


@app.get("/api/plots/all_grouped", response_model=PlotsGroupedResponse, tags=["Графики"])
async def get_all_plots_grouped():
    step_names = {
        1: "Загрузка данных", 2: "EDA", 3: "Расчёт мотива",
        4: "ML", 5: "Статистические тесты", 6: "Проверка правил"
    }
    step_patterns = {
        1: ['clean_data'],
        2: ['eda_overview', 'correlation_heatmap', 'timeseries_top5',
            'monthly_analysis', 'motiv_distribution', 'position_distribution',
            'top20_keywords', 'motiv_vs_position', 'freeze_and_updates'],
        3: ['motiv_efficiency'],
        4: ['model_comparison_1day', 'feature_importance_1day',
            'predictions_vs_actual_1day', 'events_importance'],
        5: ['hypotheses_1day'],
        6: ['rules_1day']
    }
    grouped = {}
    for step in range(1, 7):
        grouped[step] = {"name": step_names.get(step, f"Шаг {step}"), "plots": []}
    grouped[0] = {"name": "Другие", "plots": []}
    
    for f in PLOTS_PATH.glob("*.png"):
        name = f.name
        assigned = False
        for step, patterns in step_patterns.items():
            for pattern in patterns:
                if pattern in name:
                    grouped[step]["plots"].append(PlotItem(
                        name=name, url=f"/plots/{name}",
                        size=f.stat().st_size, modified=f.stat().st_mtime,
                        step=step, step_name=step_names.get(step, f"Шаг {step}")
                    ))
                    assigned = True
                    break
            if assigned:
                break
        if not assigned:
            grouped[0]["plots"].append(PlotItem(
                name=name, url=f"/plots/{name}",
                size=f.stat().st_size, modified=f.stat().st_mtime
            ))
    return PlotsGroupedResponse(
        plots_by_step=grouped,
        total=len(list(PLOTS_PATH.glob("*.png")))
    )


@app.get("/api/plots/step/{step}/count", tags=["Графики"])
async def get_plots_count_by_step(step: int):
    result = await get_plots_by_step(step)
    return {"step": step, "count": result.total}


# =====================================================
# РАСЧЁТ МОТИВА
# =====================================================

@app.get("/api/motiv/keywords", tags=["Расчёт мотива"])
async def get_keywords_list():
    if not state.data_loaded:
        raise HTTPException(status_code=400, detail="Данные не загружены")
    if state.motiv_calculator is None:
        state.motiv_calculator = MotivCalculator(DATA_PATH, state.df)
    keywords = state.motiv_calculator.get_keywords_list()
    return {"keywords": keywords, "total": len(keywords)}


@app.get("/api/motiv/keyword/{keyword:path}", tags=["Расчёт мотива"])
async def get_keyword_info(keyword: str):
    if not state.data_loaded:
        raise HTTPException(status_code=400, detail="Данные не загружены")
    if state.motiv_calculator is None:
        state.motiv_calculator = MotivCalculator(DATA_PATH, state.df)
    info = state.motiv_calculator.get_keyword_info(keyword)
    if 'error' in info:
        raise HTTPException(status_code=404, detail=info['error'])
    return clean_float_values(info)


@app.post("/api/motiv/calculate", tags=["Расчёт мотива"])
async def calculate_required_motiv(request: Dict[str, Any]):
    if not state.data_loaded:
        raise HTTPException(status_code=400, detail="Данные не загружены")
    keyword = request.get('keyword')
    target_position = request.get('target_position', 10)
    if not keyword:
        raise HTTPException(status_code=400, detail="Не указано ключевое слово")
    if not isinstance(target_position, int) or target_position < 1 or target_position > 250:
        raise HTTPException(status_code=400, detail="Целевая позиция от 1 до 250")
    if state.motiv_calculator is None:
        state.motiv_calculator = MotivCalculator(DATA_PATH, state.df)
    result = state.motiv_calculator.calculate_required_motiv(keyword, target_position)
    if 'error' in result:
        raise HTTPException(status_code=404, detail=result['error'])
    return clean_float_values(result)


@app.post("/api/motiv/batch_calculate", tags=["Расчёт мотива"])
async def batch_calculate_motiv(request: Dict[str, Any]):
    if not state.data_loaded:
        raise HTTPException(status_code=400, detail="Данные не загружены")
    items = request.get('items', [])
    if not items:
        raise HTTPException(status_code=400, detail="Не указаны элементы")
    if state.motiv_calculator is None:
        state.motiv_calculator = MotivCalculator(DATA_PATH, state.df)
    results = []
    for item in items:
        keyword = item.get('keyword')
        target = item.get('target_position', 10)
        if keyword:
            result = state.motiv_calculator.calculate_required_motiv(keyword, target)
            results.append(clean_float_values(result))
    return {"results": results, "total": len(results)}


@app.get("/api/motiv/efficiency", tags=["Расчёт мотива"])
async def get_motiv_efficiency():
    if not state.data_loaded:
        raise HTTPException(status_code=400, detail="Данные не загружены")
    if state.motiv_calculator is None:
        state.motiv_calculator = MotivCalculator(DATA_PATH, state.df)
    if state.motiv_calculator.motiv_efficiency is None or len(state.motiv_calculator.motiv_efficiency) == 0:
        eff_path = DATA_PATH / "motiv_efficiency.csv"
        if eff_path.exists():
            try:
                state.motiv_calculator.motiv_efficiency = pd.read_csv(eff_path)
            except:
                state.motiv_calculator.calculate_efficiency()
        else:
            state.motiv_calculator.calculate_efficiency()
    if state.motiv_calculator.motiv_efficiency is not None and len(state.motiv_calculator.motiv_efficiency) > 0:
        df = state.motiv_calculator.motiv_efficiency.sort_values('efficiency', ascending=False)
        return {"efficiency": clean_float_values(df.to_dict('records')), "total": len(df)}
    return {"efficiency": [], "total": 0}


@app.post("/api/motiv/recalculate", tags=["Расчёт мотива"])
async def recalculate_motiv_efficiency():
    if not state.data_loaded:
        raise HTTPException(status_code=400, detail="Данные не загружены")
    try:
        state.motiv_calculator = MotivCalculator(DATA_PATH, state.df)
        df = state.motiv_calculator.calculate_efficiency()
        return SuccessResponse(
            status="success",
            message=f"Пересчитано для {len(df)} ключей",
            data={"total": len(df)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# СОБЫТИЯ
# =====================================================

@app.get("/api/events/freezes", tags=["События"])
async def get_freeze_periods():
    analyzer = EventsAnalyzer(DATA_PATH, state.df if state.data_loaded else None)
    return {"freezes": analyzer.get_freeze_periods()}


@app.get("/api/events/ios_updates", tags=["События"])
async def get_ios_updates():
    analyzer = EventsAnalyzer(DATA_PATH, state.df if state.data_loaded else None)
    return {"updates": analyzer.get_ios_updates()}


@app.get("/api/events/timeline", tags=["События"])
async def get_events_timeline():
    analyzer = EventsAnalyzer(DATA_PATH, state.df if state.data_loaded else None)
    return {"events": analyzer.get_events_timeline()}


@app.get("/api/events/next_freeze", tags=["События"])
async def get_next_freeze():
    analyzer = EventsAnalyzer(DATA_PATH, state.df if state.data_loaded else None)
    result = analyzer.get_next_freeze()
    if result:
        return result
    return {"message": "Нет запланированных фризов"}


@app.post("/api/events/analyze_freeze", tags=["События"])
async def analyze_freeze(request: Dict[str, Any]):
    start_date = request.get('start_date')
    end_date = request.get('end_date')
    if not start_date or not end_date:
        raise HTTPException(status_code=400, detail="Не указаны даты")
    if not state.data_loaded:
        raise HTTPException(status_code=400, detail="Данные не загружены")
    analyzer = EventsAnalyzer(DATA_PATH, state.df)
    result = analyzer.analyze_freeze_period(start_date, end_date)
    return clean_float_values(result)


@app.post("/api/events/analyze_ios_update", tags=["События"])
async def analyze_ios_update(request: Dict[str, Any]):
    date = request.get('date')
    version = request.get('version', '')
    if not date:
        raise HTTPException(status_code=400, detail="Не указана дата")
    if not state.data_loaded:
        raise HTTPException(status_code=400, detail="Данные не загружены")
    analyzer = EventsAnalyzer(DATA_PATH, state.df)
    result = analyzer.analyze_ios_update(date, version)
    return clean_float_values(result)


@app.post("/api/events/predict_freeze_impact", tags=["События"])
async def predict_freeze_impact(request: Dict[str, Any]):
    keyword = request.get('keyword')
    target_position = request.get('target_position', 10)
    if not keyword:
        raise HTTPException(status_code=400, detail="Не указано ключевое слово")
    if not state.data_loaded:
        raise HTTPException(status_code=400, detail="Данные не загружены")
    analyzer = EventsAnalyzer(DATA_PATH, state.df)
    result = analyzer.predict_freeze_impact(keyword, target_position)
    return clean_float_values(result)


# =====================================================
# ML
# =====================================================

@app.post("/api/ml/train", response_model=MLTrainResponse, tags=["ML"])
async def train_models():
    if not state.data_loaded:
        raise HTTPException(status_code=400, detail="Данные не загружены")
    try:
        logger.info("Начало обучения ML моделей...")
        ml = MLModeler(DATA_PATH, state.df)
        results = ml.run_ml_pipeline()
        state.ml_completed = True
        state.ml_model = ml
        state.ml_results = results
        logger.info(f"ML обучение завершено. Лучшая модель: {ml.best_model_name}")
        return MLTrainResponse(
            status="success",
            message="Обучение завершено (прогноз на 1 день)",
            best_model=ml.best_model_name
        )
    except Exception as e:
        logger.error(f"Ошибка ML: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/results", response_model=MLResultsResponse, tags=["ML"])
async def get_ml_results():
    if not state.ml_completed:
        raise HTTPException(status_code=400, detail="ML не обучены")
    results_path = DATA_PATH / "model_results_1day.csv"
    if not results_path.exists():
        results_path = DATA_PATH / "model_results_final.csv"
    if results_path.exists():
        df = pd.read_csv(results_path)
        results = [
            ModelResult(
                Model=row['Model'],
                MAE=float(row['MAE']) if not pd.isna(row['MAE']) else 0.0,
                RMSE=float(row['RMSE']) if not pd.isna(row['RMSE']) else 0.0,
                R2=float(row['R2']) if not pd.isna(row['R2']) else 0.0
            )
            for _, row in df.iterrows()
        ]
        return MLResultsResponse(results=results)
    return MLResultsResponse(results=[])


@app.get("/api/ml/feature_importance", response_model=FeatureImportanceResponse, tags=["ML"])
async def get_feature_importance():
    if not state.ml_completed:
        raise HTTPException(status_code=400, detail="ML не обучены")
    imp_path = DATA_PATH / "feature_importance_1day.csv"
    if not imp_path.exists():
        imp_path = DATA_PATH / "feature_importance_final.csv"
    if imp_path.exists():
        df = pd.read_csv(imp_path)
        features = [
            FeatureImportanceItem(
                feature=row['feature'],
                importance=float(row['importance']) if not pd.isna(row['importance']) else 0.0,
                importance_pct=float(row['importance_pct']) if not pd.isna(row['importance_pct']) else 0.0,
                group=row['group'] if 'group' in row else 'Другие'
            )
            for _, row in df.head(20).iterrows()
        ]
        return FeatureImportanceResponse(feature_importance=features)
    return FeatureImportanceResponse(feature_importance=[])


@app.post("/api/ml/predict", response_model=PredictResponse, tags=["ML"])
async def predict(request: PredictRequest):
    if not state.ml_completed:
        ml = MLModeler(DATA_PATH, state.df)
        if ml.load_model():
            state.ml_model = ml
            state.ml_completed = True
        else:
            raise HTTPException(status_code=400, detail="Модель не обучена")
    try:
        data = request.dict()
        if data.get('keyword') and state.ml_model.le_keyword is not None:
            try:
                data['keyword_encoded'] = state.ml_model.le_keyword.transform([data['keyword']])[0]
            except:
                data['keyword_encoded'] = -1
        if data.get('cluster') and state.ml_model.le_cluster is not None:
            try:
                data['cluster_encoded'] = state.ml_model.le_cluster.transform([data['cluster']])[0]
            except:
                data['cluster_encoded'] = -1
        X_new = pd.DataFrame([data])
        for col in state.ml_model.feature_cols:
            if col not in X_new.columns:
                X_new[col] = 0
        prediction = state.ml_model.predict(X_new[X_new.columns.intersection(state.ml_model.feature_cols)])
        pred_val = float(prediction[0]) if not pd.isna(prediction[0]) else 0.0
        if pred_val < 0:
            interpretation = f"✅ Ожидается улучшение на {abs(pred_val):.2f} позиций"
        elif pred_val > 0:
            interpretation = f"⚠️ Ожидается ухудшение на {pred_val:.2f} позиций"
        else:
            interpretation = "➖ Без изменений"
        return PredictResponse(
            prediction=pred_val, prediction_original=pred_val,
            interpretation=interpretation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/load_model", response_model=SuccessResponse, tags=["ML"])
async def load_saved_model():
    ml = MLModeler(DATA_PATH, state.df)
    if ml.load_model():
        state.ml_model = ml
        state.ml_completed = True
        return SuccessResponse(status="success", message="Модель загружена")
    raise HTTPException(status_code=404, detail="Модель не найдена")


# =====================================================
# ТЕСТЫ
# =====================================================

@app.post("/api/tests/run", response_model=SuccessResponse, tags=["Тесты"])
async def run_tests():
    if not state.data_loaded:
        raise HTTPException(status_code=400, detail="Данные не загружены")
    try:
        logger.info("Начало статистических тестов...")
        tester = StatisticalTester(DATA_PATH, state.df)
        results = tester.run_tests()
        state.tests_completed = True
        state.tester = tester
        logger.info("Тесты завершены")
        return SuccessResponse(
            status="success",
            message="Тесты завершены (H1-H8)",
            data={"clusters_tested": len(tester.clusters) if hasattr(tester, 'clusters') else 0}
        )
    except Exception as e:
        logger.error(f"Ошибка тестов: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tests/results", response_model=TestsResultsResponse, tags=["Тесты"])
async def get_tests_results():
    if not state.tests_completed:
        raise HTTPException(status_code=400, detail="Тесты не выполнены")
    hyp_path = DATA_PATH / "hypotheses_summary_1day.csv"
    if not hyp_path.exists():
        hyp_path = DATA_PATH / "hypotheses_by_cluster_summary.csv"
    if hyp_path.exists():
        df = pd.read_csv(hyp_path)
        records = []
        for _, row in df.iterrows():
            record = {}
            for col, val in row.items():
                if isinstance(val, float):
                    record[col] = None if (pd.isna(val) or np.isinf(val)) else val
                else:
                    record[col] = val
            records.append(record)
        return TestsResultsResponse(hypotheses=records)
    return TestsResultsResponse(hypotheses=[])


# =====================================================
# ОТЧЁТ
# =====================================================

@app.post("/api/report/generate", response_model=ReportGenerateResponse, tags=["Отчёт"])
async def generate_report():
    try:
        logger.info("Начало генерации отчёта...")
        reporter = ReportGenerator(DATA_PATH)
        report = reporter.generate_report()
        state.report_generated = True
        state.current_report = report
        logger.info("Отчёт сгенерирован")
        return ReportGenerateResponse(
            status="success",
            message="Отчёт сгенерирован",
            report_path=str(DATA_PATH / "final_report_1day.json")
        )
    except Exception as e:
        logger.error(f"Ошибка отчёта: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/report", response_model=Dict[str, Any], tags=["Отчёт"])
async def get_report():
    if state.current_report is not None:
        return clean_float_values(state.current_report)
    
    report_path = DATA_PATH / "final_report_1day.json"
    if not report_path.exists():
        report_path = DATA_PATH / "final_report.json"
    
    if report_path.exists():
        with open(report_path, 'r', encoding='utf-8') as f:
            state.current_report = json.load(f)
            state.report_generated = True
            return clean_float_values(state.current_report)
    
    raise HTTPException(status_code=404, detail="Отчёт не найден")


@app.get("/api/report/summary", response_model=Dict[str, Any], tags=["Отчёт"])
async def get_report_summary():
    summary_path = DATA_PATH / "report_summary_1day.csv"
    if not summary_path.exists():
        summary_path = DATA_PATH / "report_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        return {"summary": clean_float_values(df.to_dict('records'))}
    raise HTTPException(status_code=404, detail="Краткий отчёт не найден")


# =====================================================
# LLM
# =====================================================

@app.post("/api/llm/analyze", response_model=LLMResponse, tags=["LLM"])
async def llm_analyze(request: LLMAnalyzeRequest):
    if not state.llm_client.is_available:
        raise HTTPException(status_code=503, detail="LLM недоступен")
    data_summary = get_data_summary()
    response = state.llm_client.analyze_data(data_summary)
    return LLMResponse(response=response)


@app.post("/api/llm/hypotheses", response_model=LLMResponse, tags=["LLM"])
async def llm_hypotheses():
    if not state.llm_client.is_available:
        raise HTTPException(status_code=503, detail="LLM недоступен")
    data_summary = get_data_summary()
    response = state.llm_client.generate_hypotheses(data_summary)
    return LLMResponse(response=response)


@app.post("/api/llm/strategy", response_model=LLMResponse, tags=["LLM"])
async def llm_strategy():
    if not state.llm_client.is_available:
        raise HTTPException(status_code=503, detail="LLM недоступен")
    if not state.clustering_completed:
        raise HTTPException(status_code=400, detail="Сначала кластеризация")
    cluster_info = state.df.groupby('cluster').agg({
        'keyword': 'nunique', 'motiv': 'mean', 'position': 'mean'
    }).round(2).to_dict()
    response = state.llm_client.generate_strategy(cluster_info)
    return LLMResponse(response=response)


@app.post("/api/llm/explain_clusters", response_model=LLMResponse, tags=["LLM"])
async def llm_explain_clusters():
    if not state.llm_client.is_available:
        raise HTTPException(status_code=503, detail="LLM недоступен")
    if not state.clustering_completed:
        raise HTTPException(status_code=400, detail="Сначала кластеризация")
    cluster_stats = state.df.groupby('cluster').agg({
        'keyword': 'nunique',
        'motiv': ['mean', 'sum'],
        'position': ['mean', 'min', 'max']
    }).round(2)
    cluster_stats.columns = ['keywords', 'motiv_mean', 'motiv_sum',
                              'position_mean', 'position_min', 'position_max']
    response = state.llm_client.explain_clusters(cluster_stats)
    return LLMResponse(response=response)


@app.post("/api/llm/explain_plot", response_model=LLMResponse, tags=["LLM"])
async def llm_explain_plot(request: ExplainPlotRequest):
    if not state.llm_client.is_available:
        raise HTTPException(status_code=503, detail="LLM недоступен")
    response = state.llm_client.explain_plot(
        request.plot_name, request.description, request.context
    )
    return LLMResponse(response=response)


@app.post("/api/llm/recommendations", response_model=LLMResponse, tags=["LLM"])
async def llm_recommendations():
    if not state.llm_client.is_available:
        raise HTTPException(status_code=503, detail="LLM недоступен")
    data_summary = get_data_summary()
    response = state.llm_client.get_recommendations(data_summary)
    return LLMResponse(response=response)


@app.post("/api/llm/model_explanation", response_model=LLMResponse, tags=["LLM"])
async def llm_model_explanation():
    if not state.llm_client.is_available:
        raise HTTPException(status_code=503, detail="LLM недоступен")
    model_results_path = DATA_PATH / "model_results_1day.csv"
    if not model_results_path.exists():
        model_results_path = DATA_PATH / "model_results_final.csv"
    model_results = {}
    if model_results_path.exists():
        df = pd.read_csv(model_results_path)
        model_results = clean_float_values(df.to_dict('records'))
    importance_path = DATA_PATH / "feature_importance_1day.csv"
    if not importance_path.exists():
        importance_path = DATA_PATH / "feature_importance_final.csv"
    importance = []
    if importance_path.exists():
        df = pd.read_csv(importance_path)
        importance = clean_float_values(df.head(10).to_dict('records'))
    data = {
        "model_results": model_results,
        "feature_importance": importance,
        "best_model": model_results[0] if model_results else None
    }
    response = state.llm_client.get_model_explanation(data)
    return LLMResponse(response=response)


@app.post("/api/llm/check_keyword_hypothesis", response_model=LLMResponse, tags=["LLM"])
async def llm_check_keyword_hypothesis(request: Dict[str, Any]):
    if not state.llm_client.is_available:
        raise HTTPException(status_code=503, detail="LLM недоступен")
    keyword = request.get('keyword')
    target_position = request.get('target_position')
    if not keyword:
        raise HTTPException(status_code=400, detail="Не указано ключевое слово")
    if state.motiv_calculator is None:
        state.motiv_calculator = MotivCalculator(DATA_PATH, state.df)
    keyword_data = state.motiv_calculator.get_keyword_info(keyword)
    if 'error' in keyword_data:
        raise HTTPException(status_code=404, detail=keyword_data['error'])
    response = state.llm_client.check_keyword_hypothesis(keyword, keyword_data, target_position)
    return LLMResponse(response=response)


@app.post("/api/llm/generate_hypotheses_from_data", response_model=LLMResponse, tags=["LLM"])
async def llm_generate_hypotheses_from_data():
    if not state.llm_client.is_available:
        raise HTTPException(status_code=503, detail="LLM недоступен")
    data_summary = get_data_summary()
    hypotheses_results = ""
    hyp_path = DATA_PATH / "hypotheses_summary_1day.csv"
    if not hyp_path.exists():
        hyp_path = DATA_PATH / "hypotheses_by_cluster_summary.csv"
    if hyp_path.exists():
        df = pd.read_csv(hyp_path)
        hypotheses_results = df.to_string()[:3000]
    clusters = ""
    if state.clustering_completed and state.df is not None:
        cluster_stats = state.df.groupby('cluster').agg({
            'keyword': 'nunique', 'motiv': 'mean', 'position': 'mean'
        }).round(2)
        clusters = cluster_stats.to_string()
    data_context = {
        'total_keywords': data_summary.get('unique_keywords', 0),
        'total_records': data_summary.get('total_records', 0),
        'period': f"{data_summary.get('period_start', '')} - {data_summary.get('period_end', '')}",
        'avg_position': round(data_summary.get('avg_position', 0), 2),
        'avg_motiv': round(data_summary.get('avg_motiv', 0), 2),
        'clusters': clusters,
        'hypotheses_results': hypotheses_results
    }
    response = state.llm_client.generate_new_hypotheses(data_context)
    return LLMResponse(response=response)


@app.post("/api/llm/analyze_freeze", response_model=LLMResponse, tags=["LLM"])
async def llm_analyze_freeze(request: Dict[str, Any]):
    if not state.llm_client.is_available:
        raise HTTPException(status_code=503, detail="LLM недоступен")
    start_date = request.get('start_date')
    end_date = request.get('end_date')
    if not start_date or not end_date:
        raise HTTPException(status_code=400, detail="Не указаны даты")
    if state.events_analyzer is None:
        state.events_analyzer = EventsAnalyzer(DATA_PATH, state.df)
    freeze_data = state.events_analyzer.analyze_freeze_period(start_date, end_date)
    if hasattr(state.llm_client, 'analyze_freeze_impact'):
        response = state.llm_client.analyze_freeze_impact(freeze_data)
    else:
        response = f"Данные фриза:\n{json.dumps(freeze_data, indent=2, ensure_ascii=False, default=str)}"
    return LLMResponse(response=response)


@app.post("/api/llm/analyze_ios_update", response_model=LLMResponse, tags=["LLM"])
async def llm_analyze_ios_update(request: Dict[str, Any]):
    if not state.llm_client.is_available:
        raise HTTPException(status_code=503, detail="LLM недоступен")
    date = request.get('date')
    version = request.get('version', '')
    if not date:
        raise HTTPException(status_code=400, detail="Не указана дата")
    if state.events_analyzer is None:
        state.events_analyzer = EventsAnalyzer(DATA_PATH, state.df)
    update_data = state.events_analyzer.analyze_ios_update(date, version)
    if hasattr(state.llm_client, 'analyze_ios_update_impact'):
        response = state.llm_client.analyze_ios_update_impact(update_data)
    else:
        response = f"Данные обновления:\n{json.dumps(update_data, indent=2, ensure_ascii=False, default=str)}"
    return LLMResponse(response=response)


@app.post("/api/llm/chat", response_model=LLMResponse, tags=["LLM"])
async def llm_chat(request: LLMChatRequest):
    if not state.llm_client.is_available:
        raise HTTPException(status_code=503, detail="LLM недоступен")
    context = request.context
    if not context and state.data_loaded:
        data_summary = get_data_summary()
        context = f"Данные ASO: {data_summary['total_records']} записей, {data_summary['unique_keywords']} ключей"
    response = state.llm_client.chat(request.message, context)
    return LLMResponse(response=response)


@app.get("/api/llm/status", response_model=LLMStatusResponse, tags=["LLM"])
async def llm_status():
    status = state.llm_client.get_status()
    return LLMStatusResponse(
        is_available=status['is_available'],
        model=status['model'],
        url=status['url']
    )


@app.get("/api/llm/methodology", response_model=MethodologyResponse, tags=["LLM"])
async def get_methodology():
    return MethodologyResponse(
        methodology=state.llm_client.ASO_METHODOLOGY,
        version="2.0",
        rules_count=14,
        clusters=[
            "top10_high", "top10_medium", "top10_low", "top10_zero",
            "top50_high", "top50_medium", "top50_low", "top50_zero",
            "above50_high", "above50_medium", "above50_low", "above50_zero"
        ],
        scenarios=[
            {"name": "sharp_growth", "label": "Резкий рост", "threshold": "+30%"},
            {"name": "gentle_growth", "label": "Плавный рост", "threshold": "+10-30%"},
            {"name": "stable", "label": "Стабильный", "threshold": "0%"},
            {"name": "gentle_decline", "label": "Плавный спад", "threshold": "-10-30%"},
            {"name": "sharp_decline", "label": "Резкий спад", "threshold": "-30%"}
        ]
    )


# =====================================================
# КЛАСТЕРИЗАЦИЯ (устаревшая)
# =====================================================

@app.post("/api/clustering/run", response_model=ClusteringRunResponse, tags=["Кластеризация"])
async def run_clustering():
    if not state.data_loaded:
        raise HTTPException(status_code=400, detail="Данные не загружены")
    try:
        clusterer = ClusterAnalyzer(DATA_PATH, state.df)
        state.df = clusterer.run_clustering()
        state.clustering_completed = True
        state.clusterer = clusterer
        return ClusteringRunResponse(
            status="success",
            message="Кластеризация завершена",
            clusters=state.df['cluster'].nunique()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/clustering/clusters", response_model=ClustersResponse, tags=["Кластеризация"])
async def get_clusters():
    if not state.clustering_completed:
        raise HTTPException(status_code=400, detail="Кластеризация не выполнена")
    clusters = {}
    for cluster, group in state.df.groupby('cluster'):
        valid_pos = group[group['position'] != -1]
        clusters[cluster] = ClusterInfo(
            keywords=group['keyword'].nunique(),
            motiv_mean=float(group['motiv'].mean()) if not pd.isna(group['motiv'].mean()) else 0.0,
            position_mean=float(valid_pos['position'].mean()) if len(valid_pos) > 0 else 0.0,
            motiv_sum=float(group['motiv'].sum()) if not pd.isna(group['motiv'].sum()) else 0.0,
            position_min=float(valid_pos['position'].min()) if len(valid_pos) > 0 else 0.0,
            position_max=float(valid_pos['position'].max()) if len(valid_pos) > 0 else 0.0
        )
    return ClustersResponse(clusters=clusters)


@app.get("/api/clustering/anchor_words", response_model=AnchorWordsResponse, tags=["Кластеризация"])
async def get_anchor_words():
    if not state.clustering_completed:
        raise HTTPException(status_code=400, detail="Кластеризация не выполнена")
    anchor_path = DATA_PATH / "anchor_words.csv"
    if anchor_path.exists():
        df = pd.read_csv(anchor_path)
        anchor_words = [
            AnchorWord(
                cluster=row['cluster'],
                anchor_word=row['anchor_word'],
                motiv_mean=float(row['motiv_mean']) if not pd.isna(row['motiv_mean']) else 0.0,
                motiv_sum=float(row['motiv_sum']) if not pd.isna(row['motiv_sum']) else 0.0,
                position_mean=float(row['position_mean']) if not pd.isna(row['position_mean']) else 0.0,
                anchor_score=float(row['anchor_score']) if not pd.isna(row['anchor_score']) else 0.0
            )
            for _, row in df.iterrows()
        ]
        return AnchorWordsResponse(anchor_words=anchor_words)
    return AnchorWordsResponse(anchor_words=[])


# =====================================================
# ОБЩИЕ
# =====================================================

@app.get("/", response_class=HTMLResponse, tags=["Общие"])
async def root():
    html_path = STATIC_PATH / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding='utf-8'))
    return HTMLResponse("<h1>ASO Analytics Service v4.1</h1><p><a href='/docs'>API Docs</a></p>")


@app.get("/ml", response_class=HTMLResponse, tags=["Общие"])
async def ml_page():
    html_path = STATIC_PATH / "ml.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding='utf-8'))
    return HTMLResponse("<h1>ML Analytics</h1><p><a href='/'>← На главную</a></p>")


@app.get("/plots/{filename}", response_class=FileResponse, tags=["Общие"])
async def get_plot(filename: str):
    plot_path = PLOTS_PATH / filename
    if plot_path.exists():
        return FileResponse(plot_path, media_type="image/png", filename=filename)
    raise HTTPException(status_code=404, detail="График не найден")


@app.get("/api/status", response_model=StatusResponse, tags=["Общие"])
async def get_system_status():
    return StatusResponse(
        data_loaded=state.data_loaded,
        eda_completed=state.eda_completed,
        clustering_completed=state.clustering_completed,
        ml_completed=state.ml_completed,
        tests_completed=state.tests_completed,
        report_generated=state.report_generated,
        llm_available=state.llm_client.is_available,
        records=len(state.df) if state.df is not None else 0,
        keywords=state.df['keyword'].nunique() if state.df is not None else 0
    )


@app.get("/api/health", response_model=Dict[str, str], tags=["Общие"])
async def health_check():
    return {"status": "healthy", "service": "ASO Analytics Service", "version": "4.1"}


@app.get("/api/info", tags=["Общие"])
async def get_info():
    has_new_features = False
    if state.df is not None:
        new_features = ['is_freeze', 'is_ios_update_day', 'position_delta_1d']
        has_new_features = all(f in state.df.columns for f in new_features)
    
    return {
        "version": "4.1",
        "project_root": str(PROJECT_ROOT),
        "cwd": str(Path.cwd()),
        "data_path": str(DATA_PATH),
        "data_exists": DATA_PATH.exists(),
        "data_loaded": state.data_loaded,
        "records": len(state.df) if state.df is not None else 0,
        "features_count": len(state.df.columns) if state.df is not None else 0,
        "has_new_features": has_new_features,
        "completed_steps": {
            "data": state.data_loaded,
            "eda": state.eda_completed,
            "ml": state.ml_completed,
            "tests": state.tests_completed,
            "report": state.report_generated
        },
        "eda_results_available": state.eda_results is not None,
        "llm_available": state.llm_client.is_available
    }


@app.post("/api/export", response_model=Dict[str, Any], tags=["Экспорт"])
async def export_data(request: ExportRequest):
    if not state.data_loaded:
        raise HTTPException(status_code=400, detail="Данные не загружены")
    df = state.df.copy()
    if request.filters:
        if request.filters.keyword:
            df = df[df['keyword'].str.contains(request.filters.keyword, case=False)]
        if request.filters.cluster:
            df = df[df['cluster'] == request.filters.cluster]
        if request.filters.date_from:
            df = df[df['date'] >= request.filters.date_from]
        if request.filters.date_to:
            df = df[df['date'] <= request.filters.date_to]
    if request.columns:
        df = df[request.columns]
    export_path = DATA_PATH / f"export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    if request.format == 'csv':
        path = f"{export_path}.csv"
        df.to_csv(path, index=False)
    elif request.format == 'json':
        path = f"{export_path}.json"
        df.to_json(path, orient='records', force_ascii=False)
    elif request.format == 'excel':
        path = f"{export_path}.xlsx"
        df.to_excel(path, index=False)
    else:
        raise HTTPException(status_code=400, detail="Неподдерживаемый формат")
    return {
        "status": "success",
        "file": str(path),
        "records": len(df),
        "format": request.format
    }


# =====================================================
# ЗАПУСК
# =====================================================

if __name__ == "__main__":
    import uvicorn
    print("""
╔══════════════════════════════════════════════════════════════╗
║   🚀 ASO Analytics Service v4.1                              ║
║                                                              ║
║   API документация: http://localhost:8000/docs              ║
║   Главная страница:  http://localhost:8000                  ║
╚══════════════════════════════════════════════════════════════╝
    """)
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")