"""
Pydantic схемы для API запросов и ответов
Валидация данных для всех эндпоинтов
"""

from pydantic import BaseModel, Field, field_serializer
from typing import Optional, List, Dict, Any
from datetime import datetime
import math


# =====================================================
# ОБЩИЕ СХЕМЫ
# =====================================================

class StatusResponse(BaseModel):
    """Общий статус системы"""
    data_loaded: bool = False
    eda_completed: bool = False
    clustering_completed: bool = False
    ml_completed: bool = False
    tests_completed: bool = False
    report_generated: bool = False
    llm_available: bool = False
    records: int = 0
    keywords: int = 0
    
    class Config:
        json_schema_extra = {
            "example": {
                "data_loaded": True,
                "eda_completed": True,
                "clustering_completed": True,
                "ml_completed": True,
                "tests_completed": True,
                "report_generated": True,
                "llm_available": True,
                "records": 94002,
                "keywords": 134
            }
        }


class ErrorResponse(BaseModel):
    """Ответ при ошибке"""
    status: str = "error"
    message: str
    detail: Optional[str] = None


class SuccessResponse(BaseModel):
    """Ответ при успехе"""
    status: str = "success"
    message: str
    data: Optional[Dict[str, Any]] = None


# =====================================================
# ЗАГРУЗКА ДАННЫХ (Шаг 1)
# =====================================================

class DataLoadResponse(BaseModel):
    """Ответ на загрузку данных"""
    status: str
    message: str
    records: int
    keywords: int
    period_start: Optional[str] = None
    period_end: Optional[str] = None


class DataStatusResponse(BaseModel):
    """Статус загрузки данных"""
    loaded: bool
    records: Optional[int] = None
    keywords: Optional[int] = None
    clusters: Optional[int] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None


# =====================================================
# EDA (Шаг 2)
# =====================================================

class EDASummaryResponse(BaseModel):
    """Сводка EDA"""
    total_records: int
    unique_keywords: int
    unique_dates: int
    years: List[int]
    period_start: str
    period_end: str


class TopKeywordsResponse(BaseModel):
    """Топ ключевых слов"""
    top_keywords: Dict[str, int]


class PlotItem(BaseModel):
    """Элемент списка графиков"""
    name: str
    url: str
    size: int = Field(..., description="Размер файла в байтах")
    modified: float = Field(..., description="Время модификации в секундах")
    step: Optional[int] = Field(None, description="Номер шага")
    step_name: Optional[str] = Field(None, description="Название шага")


class PlotsListResponse(BaseModel):
    """Ответ со списком графиков"""
    plots: List[PlotItem]


class PlotsByStepResponse(BaseModel):
    """Ответ с графиками по шагам"""
    step: int
    step_name: str
    plots: List[PlotItem]
    total: int


class PlotsGroupedResponse(BaseModel):
    """Ответ с группированными графиками"""
    plots_by_step: Dict[int, Dict[str, Any]]
    total: int


# =====================================================
# КЛАСТЕРИЗАЦИЯ (Шаг 3)
# =====================================================

class ClusteringRunResponse(BaseModel):
    """Ответ на кластеризацию"""
    status: str
    message: str
    clusters: int


class ClusterInfo(BaseModel):
    """Информация о кластере"""
    keywords: int
    motiv_mean: float = 0.0
    position_mean: float = 0.0
    motiv_sum: Optional[float] = 0.0
    position_min: Optional[float] = 0.0
    position_max: Optional[float] = 0.0
    
    @field_serializer('motiv_mean', 'position_mean', 'motiv_sum', 'position_min', 'position_max')
    def serialize_float(self, value: float) -> float:
        if value is None or math.isnan(value) or math.isinf(value):
            return 0.0
        return value


class ClustersResponse(BaseModel):
    """Ответ с кластерами"""
    clusters: Dict[str, ClusterInfo]


class AnchorWord(BaseModel):
    """Якорное слово"""
    cluster: str
    anchor_word: str
    motiv_mean: float = 0.0
    motiv_sum: float = 0.0
    position_mean: float = 0.0
    anchor_score: float = 0.0
    
    @field_serializer('motiv_mean', 'motiv_sum', 'position_mean', 'anchor_score')
    def serialize_float(self, value: float) -> float:
        if value is None or math.isnan(value) or math.isinf(value):
            return 0.0
        return value


class AnchorWordsResponse(BaseModel):
    """Ответ с якорными словами"""
    anchor_words: List[AnchorWord]


# =====================================================
# ML (Шаг 4)
# =====================================================

class MLTrainResponse(BaseModel):
    """Ответ на обучение моделей"""
    status: str
    message: str
    best_model: str


class ModelResult(BaseModel):
    """Результаты модели"""
    Model: str
    MAE: float = 0.0
    RMSE: float = 0.0
    R2: float = 0.0
    
    @field_serializer('MAE', 'RMSE', 'R2')
    def serialize_float(self, value: float) -> float:
        if value is None or math.isnan(value) or math.isinf(value):
            return 0.0
        return value


class MLResultsResponse(BaseModel):
    """Ответ с результатами моделей"""
    results: List[ModelResult]


class FeatureImportanceItem(BaseModel):
    """Элемент важности признака"""
    feature: str
    importance: float = 0.0
    importance_pct: float = 0.0
    group: str
    
    @field_serializer('importance', 'importance_pct')
    def serialize_float(self, value: float) -> float:
        if value is None or math.isnan(value) or math.isinf(value):
            return 0.0
        return value


class FeatureImportanceResponse(BaseModel):
    """Ответ с важностью признаков"""
    feature_importance: List[FeatureImportanceItem]


class PredictRequest(BaseModel):
    """Запрос на предсказание"""
    motiv: float = Field(0, description="Текущий мотив")
    position: float = Field(0, description="Текущая позиция")
    organic_us: float = Field(0, description="Органический трафик US")
    activation_us: float = Field(0, description="Активации US")
    motiv_lag_7: float = Field(0, description="Мотив за 7 дней")
    position_lag_7: float = Field(0, description="Позиция за 7 дней")
    motiv_lag_14: float = Field(0, description="Мотив за 14 дней")
    position_lag_14: float = Field(0, description="Позиция за 14 дней")
    motiv_ma_7: float = Field(0, description="Скользящая средняя мотива 7 дней")
    position_ma_7: float = Field(0, description="Скользящая средняя позиции 7 дней")
    motiv_std_7: float = Field(0, description="Стандартное отклонение мотива 7 дней")
    motiv_ma_14: float = Field(0, description="Скользящая средняя мотива 14 дней")
    position_ma_14: float = Field(0, description="Скользящая средняя позиции 14 дней")
    month: int = Field(1, description="Месяц (1-12)")
    quarter: int = Field(1, description="Квартал (1-4)")
    day_of_week: int = Field(0, description="День недели (0-6)")
    is_weekend: int = Field(0, description="Выходной (0/1)")
    keyword_encoded: Optional[int] = Field(None, description="Закодированное ключевое слово")
    cluster_encoded: Optional[int] = Field(None, description="Закодированный кластер")
    keyword: Optional[str] = Field(None, description="Ключевое слово (будет закодировано автоматически)")
    cluster: Optional[str] = Field(None, description="Кластер (будет закодирован автоматически)")


class PredictResponse(BaseModel):
    """Ответ на предсказание"""
    prediction: float = 0.0
    prediction_original: float = 0.0
    interpretation: Optional[str] = None
    
    @field_serializer('prediction', 'prediction_original')
    def serialize_float(self, value: float) -> float:
        if value is None or math.isnan(value) or math.isinf(value):
            return 0.0
        return value


# =====================================================
# СТАТИСТИЧЕСКИЕ ТЕСТЫ (Шаг 5)
# =====================================================

class HypothesisResult(BaseModel):
    """Результат гипотезы"""
    cluster: str
    keywords_count: int
    H1_result: Optional[str] = None
    H3_result: Optional[str] = None
    H4_result: Optional[str] = None
    H5_result: Optional[str] = None


class TestsResultsResponse(BaseModel):
    """Ответ с результатами тестов"""
    hypotheses: List[Dict[str, Any]]


# =====================================================
# ОТЧЁТЫ (Шаг 6)
# =====================================================

class ReportGenerateResponse(BaseModel):
    """Ответ на генерацию отчёта"""
    status: str
    message: str
    report_path: str


class ReportResponse(BaseModel):
    """Ответ с отчётом"""
    report_metadata: Dict[str, Any]
    executive_summary: Dict[str, Any]
    data_summary: Dict[str, Any]
    clustering_results: Dict[str, Any]
    ml_results: Dict[str, Any]
    hypothesis_results: Dict[str, Any]
    rules_results: Dict[str, Any]
    recommendations: Dict[str, List[Dict[str, Any]]]


# =====================================================
# LLM (Шаг 7)
# =====================================================

class LLMAnalyzeRequest(BaseModel):
    """Запрос на анализ данных через LLM"""
    question: str = Field(..., description="Вопрос для LLM")
    context: Optional[str] = Field(None, description="Дополнительный контекст")


class LLMChatRequest(BaseModel):
    """Запрос на чат с LLM"""
    message: str = Field(..., description="Сообщение пользователя")
    context: Optional[str] = Field(None, description="Дополнительный контекст")


class LLMResponse(BaseModel):
    """Ответ от LLM"""
    response: str


class LLMStatusResponse(BaseModel):
    """Статус LLM"""
    is_available: bool
    model: str
    url: str


class MethodologyResponse(BaseModel):
    """Ответ с методологией"""
    methodology: str
    version: str
    rules_count: int
    clusters: List[str]
    scenarios: List[Dict[str, str]]


# =====================================================
# ДАННЫЕ (общие схемы)
# =====================================================

class KeywordStats(BaseModel):
    """Статистика по ключевому слову"""
    keyword: str
    count: int
    motiv_mean: float = 0.0
    motiv_max: float = 0.0
    motiv_sum: float = 0.0
    position_mean: float = 0.0
    position_min: float = 0.0
    position_max: float = 0.0
    
    @field_serializer('motiv_mean', 'motiv_max', 'motiv_sum', 'position_mean', 'position_min', 'position_max')
    def serialize_float(self, value: float) -> float:
        if value is None or math.isnan(value) or math.isinf(value):
            return 0.0
        return value


class DateRange(BaseModel):
    """Диапазон дат"""
    start: str
    end: str


class PaginationParams(BaseModel):
    """Параметры пагинации"""
    page: int = Field(1, ge=1, description="Номер страницы")
    page_size: int = Field(20, ge=1, le=100, description="Размер страницы")
    sort_by: Optional[str] = Field(None, description="Поле для сортировки")
    sort_order: str = Field("asc", description="Порядок сортировки (asc/desc)")


# =====================================================
# ВСПОМОГАТЕЛЬНЫЕ СХЕМЫ
# =====================================================

class FilterParams(BaseModel):
    """Параметры фильтрации"""
    keyword: Optional[str] = None
    cluster: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    motiv_min: Optional[float] = None
    motiv_max: Optional[float] = None
    position_min: Optional[float] = None
    position_max: Optional[float] = None


class ExportRequest(BaseModel):
    """Запрос на экспорт данных"""
    format: str = Field("csv", description="Формат экспорта (csv, json, excel)")
    filters: Optional[FilterParams] = None
    columns: Optional[List[str]] = None


class ExplainPlotRequest(BaseModel):
    """Запрос на объяснение графика"""
    plot_name: str = Field(..., description="Имя графика")
    description: str = Field("", description="Описание графика")
    context: str = Field("", description="Контекст данных")


# =====================================================
# ПРИМЕРЫ ДЛЯ ДОКУМЕНТАЦИИ
# =====================================================

class Examples:
    """Примеры для Swagger документации"""
    
    PREDICT_REQUEST = {
        "motiv": 5.0,
        "position": 30.0,
        "organic_us": 100.0,
        "activation_us": 50.0,
        "motiv_lag_7": 4.0,
        "position_lag_7": 32.0,
        "motiv_lag_14": 3.0,
        "position_lag_14": 35.0,
        "motiv_ma_7": 4.5,
        "position_ma_7": 31.0,
        "motiv_std_7": 1.2,
        "motiv_ma_14": 4.0,
        "position_ma_14": 33.0,
        "month": 8,
        "quarter": 3,
        "day_of_week": 3,
        "is_weekend": 0,
        "keyword": "sound amplifier",
        "cluster": "top50_high"
    }
    
    LLM_CHAT_REQUEST = {
        "message": "Какие ключевые слова имеют наибольший потенциал для продвижения?",
        "context": "Данные за 2026 год показывают рост позиций для ключей в кластере top10_high"
    }
    
    EXPLAIN_PLOT_REQUEST = {
        "plot_name": "eda_overview.png",
        "description": "Общий обзор EDA с распределением мотива и позиций",
        "context": "Данные за период 2023-2026"
    }