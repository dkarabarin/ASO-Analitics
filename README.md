# ASO Analytics Service

Сервис для анализа ASO-данных мобильного приложения: EDA, расчёт эффективности мотива, анализ событий (фризы, обновления iOS), статистическая проверка гипотез, ML-прогноз изменения позиции ключевых слов (XGBoost / LightGBM), кластеризация ключевых слов и LLM-ассистент (Qwen через Ollama).

## Стек

FastAPI · pandas · scikit-learn · XGBoost · LightGBM · Optuna · SciPy · statsmodels · matplotlib / seaborn / plotly

## Структура

```
app/
  api/        — Pydantic-схемы
  core/       — загрузка данных, EDA, статистика, ML, кластеризация, отчёты
  llm/        — клиент LLM (Ollama)
  static/     — веб-интерфейс
  main.py     — FastAPI-приложение
notebooks/    — Jupyter-ноутбук с полным анализом
data/         — входные данные и результаты (не входят в репозиторий)
run.py        — точка входа
```

## Данные

Исходные данные в репозиторий не включены. Для запуска положите в папку `data/` Excel-файлы:

```
data/app_data_2024.xlsx
data/app_data_2025.xlsx
data/app_data_2026.xlsx
```

Формат листа:

| Строка | Содержимое |
|---|---|
| 0 | даты по колонкам |
| 1 | заголовки |
| 2 | органика и активации (US), общие для всех ключей |
| 3 | заголовки данных (мотив / позиция) |
| 4+ | ключевое слово в колонке C, далее мотив и позиция по датам |

Очищенный датасет, модели и графики сохраняются в `data/` при запуске анализа.

## Запуск

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

- Главная страница: http://localhost:8000
- ML-страница: http://localhost:8000/ml
- API-документация: http://localhost:8000/docs

Для LLM-ассистента нужен локально запущенный [Ollama](https://ollama.com) с моделью Qwen.
