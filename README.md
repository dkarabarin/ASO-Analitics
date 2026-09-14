# 📊 ASO Analytics Service

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production-brightgreen?style=for-the-badge)

Веб-сервис для анализа ASO-данных приложения **Sound Amplifier**

</div>

---

## О проекте

Сервис автоматизирует полный пайплайн ASO-анализа: от загрузки Excel-файлов до расчёта мотива, ML-прогнозов и рекомендаций с помощью LLM.

| | Возможность |
|---|---|
| 🎯 | Расчёт требуемого мотива для достижения целевой позиции |
| 📈 | Прогноз изменения позиции на 1 день |
| ❄️ | Учёт 10 исторических периодов фризов |
| 📱 | Учёт 14 iOS-обновлений и их влияния |
| 🤖 | LLM-ассистент (Qwen 2.5) для анализа и стратегии |
| 📊 | 15+ графиков, 8 статистических гипотез, 14 правил ASO |

---

## Скриншоты

### Главная страница
<img width="1913" alt="main" src="https://github.com/user-attachments/assets/c53f7d30-5fd3-47de-8c97-decfcc9b7cd0" />

### Расчёт мотива и прогноз позиции
<img width="1916" alt="motiv-form" src="https://github.com/user-attachments/assets/63566b50-d750-4bf9-a31a-686a4b76f1d8" />
<img width="1916" alt="motiv-result" src="https://github.com/user-attachments/assets/34b0b12e-c9ba-4f4f-b5ba-cc95f99e9af9" />

### Анализ фризов и iOS-обновлений
<img width="2683" alt="freeze_and_updates" src="https://github.com/user-attachments/assets/84e5cdbd-8241-4e19-a82e-df236d8290ed" />

### ML-модели
<img width="2383" alt="feature_importance" src="https://github.com/user-attachments/assets/d29ed413-2c2c-4763-8b93-b4b545727ac7" />
<img width="2673" alt="model_comparison" src="https://github.com/user-attachments/assets/1dc09c41-c684-4fc8-835b-be57f8828a36" />

### Проверка гипотез и правил
<img width="1914" alt="hypotheses" src="https://github.com/user-attachments/assets/29a98426-0b87-4a1d-a827-7d22b9e5303c" />

### LLM-ассистент
<img width="1917" alt="llm" src="https://github.com/user-attachments/assets/590fc0c9-a2be-457f-a8f9-797194ce7ed5" />

---

## Установка

### Требования

- Python 3.11+
- 16+ GB ОЗУ (при использовании LLM)
- Ollama (опционально)

### Шаги

```bash
# 1. Клонирование
git clone https://github.com/your-username/aso-analytics-service.git
cd aso-analytics-service

# 2. Виртуальное окружение
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Зависимости
pip install --upgrade pip
pip install -r requirements.txt
```

### Ollama + LLM (опционально)

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh
# macOS
brew install ollama

# Запуск и загрузка модели
ollama serve
ollama pull qwen2.5:14b
```

### Подготовка данных

Поместите Excel-файлы в папку `data/`:

```
data/
├── Sound Amplifier_2024.xlsx
├── Sound Amplifier_2025.xlsx
└── Sound Amplifier_2026.xlsx
```

---

## Запуск

```bash
python run.py
# или
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

| Страница | URL |
|---|---|
| Главная | http://localhost:8000 |
| ML | http://localhost:8000/ml |
| Swagger | http://localhost:8000/docs |

---

## Пайплайн анализа

| Шаг | Описание |
|---|---|
| **1. Загрузка** | Парсинг Excel за 2024–2026, дедупликация, создание 121 признака |
| **2. EDA** | Статистика, корреляции, топ-20 ключей, 9 графиков |
| **3. Мотив** | Расчёт требуемого мотива, прогноз позиции, рекомендации R5/R8/R9/R12 |
| **4. ML** | Linear Regression, Random Forest, XGBoost, LightGBM + Optuna |
| **5. Тесты** | Гипотезы H1–H8 (Kruskal-Wallis, Mann-Whitney, Pearson/Spearman) |
| **6. Правила** | Проверка 14 правил ASO (R1–R14) |
| **7. Отчёт** | JSON-отчёт + LLM-анализ и стратегия продвижения |

---

## Структура проекта

```
aso-analytics-service/
├── app/
│   ├── main.py                  # FastAPI приложение
│   ├── config.py                # Конфигурация
│   ├── api/schemas.py           # Pydantic-схемы
│   ├── core/
│   │   ├── data_loader.py       # Шаг 1
│   │   ├── eda.py               # Шаг 2
│   │   ├── motiv_calculator.py  # Шаг 3
│   │   ├── ml_model.py          # Шаг 4
│   │   ├── statistical_tests.py # Шаг 5
│   │   ├── events_analyzer.py   # Фризы и iOS
│   │   └── reporting.py         # Шаги 6–7
│   ├── llm/client.py            # Qwen / Ollama
│   └── static/                  # Frontend (HTML/JS)
├── data/                        # Данные и результаты
├── notebooks/
│   └── ASO_Analysis.ipynb
├── requirements.txt
├── run.py
└── README.md
```

---

## API

<details>
<summary>Показать все эндпоинты</summary>

**Данные**

| Метод | Эндпоинт | Описание |
|---|---|---|
| POST | `/api/data/load` | Загрузка Excel |
| GET | `/api/data/status` | Статус |
| POST | `/api/data/reset` | Сброс |

**Мотив**

| Метод | Эндпоинт | Описание |
|---|---|---|
| GET | `/api/motiv/keywords` | Список ключей |
| POST | `/api/motiv/calculate` | Расчёт мотива |
| POST | `/api/motiv/batch_calculate` | Пакетный расчёт |
| GET | `/api/motiv/efficiency` | Эффективность |

**ML**

| Метод | Эндпоинт | Описание |
|---|---|---|
| POST | `/api/ml/train` | Обучение |
| GET | `/api/ml/results` | Результаты |
| GET | `/api/ml/feature_importance` | Важность признаков |
| POST | `/api/ml/predict` | Предсказание |

**События / Тесты / LLM**

| Метод | Эндпоинт | Описание |
|---|---|---|
| GET | `/api/events/timeline` | Таймлайн фризов и iOS |
| POST | `/api/tests/run` | Тесты H1–H8 |
| POST | `/api/llm/chat` | Диалог с LLM |
| POST | `/api/llm/analyze` | Анализ данных |
| GET | `/api/llm/status` | Статус LLM |

</details>

### Пример: расчёт мотива

```bash
curl -X POST http://localhost:8000/api/motiv/calculate \
  -H "Content-Type: application/json" \
  -d '{"keyword": "sound amplifier", "target_position": 10}'
```

```json
{
  "keyword": "sound amplifier",
  "current_position": 25.0,
  "target_position": 10,
  "required_motiv": 15,
  "predicted_position_1d": 10.0,
  "confidence": "medium",
  "recommendation": "Агрессивное продвижение (R12)"
}
```

---

## Проверка гипотез (H1–H8)

Автоматически запускается на данных после 20.06.2026 (или последних 30% записей). Результат — таблица + график `hypotheses_1day.png`.

| # | Гипотеза | Метод | Критерий | Результат |
|---|---|---|---|---|
| H1 | Похожие запросы коррелируют | t-test | Ср. корреляция > 0.7 | ❌ |
| H2 | Сценарии согласованы между ключами | Kruskal-Wallis | p < 0.05 | ❌ |
| H3 | Разные сценарии — разный эффект | Kruskal-Wallis | p < 0.05 | ❌ |
| H4 | Мотив = 0 меняет позицию | Mann-Whitney | p < 0.05 | ❌ |
| H5 | Мотив влияет на Δ позиции | Pearson + Spearman | p < 0.05 | ❌ |
| H6 | Средняя эффективность мотива > 0 | t-test (one-sided) | p < 0.05, ср. > 0 | ✅ |
| H7 🆕 | Фризы (события) меняют Δ позиции | Mann-Whitney | p < 0.05 | ✅ |
| H8 🆕 | iOS-обновления меняют Δ позиции | Mann-Whitney | p < 0.05 | ✅ |

**Итого: подтверждено 3 из 8 (37.5%)**

<details>
<summary>Примеры вывода</summary>

**H6 — эффективность мотива:**
```
Ключей: 99 | Ср. эффективность: 8.887 | Медиана: 4.402
Положительных: 93 (93.9%) | t: 8.13 | p: 0.0000 → ✅ ПОДТВЕРЖДЕНА
```

**H7 — влияние фризов:**
```
Во фризе:  Δ = 0.000  (n=1989)
Вне фриза: Δ = −5.175 (n=989)
Cohen's d: 0.366 | p: 0.0000 → ✅ ПОДТВЕРЖДЕНА
⚠️ Во время фризов продвижение не работает
```

**H8 — влияние iOS:**
```
После iOS: Δ = 0.000  (n=819)
Обычные:   Δ = −2.371
p: 0.0033 → ✅ ПОДТВЕРЖДЕНА
```

**H1 — корреляции (не подтверждена):**
```
Пар: 190 | Ср. корреляция: 0.980 | t: 196.74 | p: 1.0000 → ❌
Топ-пары: audio amplifier ↔ audio enhancer: 1.000
          amplify ↔ audio amplifier: 1.000
```

</details>

---

## Проверка правил (R1–R14)

Запускается на данных после 20.06.2026. Результат — таблица + график `rules_1day.png`.

| # | Правило | Критерий | Результат |
|---|---|---|---|
| R1 | Базовый цикл M = 1→1→0 | >50% паттернов дают улучшение | ⚠️ нет данных |
| R2 | Мотив > 0 улучшает позицию | Δ при мотив>0 < Δ при мотив=0 | ❌ |
| R3 | Рост >10 → мотив↑ | Ср. мотив после > до | ❌ |
| R4 | Пропуск при падении | Мотив при падении < в остальных | ✅ |
| R5 | Мотив ≤ 2 для новых ключей | max(motiv) ≤ 2 | ⚠️ нет данных |
| R6 | Целевые позиции по частотности | ВЧ ≤ 40, НЧ ≤ 20 | ❌ |
| R7 | Максимум 3 цикла с мотивом 2 | Ср. Δ < 0 после паттерна 2→2→0 | ⚠️ нет данных |
| R8 | Мотив 3 при росте 50% к цели | Мотив 3 лучше мотива 1 | ❌ |
| R9 | Процентный шаг 15%/10% | Ср. прирост мотива > 0 | ✅ |
| R10 | Откат −30% при отсутствии роста | >30% откатов дают восстановление | ❌ |
| R11 | Поддержка снижением на 2–3 | Паттерн найден | ⚠️ нет данных |
| R12 | Мотив ≤ 30% от органики | Нарушений < 10% | ✅ |
| R13 🆕 | Пауза во время фризов | Мотив↓ + эффект хуже | ✅ |
| R14 🆕 | Усиление после iOS | Мотив↑ + эффект лучше | ❌ |

**Итого: подтверждено 4 из 14 (28.6%)**

<details>
<summary>Примеры вывода</summary>

**R4 — пропуск при падении:**
```
При падении:  ср. мотив = 0.156 (n=173)
В остальных:  ср. мотив = 0.280
→ ✅ ПОДТВЕРЖДЕНО
```

**R12 — мотив ≤ 30% от органики:**
```
Записей: 2978 | Нарушений: 138 (4.6%) → ✅ ПОДТВЕРЖДЕНО
```

**R13 — пауза во фризах:**
```
Во фризе:  мотив = 0.043, Δ = 0.000  (n=1989)
Вне фриза: мотив = 0.734, Δ = −0.669
→ ✅ ПОДТВЕРЖДЕНО
```

**R6 — целевые позиции (не подтверждено):**
```
НЧ ср. позиция = 41.9 (цель ≤ 20) → ❌
```

**R8 — мотив 3 (не подтверждено):**
```
Мотив 3: ср. Δ = +0.053 (n=19)
Мотив 1: ср. Δ = −0.042 (n=24) → ❌
```

**R14 — усиление после iOS (не подтверждено):**
```
После iOS: мотив = 0.000, Δ = 0.000  (n=819)
Обычные:   мотив = 0.376, Δ = −2.371
→ ❌
```

</details>

---

## Методология

### Эффективность мотива

```
efficiency     = (avg_pos_without − avg_pos_with) / avg_motiv
required_motiv = (current_position − target_position) / efficiency
```

| Значение | Интерпретация |
|---|---|
| > 5 | 🟢 Отличная |
| 1–5 | 🟢 Хорошая |
| 0.3–1 | 🟡 Средняя |
| < 0.3 | 🔴 Низкая |
| ≤ 0 | ❌ Не работает |

---

## Решение проблем

**`ModuleNotFoundError: No module named 'numpy._core'`** — разные версии numpy:
```bash
pip install --upgrade --force-reinstall "numpy>=2.1" "pandas>=2.2"
```

**`clean_data.pkl` не загружается** — несовместимая версия pickle:
```python
df.to_csv("data/clean_data.csv", index=False)
```
Сервис автоматически прочитает CSV, если pickle недоступен.

**43 признака вместо 121** — в `data/` лежит старый `clean_data.pkl`. Перезапустите Шаг 1 в ноутбуке и перезапустите сервис.

**LLM недоступен** — не запущен Ollama:
```bash
ollama serve && ollama pull qwen2.5:14b
```

---

## Лицензия

MIT © 2024 Denis
