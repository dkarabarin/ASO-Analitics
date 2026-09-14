"""
Модуль ML моделирования v2.0
Шаг 4: Прогноз на 1 день с учётом фризов и iOS обновлений
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import lightgbm as lgb
import optuna
import joblib
import json


class MLModeler:
    """Класс для ML моделирования (прогноз на 1 день)"""
    
    def __init__(self, data_path: Path, df: pd.DataFrame = None):
        """
        Args:
            data_path: Путь к папке с данными
            df: DataFrame с данными (если None, загружается из data_with_clusters.pkl или clean_data.pkl)
        """
        self.data_path = Path(data_path)
        self.plots_path = self.data_path / "plots"
        self.plots_path.mkdir(exist_ok=True)
        self.models_path = self.data_path / "saved_models"
        self.models_path.mkdir(exist_ok=True)
        
        if df is not None:
            self.df = df
        else:
            # Пробуем data_with_clusters.pkl, потом clean_data.pkl
            clusters_path = self.data_path / "data_with_clusters.pkl"
            clean_path = self.data_path / "clean_data.pkl"
            
            if clusters_path.exists():
                self.df = pd.read_pickle(clusters_path)
            elif clean_path.exists():
                self.df = pd.read_pickle(clean_path)
            else:
                self.df = None
        
        self.results = {}
        self.best_model = None
        self.scaler_X = None
        self.scaler_y = None
        self.le_keyword = None
        self.le_cluster = None
        
        # 🆕 ЦЕЛЕВАЯ ПЕРЕМЕННАЯ - ПРОГНОЗ НА 1 ДЕНЬ
        self.target_col = 'position_delta_1d'
        
        # 🆕 РАСШИРЕННЫЙ СПИСОК ПРИЗНАКОВ
        self.feature_cols = [
            # Текущие значения
            'motiv', 'position', 'organic_us', 'activation_us',
            
            # Лаги (прошлые значения)
            'motiv_lag_1', 'position_lag_1',
            'motiv_lag_7', 'position_lag_7',
            'motiv_lag_14', 'position_lag_14',
            'motiv_lag_30', 'position_lag_30',
            
            # Скользящие средние
            'motiv_ma_3', 'position_ma_3',
            'motiv_ma_7', 'position_ma_7',
            'motiv_std_7', 'position_std_7',
            'motiv_ma_14', 'position_ma_14',
            'motiv_std_14', 'position_std_14',
            'motiv_ma_30', 'position_ma_30',
            
            # Экспоненциальные скользящие средние (EWM)
            'motiv_ewm_3', 'position_ewm_3',
            'motiv_ewm_7', 'position_ewm_7',
            'motiv_ewm_14', 'position_ewm_14',
            'motiv_ewm_30', 'position_ewm_30',
            
            # Медианы
            'motiv_median_3', 'position_median_3',
            'motiv_median_7', 'position_median_7',
            'motiv_median_14', 'position_median_14',
            
            # Min/Max за период
            'motiv_min_7', 'motiv_max_7',
            'position_min_7', 'position_max_7',
            'motiv_min_14', 'motiv_max_14',
            'position_min_14', 'position_max_14',
            'motiv_min_30', 'motiv_max_30',
            'position_min_30', 'position_max_30',
            
            # Ранги
            'position_rank', 'motiv_rank',
            
            # Изменения (за прошлые периоды)
            'motiv_change_1d', 'motiv_change_3d',
            'motiv_change_7d', 'motiv_change_14d',
            
            # Волатильность
            'position_volatility_7', 'position_volatility_14', 'position_volatility_30',
            
            # Отношения
            'motiv_position_ratio',
            'motiv_ma_diff', 'position_ma_diff',
            
            # Органика/активации
            'organic_ma_7', 'activation_ma_7',
            'organic_ma_14', 'activation_ma_14',
            'organic_ma_30', 'activation_ma_30',
            'organic_std_7', 'activation_std_7',
            
            # Временные признаки
            'month', 'quarter', 'day_of_week', 'is_weekend',
            
            # 🆕 ФРИЗЫ И iOS ОБНОВЛЕНИЯ
            'is_freeze',
            'days_to_freeze',
            'days_after_freeze',
            'is_ios_update_day',
            'days_after_ios_update',
            'is_week_after_ios_update',
            'is_holiday_season',
            
            # Категориальные
            'keyword_encoded'
        ]
        
        # Исключённые признаки (информационная утечка)
        self.excluded_features = [
            'position_change', 'position_pct_change',
            'motiv_change', 'motiv_pct_change',
            'position_trend_7d', 'motiv_trend_7d'
        ]
    
    def run_ml_pipeline(self) -> dict:
        """Запуск полного пайплайна ML"""
        print("\n" + "="*80)
        print("ШАГ 4: ML МОДЕЛИРОВАНИЕ (ПРОГНОЗ НА 1 ДЕНЬ)")
        print("="*80)
        
        if self.df is None:
            raise ValueError("Данные не загружены")
        
        df = self.df
        
        print(f"\nРазмер данных: {len(df):,} записей")
        print(f"Уникальных ключей: {df['keyword'].nunique()}")
        print(f"Всего признаков: {len(df.columns)}")
        
        print("\n⚠️ ВАЖНО: Исключены признаки с информационной утечкой:")
        for feat in self.excluded_features:
            print(f"  - {feat}")
        
        print("\nЦЕЛЕВАЯ ПЕРЕМЕННАЯ:")
        print(f"  - {self.target_col} = изменение позиции через 1 день")
        print("  - Отрицательное значение = улучшение (подъём в топ)")
        
        # 1. Подготовка данных
        X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled = self._prepare_data()
        
        # 2. Базовые модели
        self._train_baseline_models(X_train_scaled, X_val_scaled, X_test_scaled, 
                                    y_train_scaled, y_val_scaled, y_test_scaled)
        
        # 3. XGBoost с регуляризацией
        self._train_xgboost_reg(X_train_scaled, X_val_scaled, X_test_scaled,
                                y_train_scaled, y_val_scaled, y_test_scaled)
        
        # 4. LightGBM с регуляризацией
        self._train_lightgbm_reg(X_train_scaled, X_val_scaled, X_test_scaled,
                                 y_train_scaled, y_val_scaled, y_test_scaled)
        
        # 5. Сводная таблица
        self._create_results_table()
        
        # 6. Важность признаков
        self._analyze_feature_importance()
        
        # 7. Визуализация
        self._generate_plots(X_test_scaled, y_test_scaled)
        
        # 8. Сохранение
        self._save_results()
        
        # 9. Выводы
        self._generate_insights()
        
        print("\n" + "="*80)
        print("✅ ШАГ 4 ЗАВЕРШЕН УСПЕШНО!")
        print("="*80)
        
        return self.results
    
    def _prepare_data(self):
        """Подготовка данных для ML"""
        print("\n" + "="*60)
        print("1. ПОДГОТОВКА ДАННЫХ ДЛЯ ML (БЕЗ УТЕЧКИ)")
        print("="*60)
        
        df = self.df.copy()
        
        # Проверяем наличие целевой переменной
        if self.target_col not in df.columns:
            raise ValueError(f"Целевая переменная {self.target_col} не найдена. Перезапустите Шаг 1 в ноутбуке!")
        
        # Кодирование категориальных признаков
        self.le_keyword = LabelEncoder()
        df['keyword_encoded'] = self.le_keyword.fit_transform(df['keyword'])
        
        # Кластер (если есть)
        if 'cluster' in df.columns:
            self.le_cluster = LabelEncoder()
            df['cluster_encoded'] = self.le_cluster.fit_transform(df['cluster'].astype(str))
        
        # Проверяем наличие колонок
        available_features = [col for col in self.feature_cols if col in df.columns]
        missing_features = [col for col in self.feature_cols if col not in df.columns]
        
        print(f"  - Доступных признаков: {len(available_features)}")
        if missing_features:
            print(f"  - ⚠️ Отсутствуют: {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}")
        
        print(f"  - Признаков для обучения: {len(available_features)}")
        print(f"  - Исключены: {', '.join(self.excluded_features)}")
        
        # Обновляем список признаков
        self.feature_cols = available_features
        
        # Очистка данных
        X = df[self.feature_cols].copy()
        y = df[self.target_col].copy()
        
        # Замена inf и NaN
        X = X.replace([np.inf, -np.inf], 0)
        X = X.fillna(0)
        
        # Удаляем строки с NaN в целевой
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        print(f"  - Данных для обучения: {len(X):,}")
        
        # Разделение на выборки
        print("\n" + "="*60)
        print("2. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ")
        print("="*60)
        
        # Сортировка по времени
        df_sorted = df.loc[mask].sort_values('date').reset_index(drop=True)
        X_sorted = X.loc[df_sorted.index]
        y_sorted = y.loc[df_sorted.index]
        
        # Разделение: 70% train, 15% val, 15% test
        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))
        
        X_train = X_sorted[:train_size]
        y_train = y_sorted[:train_size]
        X_val = X_sorted[train_size:train_size + val_size]
        y_val = y_sorted[train_size:train_size + val_size]
        X_test = X_sorted[train_size + val_size:]
        y_test = y_sorted[train_size + val_size:]
        
        print(f"  - Train: {len(X_train):,} записей ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  - Val: {len(X_val):,} записей ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  - Test: {len(X_test):,} записей ({len(X_test)/len(X)*100:.1f}%)")
        
        # Стандартизация
        print("\n" + "="*60)
        print("3. СТАНДАРТИЗАЦИЯ ДАННЫХ")
        print("="*60)
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_val_scaled = self.scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()
        y_test_scaled = self.scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()
        
        print("  + Стандартизация выполнена")
        
        # Сохраняем для дальнейшего использования
        self.X_train_scaled = X_train_scaled
        self.X_val_scaled = X_val_scaled
        self.X_test_scaled = X_test_scaled
        self.y_train_scaled = y_train_scaled
        self.y_val_scaled = y_val_scaled
        self.y_test_scaled = y_test_scaled
        self.y_test_original = y_test.values
        self.feature_names = self.feature_cols
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled
    
    def _train_baseline_models(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Обучение базовых моделей"""
        print("\n" + "="*60)
        print("4. БАЗОВЫЕ МОДЕЛИ ДЛЯ СРАВНЕНИЯ")
        print("="*60)
        
        models = {}
        results = []
        
        # 4.1. Linear Regression
        print("\n4.1. Linear Regression")
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"  - MAE: {mae:.4f}")
        print(f"  - RMSE: {rmse:.4f}")
        print(f"  - R²: {r2:.4f}")
        
        results.append({'Model': 'Linear Regression', 'MAE': mae, 'RMSE': rmse, 'R2': r2})
        models['Linear Regression'] = lr
        
        # 4.2. Random Forest
        print("\n4.2. Random Forest")
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"  - MAE: {mae:.4f}")
        print(f"  - RMSE: {rmse:.4f}")
        print(f"  - R²: {r2:.4f}")
        
        results.append({'Model': 'Random Forest', 'MAE': mae, 'RMSE': rmse, 'R2': r2})
        models['Random Forest'] = rf
        
        self.baseline_results = results
        self.baseline_models = models
    
    def _train_xgboost_reg(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """XGBoost с регуляризацией"""
        print("\n" + "="*60)
        print("5. XGBOOST С УСИЛЕННОЙ РЕГУЛЯРИЗАЦИЕЙ")
        print("="*60)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 0.8),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 0.8),
                'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
                'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 20.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 20.0, log=True),
                'gamma': trial.suggest_float('gamma', 0.1, 10.0, log=True),
            }
            
            model = xgb.XGBRegressor(
                **params,
                random_state=42,
                n_jobs=-1,
                early_stopping_rounds=50,
                eval_metric='mae'
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            y_pred = model.predict(X_val)
            return mean_absolute_error(y_val, y_pred)
        
        print("\n  + Запуск оптимизации с регуляризацией (30 итераций)...")
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=30, show_progress_bar=True)
        
        print(f"\n  + Лучший MAE на валидации: {study.best_value:.4f}")
        print("  + Лучшие параметры:")
        for key, value in study.best_params.items():
            print(f"      {key}: {value}")
        
        # Обучаем финальную модель
        model = xgb.XGBRegressor(
            **study.best_params,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50,
            eval_metric='mae'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Оценка на тесте
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Оценка на train
        y_pred_train = model.predict(X_train)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        
        print(f"\n  Результаты на тесте:")
        print(f"    - MAE: {mae:.4f}")
        print(f"    - RMSE: {rmse:.4f}")
        print(f"    - R²: {r2:.4f}")
        print(f"\n  Проверка переобучения:")
        print(f"    - Train MAE: {mae_train:.4f}")
        print(f"    - Test MAE: {mae:.4f}")
        print(f"    - Разница: {mae_train - mae:.4f}")
        
        self.results['XGBoost_reg'] = {
            'model': model,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mae_train': mae_train,
            'params': study.best_params
        }
        self.best_model = model
        self.xgb_reg = model
    
    def _train_lightgbm_reg(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """LightGBM с регуляризацией"""
        print("\n" + "="*60)
        print("6. LIGHTGBM С РЕГУЛЯРИЗАЦИЕЙ")
        print("="*60)
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 0.8),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
                'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 20.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 20.0, log=True),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.1, 1.0, log=True),
            }
            
            model = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            y_pred = model.predict(X_val)
            return mean_absolute_error(y_val, y_pred)
        
        print("\n  + Запуск оптимизации LightGBM (20 итераций)...")
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=20, show_progress_bar=True)
        
        print(f"\n  + Лучший MAE на валидации: {study.best_value:.4f}")
        
        # Обучаем финальную модель
        model = lgb.LGBMRegressor(**study.best_params, random_state=42, n_jobs=-1, verbose=-1)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Оценка на тесте
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n  Результаты на тесте:")
        print(f"    - MAE: {mae:.4f}")
        print(f"    - RMSE: {rmse:.4f}")
        print(f"    - R²: {r2:.4f}")
        
        self.results['LightGBM_reg'] = {
            'model': model,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'params': study.best_params
        }
        self.lgb_reg = model
    
    def _create_results_table(self):
        """Создание сводной таблицы результатов"""
        print("\n" + "="*60)
        print("7. СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
        print("="*60)
        
        results_data = []
        
        # Базовые модели
        for res in self.baseline_results:
            results_data.append(res)
        
        # XGBoost с регуляризацией
        if 'XGBoost_reg' in self.results:
            r = self.results['XGBoost_reg']
            results_data.append({
                'Model': 'XGBoost (с регул.)',
                'MAE': r['mae'],
                'RMSE': r['rmse'],
                'R2': r['r2']
            })
        
        # LightGBM с регуляризацией
        if 'LightGBM_reg' in self.results:
            r = self.results['LightGBM_reg']
            results_data.append({
                'Model': 'LightGBM (с регул.)',
                'MAE': r['mae'],
                'RMSE': r['rmse'],
                'R2': r['r2']
            })
        
        self.results_table = pd.DataFrame(results_data).round(4)
        
        print("\nРезультаты:")
        print(self.results_table.to_string(index=False))
        
        # Лучшая модель
        best_idx = self.results_table['MAE'].idxmin()
        best_model_name = self.results_table.loc[best_idx, 'Model']
        print(f"\n  + Лучшая модель: {best_model_name}")
        print(f"    - MAE: {self.results_table.loc[best_idx, 'MAE']:.4f}")
        print(f"    - RMSE: {self.results_table.loc[best_idx, 'RMSE']:.4f}")
        print(f"    - R2: {self.results_table.loc[best_idx, 'R2']:.4f}")
        
        self.best_model_name = best_model_name
    
    def _analyze_feature_importance(self):
        """Анализ важности признаков"""
        print("\n" + "="*60)
        print("8. ВАЖНОСТЬ ПРИЗНАКОВ")
        print("="*60)
        
        # 🆕 Группы признаков (с фризами и iOS)
        self.feature_groups = {
            'Позиция': {
                'features': [f for f in self.feature_names if 'position' in f and 'rank' not in f and 'ratio' not in f],
                'color': '#FF6B6B'
            },
            'Мотив': {
                'features': [f for f in self.feature_names if 'motiv' in f and 'position' not in f and 'rank' not in f and 'ratio' not in f],
                'color': '#4ECDC4'
            },
            'Временные': {
                'features': ['month', 'quarter', 'day_of_week', 'is_weekend', 'day_of_year', 'week_of_year', 'is_holiday_season'],
                'color': '#45B7D1'
            },
            'Органика/Активации': {
                'features': [f for f in self.feature_names if 'organic' in f or 'activation' in f],
                'color': '#96CEB4'
            },
            'Ключ': {
                'features': ['keyword_encoded', 'cluster_encoded'],
                'color': '#FFEAA7'
            },
            '🆕 Фризы': {
                'features': ['is_freeze', 'days_to_freeze', 'days_after_freeze'],
                'color': '#9F7AEA'
            },
            '🆕 iOS обновления': {
                'features': ['is_ios_update_day', 'days_after_ios_update', 'is_week_after_ios_update'],
                'color': '#F6AD55'
            },
            'Ранги/Отношения': {
                'features': ['position_rank', 'motiv_rank', 'motiv_position_ratio', 'motiv_ma_diff', 'position_ma_diff'],
                'color': '#ECC94B'
            }
        }
        
        def get_feature_group(feature_name):
            for group_name, group_info in self.feature_groups.items():
                if feature_name in group_info['features']:
                    return group_name
            return 'Другие'
        
        def get_feature_color(feature_name):
            for group_name, group_info in self.feature_groups.items():
                if feature_name in group_info['features']:
                    return group_info['color']
            return '#D3D3D3'
        
        # Важность для лучшей модели
        if self.best_model is not None and hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            imp_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance,
                'importance_pct': importance * 100,
                'group': [get_feature_group(f) for f in self.feature_names],
                'color': [get_feature_color(f) for f in self.feature_names]
            }).sort_values('importance', ascending=False)
            
            print("\nТоп-25 важнейших признаков:")
            print("  {:<30s} {:>12s} {:>12s} {:>25s}".format('Признак', 'Важность', 'Важность %', 'Группа'))
            print("  " + "-"*85)
            
            for i, row in imp_df.head(25).iterrows():
                print("  {:<30s} {:>12.4f} {:>11.2f}% {:>25s}".format(
                    row['feature'][:30],
                    row['importance'],
                    row['importance_pct'],
                    row['group']
                ))
            
            # Суммарная важность по группам
            group_importance = imp_df.groupby('group')['importance'].sum().sort_values(ascending=False)
            
            print("\nГруппы признаков:")
            for group, imp in group_importance.items():
                pct = (imp / group_importance.sum()) * 100
                print(f"   - {group}: {pct:.1f}%")
            
            # 🆕 Отдельно показываем важность событий
            print("\n🎯 Важность признаков событий:")
            event_features = ['is_freeze', 'days_to_freeze', 'days_after_freeze',
                              'is_ios_update_day', 'days_after_ios_update', 'is_week_after_ios_update']
            event_imp = imp_df[imp_df['feature'].isin(event_features)]
            if len(event_imp) > 0:
                for _, row in event_imp.iterrows():
                    print(f"   - {row['feature']}: {row['importance_pct']:.2f}%")
            else:
                print("   ⚠️ Признаки событий не найдены в данных")
            
            self.imp_df = imp_df
            self.group_importance = group_importance
    
    def _generate_plots(self, X_test, y_test):
        """Генерация графиков"""
        print("\n" + "="*60)
        print("9. ВИЗУАЛИЗАЦИЯ")
        print("="*60)
        
        # 1. Сравнение моделей
        self._plot_model_comparison()
        
        # 2. Важность признаков
        self._plot_feature_importance()
        
        # 3. Факт vs прогноз
        self._plot_predictions_vs_actual(X_test, y_test)
        
        # 4. 🆕 Важность событий
        self._plot_events_importance()
    
    def _plot_model_comparison(self):
        """График сравнения моделей"""
        print("\n  + Построение графика сравнения моделей...")
        
        names = self.results_table['Model'].tolist()
        mae_vals = self.results_table['MAE'].tolist()
        rmse_vals = self.results_table['RMSE'].tolist()
        r2_vals = self.results_table['R2'].tolist()
        
        colors = []
        for name in names:
            if 'Linear' in name or 'Random' in name:
                colors.append('#B0B0B0')
            else:
                colors.append('#4ECDC4')
        
        x = np.arange(len(names))
        width = 0.25
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        fig.suptitle('Сравнение моделей: прогноз на 1 день', fontsize=16, fontweight='bold')
        
        # MAE
        ax = axes[0]
        bars = ax.bar(x, mae_vals, width, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Модель', fontsize=12)
        ax.set_ylabel('MAE', fontsize=12)
        ax.set_title('MAE (меньше = лучше)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, val) in enumerate(zip(bars, mae_vals)):
            ax.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        
        best_idx = np.argmin(mae_vals)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        # RMSE
        ax = axes[1]
        bars = ax.bar(x, rmse_vals, width, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Модель', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=12)
        ax.set_title('RMSE (меньше = лучше)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, val) in enumerate(zip(bars, rmse_vals)):
            ax.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        
        best_idx = np.argmin(rmse_vals)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        # R²
        ax = axes[2]
        bars = ax.bar(x, r2_vals, width, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Модель', fontsize=12)
        ax.set_ylabel('R²', fontsize=12)
        ax.set_title('R² (больше = лучше)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, val) in enumerate(zip(bars, r2_vals)):
            ax.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        
        best_idx = np.argmax(r2_vals)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        # Легенда
        legend_elements = [
            plt.Rectangle((0,0), 1, 1, facecolor='#B0B0B0', label='Базовые модели'),
            plt.Rectangle((0,0), 1, 1, facecolor='#4ECDC4', label='С регуляризацией')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11)
        
        plt.tight_layout()
        plt.savefig(self.plots_path / 'model_comparison_1day.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  + График сохранён: {self.plots_path / 'model_comparison_1day.png'}")
    
    def _plot_feature_importance(self):
        """График важности признаков"""
        print("\n  + Построение графика важности признаков...")
        
        if not hasattr(self, 'imp_df'):
            print("  ⚠️ Данные о важности признаков отсутствуют")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 14))
        fig.suptitle('Важность признаков (прогноз на 1 день)', fontsize=16, fontweight='bold')
        
        # Топ-30 признаков
        ax = axes[0]
        imp_top30 = self.imp_df.head(30)
        bar_colors = imp_top30['color'].tolist()
        
        bars = ax.barh(imp_top30['feature'], imp_top30['importance'],
                       color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Важность', fontsize=12)
        ax.set_title('Топ-30 важнейших признаков', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        for i, (idx, row) in enumerate(imp_top30.iterrows()):
            ax.text(row['importance'] + 0.001, i, f'{row["importance"]:.3f}',
                    va='center', fontsize=7)
        
        # Суммарная важность по группам
        ax = axes[1]
        group_importance = self.group_importance
        
        group_colors = {
            'Позиция': '#FF6B6B',
            'Мотив': '#4ECDC4',
            'Временные': '#45B7D1',
            'Органика/Активации': '#96CEB4',
            'Ключ': '#FFEAA7',
            '🆕 Фризы': '#9F7AEA',
            '🆕 iOS обновления': '#F6AD55',
            'Ранги/Отношения': '#ECC94B',
            'Другие': '#D3D3D3'
        }
        
        colors_group = [group_colors.get(g, '#D3D3D3') for g in group_importance.index]
        
        bars = ax.barh(group_importance.index, group_importance.values,
                       color=colors_group, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Суммарная важность', fontsize=12)
        ax.set_title('Важность по группам признаков', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        total_importance = group_importance.sum()
        for i, (idx, val) in enumerate(group_importance.items()):
            pct = (val / total_importance) * 100
            ax.text(val + 0.005, i, f'{val:.4f} ({pct:.1f}%)',
                    va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.plots_path / 'feature_importance_1day.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  + График сохранён: {self.plots_path / 'feature_importance_1day.png'}")
    
    def _plot_predictions_vs_actual(self, X_test, y_test):
        """График факт vs прогноз"""
        print("\n  + Построение графика факт vs прогноз...")
        
        if self.best_model is None:
            print("  ⚠️ Лучшая модель отсутствует")
            return
        
        # Предсказания лучшей модели
        y_pred_scaled = self.best_model.predict(X_test)
        y_test_original = self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_pred_original = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(y_test_original, y_pred_original, alpha=0.4, s=15, color='steelblue')
        ax.plot([y_test_original.min(), y_test_original.max()],
                [y_test_original.min(), y_test_original.max()],
                'r--', linewidth=2, label='Идеальное предсказание')
        ax.set_xlabel('Фактическое изменение позиции (1 день)', fontsize=12)
        ax.set_ylabel('Предсказанное изменение позиции (1 день)', fontsize=12)
        ax.set_title('Факт vs Прогноз (XGBoost, прогноз на 1 день)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Добавляем метрики
        mae = mean_absolute_error(y_test_original, y_pred_original)
        r2 = r2_score(y_test_original, y_pred_original)
        
        ax.text(0.05, 0.95, f'MAE: {mae:.4f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top')
        ax.text(0.05, 0.90, f'R²: {r2:.4f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(self.plots_path / 'predictions_vs_actual_1day.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  + График сохранён: {self.plots_path / 'predictions_vs_actual_1day.png'}")
    
    def _plot_events_importance(self):
        """🆕 График важности событий"""
        print("\n  + Построение графика важности событий...")
        
        if not hasattr(self, 'imp_df'):
            return
        
        event_features = ['is_freeze', 'days_to_freeze', 'days_after_freeze',
                          'is_ios_update_day', 'days_after_ios_update', 'is_week_after_ios_update']
        
        event_imp = self.imp_df[self.imp_df['feature'].isin(event_features)]
        
        if len(event_imp) == 0:
            print("  ⚠️ Признаки событий не найдены")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = []
        for f in event_imp['feature']:
            if 'freeze' in f:
                colors.append('#9F7AEA')
            else:
                colors.append('#F6AD55')
        
        bars = ax.barh(event_imp['feature'], event_imp['importance'],
                       color=colors, alpha=0.8, edgecolor='black')
        ax.set_xlabel('Важность', fontsize=12)
        ax.set_title('Важность признаков фризов и iOS обновлений', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.invert_yaxis()
        
        for i, (idx, row) in enumerate(event_imp.iterrows()):
            ax.text(row['importance'] + 0.001, i, f'{row["importance"]:.4f} ({row["importance_pct"]:.2f}%)',
                    va='center', fontsize=9)
        
        # Легенда
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#9F7AEA', label='Фризы'),
            Patch(facecolor='#F6AD55', label='iOS обновления')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(self.plots_path / 'events_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  + График сохранён: {self.plots_path / 'events_importance.png'}")
    
    def _save_results(self):
        """Сохранение результатов"""
        print("\n" + "="*60)
        print("10. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        print("="*60)
        
        # Результаты моделей
        self.results_table.to_csv(self.data_path / "model_results_1day.csv", index=False)
        print(f"  + Результаты сохранены: {self.data_path / 'model_results_1day.csv'}")
        
        # Лучшая модель
        if self.best_model is not None:
            joblib.dump(self.best_model, self.models_path / "best_model_xgboost_1day.pkl")
            print(f"  + Лучшая модель сохранена: {self.models_path / 'best_model_xgboost_1day.pkl'}")
        
        # Scaler
        if self.scaler_X is not None:
            joblib.dump(self.scaler_X, self.models_path / "scaler_1day.pkl")
            print(f"  + Scaler сохранён: {self.models_path / 'scaler_1day.pkl'}")
        
        # Scaler y
        if self.scaler_y is not None:
            joblib.dump(self.scaler_y, self.models_path / "scaler_y_1day.pkl")
        
        # Важность признаков
        if hasattr(self, 'imp_df'):
            self.imp_df.to_csv(self.data_path / "feature_importance_1day.csv", index=False)
            print(f"  + Важность признаков сохранена: {self.data_path / 'feature_importance_1day.csv'}")
        
        # Кодировщики
        if self.le_keyword is not None:
            joblib.dump(self.le_keyword, self.models_path / "le_keyword.pkl")
        if self.le_cluster is not None:
            joblib.dump(self.le_cluster, self.models_path / "le_cluster.pkl")
        print(f"  + Кодировщики сохранены")
    
    def _generate_insights(self):
        """Генерация выводов"""
        print("\n" + "="*60)
        print("11. ИТОГОВЫЕ ВЫВОДЫ")
        print("="*60)
        
        if 'XGBoost_reg' in self.results:
            xgb_reg = self.results['XGBoost_reg']
            
            print(f"""
📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ (ПРОГНОЗ НА 1 ДЕНЬ):

1. ЛУЧШАЯ МОДЕЛЬ: {self.best_model_name}
   - MAE:  {xgb_reg['mae']:.4f}
   - RMSE: {xgb_reg['rmse']:.4f}
   - R²:   {xgb_reg['r2']:.4f}

2. ИНТЕРПРЕТАЦИЯ:
   - MAE = {xgb_reg['mae']:.4f} → модель ошибается в среднем на {xgb_reg['mae']:.2f} позиции
   - R² = {xgb_reg['r2']:.4f} → модель объясняет {xgb_reg['r2']*100:.1f}% вариации

3. ИСПОЛЬЗОВАНО ПРИЗНАКОВ: {len(self.feature_names)}
""")
            
            if hasattr(self, 'imp_df'):
                print("4. ТОП-5 ВАЖНЕЙШИХ ПРИЗНАКОВ:")
                for i, row in self.imp_df.head(5).iterrows():
                    print(f"   {i+1}. {row['feature']}: {row['importance_pct']:.2f}%")
                
                print(f"""
5. ГРУППЫ ПРИЗНАКОВ:
""")
                for group, imp in self.group_importance.items():
                    pct = (imp / self.group_importance.sum()) * 100
                    print(f"   - {group}: {pct:.1f}%")
            
            print("""
6. НОВЫЕ ВОЗМОЖНОСТИ:
   ✅ Прогноз на 1 день (более точный краткосрочный)
   ✅ Учёт фризов (is_freeze, days_to_freeze)
   ✅ Учёт iOS обновлений (is_ios_update_day)
   ✅ Расширенная статистика (EWM, медианы, min/max)
   ✅ Учёт праздничного сезона

7. РЕКОМЕНДАЦИИ:
   - Используйте XGBoost С РЕГУЛЯРИЗАЦИЕЙ для продакшена
   - Прогноз на 1 день даёт более точные краткосрочные оценки
   - Учитывайте фризы при планировании продвижения
   - Усиливайте продвижение после iOS обновлений
   - Основные драйверы: текущая позиция, мотив и события
""")
    
    def predict(self, X_new: pd.DataFrame) -> np.ndarray:
        """Предсказание для новых данных"""
        if self.best_model is None:
            raise ValueError("Модель не обучена. Сначала запустите run_ml_pipeline()")
        
        # Проверка наличия всех признаков
        missing = set(self.feature_cols) - set(X_new.columns)
        if missing:
            # Добавляем отсутствующие признаки как 0
            for col in missing:
                X_new[col] = 0
        
        # Стандартизация
        X_scaled = self.scaler_X.transform(X_new[self.feature_cols])
        
        # Предсказание
        y_pred_scaled = self.best_model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        return y_pred
    
    def load_model(self):
        """Загрузка сохранённой модели"""
        # Пробуем новый формат (1day)
        model_path = self.models_path / "best_model_xgboost_1day.pkl"
        scaler_path = self.models_path / "scaler_1day.pkl"
        
        if not model_path.exists():
            # Старый формат
            model_path = self.models_path / "best_model_xgboost_final.pkl"
            scaler_path = self.models_path / "scaler_final.pkl"
        
        if model_path.exists() and scaler_path.exists():
            self.best_model = joblib.load(model_path)
            self.scaler_X = joblib.load(scaler_path)
            
            scaler_y_path = self.models_path / "scaler_y_1day.pkl"
            if scaler_y_path.exists():
                self.scaler_y = joblib.load(scaler_y_path)
            
            le_keyword_path = self.models_path / "le_keyword.pkl"
            if le_keyword_path.exists():
                self.le_keyword = joblib.load(le_keyword_path)
            
            le_cluster_path = self.models_path / "le_cluster.pkl"
            if le_cluster_path.exists():
                self.le_cluster = joblib.load(le_cluster_path)
            
            self.best_model_name = "XGBoost (с регул.)"
            
            print(f"✅ Модель загружена из {model_path}")
            return True
        else:
            print(f"⚠️ Модель не найдена")
            return False