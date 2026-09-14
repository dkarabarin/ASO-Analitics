"""
Модуль генерации отчётов v2.1
Шаг 6/7: Сбор результатов, генерация JSON с событиями (фризы, iOS)
ИСПРАВЛЕНО: рекомендации как строки, правильный подсчёт событий
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ReportGenerator:
    """Класс для генерации итоговых отчётов v2.1"""
    
    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        self.df = None
        self.results = {}
        self.freezes = None
        self.ios_updates = None
        self.motiv_efficiency = None
        self.model_results = None
        self.importance = None
        self.hypotheses = None
        self.rules = None
    
    def generate_report(self) -> dict:
        """Генерация полного отчёта"""
        print("\n" + "="*80)
        print("ШАГ 6: ГЕНЕРАЦИЯ ИТОГОВОГО ОТЧЁТА")
        print("="*80)
        
        self._load_data()
        self._collect_results()
        report = self._build_report()
        self._save_report(report)
        self._print_summary(report)
        
        print("\n" + "="*80)
        print("✅ ШАГ 6 ЗАВЕРШЕН УСПЕШНО!")
        print("="*80)
        
        return report
    
    def _load_data(self):
        """Загрузка данных"""
        print("\n" + "="*60)
        print("1. ЗАГРУЗКА ДАННЫХ")
        print("="*60)
        
        # Основные данные — пробуем pickle, при ошибке CSV
        pkl_path = self.data_path / "clean_data.pkl"
        csv_path = self.data_path / "clean_data.csv"
        
        if pkl_path.exists():
            try:
                self.df = pd.read_pickle(pkl_path)
                print(f"  + Загружены данные (pickle): {len(self.df):,} записей")
            except Exception as e:
                print(f"  ⚠️ Ошибка pickle: {e}")
                if csv_path.exists():
                    self.df = pd.read_csv(csv_path, parse_dates=['date'])
                    print(f"  + Загружены данные (CSV): {len(self.df):,} записей")
        elif csv_path.exists():
            self.df = pd.read_csv(csv_path, parse_dates=['date'])
            print(f"  + Загружены данные (CSV): {len(self.df):,} записей")
        
        # Результаты моделей
        for name in ["model_results_1day.csv", "model_results_final.csv"]:
            path = self.data_path / name
            if path.exists():
                self.model_results = pd.read_csv(path)
                print(f"  + Загружены результаты моделей: {len(self.model_results)}")
                break
        
        # Важность признаков
        for name in ["feature_importance_1day.csv", "feature_importance_final.csv"]:
            path = self.data_path / name
            if path.exists():
                self.importance = pd.read_csv(path)
                print(f"  + Загружена важность признаков")
                break
        
        # Гипотезы
        for name in ["hypotheses_summary_1day.csv", "hypotheses_by_cluster_summary.csv"]:
            path = self.data_path / name
            if path.exists():
                self.hypotheses = pd.read_csv(path)
                print(f"  + Загружены гипотезы: {len(self.hypotheses)}")
                break
        
        # Правила
        for name in ["rules_summary_1day.csv", "rules_from_document_summary_rules.csv"]:
            path = self.data_path / name
            if path.exists():
                self.rules = pd.read_csv(path)
                print(f"  + Загружены правила: {len(self.rules)}")
                break
        
        # Эффективность мотива
        eff_path = self.data_path / "motiv_efficiency.csv"
        if eff_path.exists():
            self.motiv_efficiency = pd.read_csv(eff_path)
            print(f"  + Загружена эффективность мотива: {len(self.motiv_efficiency)}")
        
        # Справочники событий
        freeze_path = self.data_path / "freeze_periods.csv"
        if freeze_path.exists():
            self.freezes = pd.read_csv(freeze_path)
            print(f"  + Загружены периоды фризов: {len(self.freezes)}")
        
        ios_path = self.data_path / "ios_updates.csv"
        if ios_path.exists():
            self.ios_updates = pd.read_csv(ios_path)
            print(f"  + Загружены iOS обновления: {len(self.ios_updates)}")
    
    def _collect_results(self):
        """Сбор результатов"""
        print("\n" + "="*60)
        print("2. СБОР РЕЗУЛЬТАТОВ")
        print("="*60)
        
        df = self.df
        
        # Общая статистика
        self.results['general'] = {
            'total_records': int(len(df)),
            'unique_keywords': int(df['keyword'].nunique()),
            'total_features': int(len(df.columns)),
            'period_start': df['date'].min().strftime('%Y-%m-%d'),
            'period_end': df['date'].max().strftime('%Y-%m-%d'),
            'days': int((df['date'].max() - df['date'].min()).days)
        }
        
        # Мотив
        motiv_positive = df[df['motiv'] > 0]['motiv']
        self.results['motiv_stats'] = {
            'mean': float(df['motiv'].mean()),
            'median': float(df['motiv'].median()),
            'max': float(df['motiv'].max()),
            'positive_count': int(len(motiv_positive)),
            'positive_pct': float(len(motiv_positive) / len(df) * 100),
            'mean_positive': float(motiv_positive.mean()) if len(motiv_positive) > 0 else 0
        }
        
        # Позиция
        position_valid = df[df['position'] != -1]['position']
        self.results['position_stats'] = {
            'mean': float(position_valid.mean()) if len(position_valid) > 0 else 0,
            'median': float(position_valid.median()) if len(position_valid) > 0 else 0,
            'min': float(position_valid.min()) if len(position_valid) > 0 else 0,
            'max': float(position_valid.max()) if len(position_valid) > 0 else 0,
            'valid_count': int(len(position_valid)),
            'valid_pct': float(len(position_valid) / len(df) * 100)
        }
        
        # 🆕 СОБЫТИЯ — правильно считаем
        events = {}
        
        if 'is_freeze' in df.columns:
            events['n_freeze_records'] = int(df['is_freeze'].sum())
            events['freeze_pct'] = float(df['is_freeze'].sum() / len(df) * 100)
        else:
            events['n_freeze_records'] = 0
            events['freeze_pct'] = 0
        
        # ⚠️ Количество ПЕРИОДОВ фризов (из справочника, не записей!)
        events['n_freezes'] = len(self.freezes) if self.freezes is not None else 0
        
        # iOS обновления
        if 'is_ios_update_day' in df.columns:
            events['n_ios_update_days'] = int(df['is_ios_update_day'].sum())
        else:
            events['n_ios_update_days'] = 0
        
        if 'is_week_after_ios_update' in df.columns:
            events['n_week_after_ios'] = int(df['is_week_after_ios_update'].sum())
        else:
            events['n_week_after_ios'] = 0
        
        # ⚠️ Количество iOS ОБНОВЛЕНИЙ (из справочника!)
        events['n_ios_updates'] = len(self.ios_updates) if self.ios_updates is not None else 0
        
        self.results['events'] = events
        
        # Эффективность мотива
        if self.motiv_efficiency is not None and len(self.motiv_efficiency) > 0:
            eff = self.motiv_efficiency['efficiency'].dropna()
            eff = eff[(eff > -10) & (eff < 50)]
            self.results['motiv_efficiency'] = {
                'n_keywords': int(len(eff)),
                'mean': float(eff.mean()),
                'median': float(eff.median()),
                'max': float(eff.max()),
                'min': float(eff.min()),
                'positive_count': int((eff > 0).sum()),
                'positive_pct': float((eff > 0).sum() / len(eff) * 100) if len(eff) > 0 else 0
            }
        
        # Модели
        if self.model_results is not None and len(self.model_results) > 0:
            self.results['model_results'] = self.model_results.to_dict('records')
            best_idx = self.model_results['MAE'].idxmin()
            self.results['best_model'] = self.model_results.loc[best_idx].to_dict()
        
        # Важность признаков
        if self.importance is not None and len(self.importance) > 0:
            self.results['feature_importance'] = self.importance.head(15).to_dict('records')
        
        # Гипотезы
        if self.hypotheses is not None and len(self.hypotheses) > 0:
            self.results['hypotheses'] = self.hypotheses.to_dict('records')
        
        # Правила
        if self.rules is not None and len(self.rules) > 0:
            self.results['rules'] = self.rules.to_dict('records')
        
        print(f"  + События: {events['n_freezes']} периодов фризов, {events['n_ios_updates']} iOS обновлений")
        print(f"  + Записей в фризах: {events['n_freeze_records']:,}")
        print(f"  + Записей в неделю после iOS: {events['n_week_after_ios']:,}")
        print("  + Все результаты собраны")
    
    def _build_report(self) -> dict:
        """Формирование отчёта"""
        print("\n" + "="*60)
        print("3. ФОРМИРОВАНИЕ ОТЧЁТА")
        print("="*60)
        
        report = {
            'report_metadata': {
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'version': '2.1',
                'app': 'Sound Amplifier ASO Analytics',
                'target_metric': 'position_delta_1d'
            },
            'executive_summary': self._build_executive_summary(),
            'data_summary': self._build_data_summary(),
            'motiv_efficiency_summary': self.results.get('motiv_efficiency', {}),
            'events_summary': self.results.get('events', {}),
            'ml_results': self._build_ml_summary(),
            'hypothesis_results': self._build_hypothesis_summary(),
            'rules_results': self._build_rules_summary(),
            'recommendations': self._build_recommendations(),
            'events_details': self._build_events_details(),
            'detailed_results': self.results
        }
        
        print("  + Отчёт сформирован")
        return report
    
    def _build_executive_summary(self) -> dict:
        """Краткое резюме"""
        df = self.df
        valid_pos = df[df['position'] != -1]
        
        best_model = self.results.get('best_model', {})
        events = self.results.get('events', {})
        
        return {
            'total_keywords': int(df['keyword'].nunique()),
            'total_records': int(len(df)),
            'total_features': int(len(df.columns)),
            'period': f"{df['date'].min().strftime('%Y-%m-%d')} - {df['date'].max().strftime('%Y-%m-%d')}",
            'avg_position': float(valid_pos['position'].mean()) if len(valid_pos) > 0 else 0,
            'avg_motiv': float(df['motiv'].mean()),
            'best_model': best_model.get('Model', 'N/A'),
            'best_model_mae': best_model.get('MAE', 'N/A'),
            'best_model_r2': best_model.get('R2', 'N/A'),
            'hypotheses_confirmed': self._count_confirmed_hypotheses(),
            'hypotheses_total': len(self.hypotheses) if self.hypotheses is not None else 0,
            'rules_confirmed': self._count_confirmed_rules(),
            'rules_total': len(self.rules) if self.rules is not None else 0,
            'avg_efficiency': self.results.get('motiv_efficiency', {}).get('mean', 0),
            'n_freezes': events.get('n_freezes', 0),
            'n_ios_updates': events.get('n_ios_updates', 0),
            'n_freeze_records': events.get('n_freeze_records', 0),
            'n_week_after_ios': events.get('n_week_after_ios', 0),
            'freeze_pct': events.get('freeze_pct', 0),
            'status': '✅ Анализ завершён успешно'
        }
    
    def _build_data_summary(self) -> dict:
        return {
            'general': self.results.get('general', {}),
            'motiv': self.results.get('motiv_stats', {}),
            'position': self.results.get('position_stats', {})
        }
    
    def _build_ml_summary(self) -> dict:
        if 'best_model' not in self.results:
            return {}
        
        best = self.results['best_model']
        
        return {
            'best_model': best.get('Model', 'N/A'),
            'mae': best.get('MAE', 'N/A'),
            'rmse': best.get('RMSE', 'N/A'),
            'r2': best.get('R2', 'N/A'),
            'top_features': self.results.get('feature_importance', [])[:5]
        }
    
    def _build_hypothesis_summary(self) -> dict:
        if 'hypotheses' not in self.results:
            return {}
        
        hypotheses = self.results['hypotheses']
        
        confirmed = 0
        total = 0
        for h in hypotheses:
            if 'Результат' in h:
                total += 1
                if '✅' in str(h['Результат']):
                    confirmed += 1
        
        return {
            'total_tests': total,
            'confirmed': confirmed,
            'confirmation_rate': f"{confirmed/total*100:.1f}%" if total > 0 else "0%",
            'details': hypotheses
        }
    
    def _build_rules_summary(self) -> dict:
        if 'rules' not in self.results:
            return {}
        
        rules = self.results['rules']
        
        confirmed = sum(1 for r in rules if '✅' in str(r.get('Результат', r.get('Статус', ''))))
        total = len(rules)
        
        return {
            'total_rules': total,
            'confirmed': confirmed,
            'confirmation_rate': f"{confirmed/total*100:.1f}%" if total > 0 else "0%",
            'details': rules
        }
    
    def _build_events_details(self) -> dict:
        """Детали событий"""
        events = {
            'freeze_periods': [],
            'ios_updates': []
        }
        
        if self.freezes is not None:
            for _, row in self.freezes.iterrows():
                df_period = self.df[
                    (self.df['date'] >= pd.to_datetime(row['start'])) & 
                    (self.df['date'] <= pd.to_datetime(row['end']))
                ]
                
                events['freeze_periods'].append({
                    'start': str(row['start']),
                    'end': str(row['end']),
                    'name': str(row['name']),
                    'n_records': int(len(df_period)),
                    'avg_motiv': float(df_period['motiv'].mean()) if len(df_period) > 0 else 0,
                    'avg_delta_1d': float(df_period['position_delta_1d'].mean()) 
                                    if len(df_period) > 0 and 'position_delta_1d' in df_period.columns else 0
                })
        
        if self.ios_updates is not None:
            for _, row in self.ios_updates.iterrows():
                df_period = self.df[
                    (self.df['date'] >= pd.to_datetime(row['date'])) & 
                    (self.df['date'] <= pd.to_datetime(row['date']) + pd.Timedelta(days=7))
                ]
                
                events['ios_updates'].append({
                    'date': str(row['date']),
                    'version': str(row['version']),
                    'n_records': int(len(df_period)),
                    'avg_delta_1d': float(df_period['position_delta_1d'].mean()) 
                                    if len(df_period) > 0 and 'position_delta_1d' in df_period.columns else 0
                })
        
        return events
    
    def _build_recommendations(self) -> dict:
        """⚠️ Рекомендации — простые строки (не объекты!)"""
        return {
            'critical': [
                '🎯 Используйте расчёт требуемого мотива для планирования продвижения',
                '📅 Прогноз на 1 день даёт наиболее точные краткосрочные оценки',
                '❄️ Ставьте паузу во время фризов — продвижение не работает',
                '📱 Усиливайте активность после iOS обновлений — пик активности пользователей'
            ],
            'important': [
                '📊 Эффективность мотива разная по ключам — учитывайте при планировании',
                '⚡ Для ключей с низкой эффективностью используйте агрессивный мотив',
                '🎯 Для ключей с высокой эффективностью достаточно умеренного мотива',
                '📋 Учитывайте правило R12 (мотив ≤ 30% от трафика)'
            ],
            'recommended': [
                '🔬 Проверяйте гипотезы на свежих данных еженедельно',
                '🔄 Переобучайте модель каждую неделю',
                '🤖 Используйте LLM-ассистента для генерации новых гипотез',
                '📅 Планируйте продвижение с учётом календаря фризов',
                '📝 Готовьте контент заранее для усиления после фризов'
            ]
        }
    
    def _count_confirmed_hypotheses(self) -> int:
        if 'hypotheses' not in self.results:
            return 0
        
        confirmed = 0
        for h in self.results['hypotheses']:
            if 'Результат' in h and '✅' in str(h['Результат']):
                confirmed += 1
        
        return confirmed
    
    def _count_confirmed_rules(self) -> int:
        if 'rules' not in self.results:
            return 0
        
        confirmed = 0
        for r in self.results['rules']:
            status = r.get('Результат', r.get('Статус', ''))
            if '✅' in str(status):
                confirmed += 1
        
        return confirmed
    
    def _save_report(self, report: dict):
        """Сохранение"""
        print("\n" + "="*60)
        print("4. СОХРАНЕНИЕ ОТЧЁТА")
        print("="*60)
        
        report_path = self.data_path / "final_report_1day.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"  + Итоговый отчёт: {report_path}")
        
        summary_data = [
            {'Показатель': 'Всего записей', 'Значение': report['executive_summary'].get('total_records', 'N/A')},
            {'Показатель': 'Уникальных ключей', 'Значение': report['executive_summary'].get('total_keywords', 'N/A')},
            {'Показатель': 'Всего признаков', 'Значение': report['executive_summary'].get('total_features', 'N/A')},
            {'Показатель': 'Средняя позиция', 'Значение': round(report['executive_summary'].get('avg_position', 0), 2)},
            {'Показатель': 'Средний мотив', 'Значение': round(report['executive_summary'].get('avg_motiv', 0), 2)},
            {'Показатель': 'Ср. эффективность мотива', 'Значение': round(report['executive_summary'].get('avg_efficiency', 0), 3)},
            {'Показатель': 'Лучшая модель', 'Значение': report['executive_summary'].get('best_model', 'N/A')},
            {'Показатель': 'MAE модели', 'Значение': report['executive_summary'].get('best_model_mae', 'N/A')},
            {'Показатель': 'R² модели', 'Значение': report['executive_summary'].get('best_model_r2', 'N/A')},
            {'Показатель': 'Подтверждено гипотез', 'Значение': report['executive_summary'].get('hypotheses_confirmed', 0)},
            {'Показатель': 'Подтверждено правил', 'Значение': report['executive_summary'].get('rules_confirmed', 0)},
            {'Показатель': 'Периодов фризов', 'Значение': report['executive_summary'].get('n_freezes', 0)},
            {'Показатель': 'iOS обновлений', 'Значение': report['executive_summary'].get('n_ios_updates', 0)},
        ]
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.data_path / "report_summary_1day.csv", index=False)
        print(f"  + Краткий отчёт: {self.data_path / 'report_summary_1day.csv'}")
    
    def _print_summary(self, report: dict):
        """Вывод"""
        print("\n" + "="*60)
        print("5. КРАТКИЙ ОТЧЁТ")
        print("="*60)
        
        es = report.get('executive_summary', {})
        
        print(f"""
📊 ИТОГОВЫЙ ОТЧЁТ ПО ASO АНАЛИЗУ (v2.1)

ОБЩАЯ ИНФОРМАЦИЯ:
  - Всего записей: {es.get('total_records', 'N/A'):,}
  - Уникальных ключей: {es.get('total_keywords', 'N/A')}
  - Всего признаков: {es.get('total_features', 'N/A')}
  - Период: {es.get('period', 'N/A')}
  - Средняя позиция: {es.get('avg_position', 0):.2f}
  - Средний мотив: {es.get('avg_motiv', 0):.2f}

ЭФФЕКТИВНОСТЬ МОТИВА:
  - Средняя: {es.get('avg_efficiency', 0):.3f}

СОБЫТИЯ:
  - Периодов фризов: {es.get('n_freezes', 0)}
  - iOS обновлений: {es.get('n_ios_updates', 0)}
  - Записей в фризах: {es.get('n_freeze_records', 0):,}
  - Записей в неделю после iOS: {es.get('n_week_after_ios', 0):,}

МОДЕЛИРОВАНИЕ:
  - Лучшая модель: {es.get('best_model', 'N/A')}
  - MAE: {es.get('best_model_mae', 'N/A')}
  - R²: {es.get('best_model_r2', 'N/A')}

ГИПОТЕЗЫ:
  - Подтверждено: {es.get('hypotheses_confirmed', 0)}/{es.get('hypotheses_total', 0)}

ПРАВИЛА:
  - Подтверждено: {es.get('rules_confirmed', 0)}/{es.get('rules_total', 0)}

СТАТУС: {es.get('status', 'N/A')}
""")
    
    def get_report(self) -> dict:
        """Получить сохранённый отчёт"""
        report_path = self.data_path / "final_report_1day.json"
        if report_path.exists():
            with open(report_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None