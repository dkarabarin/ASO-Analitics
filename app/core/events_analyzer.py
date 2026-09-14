"""
Модуль анализа фризов и iOS обновлений
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


FREEZE_PERIODS = [
    ('2025-01-04', '2025-01-08', 'Новогодний фриз 2025'),
    ('2025-02-25', '2025-03-10', 'Февральский фриз'),
    ('2025-03-15', '2025-03-15', 'Короткий фриз'),
    ('2025-05-05', '2025-05-12', 'Майский фриз'),
    ('2025-08-15', '2025-08-30', 'Августовский фриз'),
    ('2025-09-10', '2025-09-16', 'Сентябрьский фриз'),
    ('2025-09-28', '2025-10-31', 'Осенний фриз'),
    ('2026-01-06', '2026-01-19', 'Новогодний фриз 2026'),
    ('2026-04-07', '2026-04-21', 'Апрельский фриз'),
    ('2026-06-29', '2026-07-18', 'Летний фриз'),
]

IOS_UPDATES = [
    ('2025-03-31', '18.4'),
    ('2025-05-12', '18.5'),
    ('2025-07-29', '18.6'),
    ('2025-09-15', '18.7/26.0'),
    ('2025-11-03', '26.1'),
    ('2025-12-12', '26.2'),
    ('2026-02-11', '26.3'),
    ('2026-03-24', '26.4'),
    ('2026-03-31', '26.4.1'),
    ('2026-05-11', '26.5'),
    ('2026-06-01', '26.5.1'),
    ('2026-06-29', '26.5.2'),
    ('2026-07-27', '26.6'),
    ('2026-08-17', '26.6.1'),
]


class EventsAnalyzer:
    """Анализатор фризов и обновлений iOS"""
    
    def __init__(self, data_path: Path, df: pd.DataFrame = None):
        self.data_path = Path(data_path)
        
        if df is not None:
            self.df = df
        else:
            pkl_path = self.data_path / "clean_data.pkl"
            if pkl_path.exists():
                self.df = pd.read_pickle(pkl_path)
            else:
                self.df = None
    
    def get_freeze_periods(self) -> List[Dict]:
        return [
            {
                'start': start,
                'end': end,
                'name': name,
                'days': (pd.to_datetime(end) - pd.to_datetime(start)).days + 1
            }
            for start, end, name in FREEZE_PERIODS
        ]
    
    def get_ios_updates(self) -> List[Dict]:
        return [{'date': date, 'version': version} for date, version in IOS_UPDATES]
    
    def analyze_freeze_period(self, start_date: str, end_date: str) -> Dict:
        if self.df is None:
            return {'error': 'Данные не загружены'}
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        df_period = self.df[(self.df['date'] >= start) & (self.df['date'] <= end)]
        df_before = self.df[(self.df['date'] >= start - pd.Timedelta(days=7)) & (self.df['date'] < start)]
        df_after = self.df[(self.df['date'] > end) & (self.df['date'] <= end + pd.Timedelta(days=7))]
        
        def get_stats(data):
            if len(data) == 0:
                return None
            valid_pos = data[data['position'] != -1]
            return {
                'n_records': len(data),
                'avg_motiv': float(data['motiv'].mean()),
                'avg_position': float(valid_pos['position'].mean()) if len(valid_pos) > 0 else -1,
                'avg_delta_1d': float(data['position_delta_1d'].mean()) if 'position_delta_1d' in data.columns else 0,
                'active_keywords': int(data[data['motiv'] > 0]['keyword'].nunique())
            }
        
        return {
            'period': f"{start_date} - {end_date}",
            'during_freeze': get_stats(df_period),
            'before_freeze': get_stats(df_before),
            'after_freeze': get_stats(df_after)
        }
    
    def analyze_ios_update(self, update_date: str, version: str) -> Dict:
        if self.df is None:
            return {'error': 'Данные не загружены'}
        
        update_dt = pd.to_datetime(update_date)
        
        df_before = self.df[(self.df['date'] >= update_dt - pd.Timedelta(days=7)) & (self.df['date'] < update_dt)]
        df_week1 = self.df[(self.df['date'] >= update_dt) & (self.df['date'] <= update_dt + pd.Timedelta(days=7))]
        df_week2 = self.df[(self.df['date'] > update_dt + pd.Timedelta(days=7)) & (self.df['date'] <= update_dt + pd.Timedelta(days=14))]
        
        def get_stats(data):
            if len(data) == 0:
                return None
            valid_pos = data[data['position'] != -1]
            return {
                'n_records': len(data),
                'avg_motiv': float(data['motiv'].mean()),
                'avg_position': float(valid_pos['position'].mean()) if len(valid_pos) > 0 else -1,
                'avg_delta_1d': float(data['position_delta_1d'].mean()) if 'position_delta_1d' in data.columns else 0
            }
        
        return {
            'date': update_date,
            'version': version,
            'before': get_stats(df_before),
            'week_1_after': get_stats(df_week1),
            'week_2_after': get_stats(df_week2)
        }
    
    def get_events_timeline(self) -> List[Dict]:
        events = []
        
        for start, end, name in FREEZE_PERIODS:
            events.append({
                'type': 'freeze',
                'start': start,
                'end': end,
                'name': name,
                'icon': '❄️'
            })
        
        for date, version in IOS_UPDATES:
            events.append({
                'type': 'ios_update',
                'date': date,
                'version': version,
                'name': f'iOS {version}',
                'icon': '📱'
            })
        
        events.sort(key=lambda x: x.get('start', x.get('date')))
        return events
    
    def get_next_freeze(self, from_date=None) -> Optional[Dict]:
        if from_date is None:
            from_date = datetime.now()
        elif isinstance(from_date, str):
            from_date = pd.to_datetime(from_date)
        
        future_freezes = [
            (start, end, name) for start, end, name in FREEZE_PERIODS
            if pd.to_datetime(start) > from_date
        ]
        
        if not future_freezes:
            return None
        
        next_freeze = min(future_freezes, key=lambda x: pd.to_datetime(x[0]))
        start = pd.to_datetime(next_freeze[0])
        
        return {
            'start': next_freeze[0],
            'end': next_freeze[1],
            'name': next_freeze[2],
            'days_until': (start - from_date).days
        }
    
    def predict_freeze_impact(self, keyword: str, target_position: int) -> Dict:
        if self.df is None:
            return {'error': 'Данные не загружены'}
        
        kw_data = self.df[self.df['keyword'] == keyword]
        
        if len(kw_data) == 0:
            return {'error': f'Ключ "{keyword}" не найден'}
        
        if 'is_freeze' not in kw_data.columns:
            return {'error': 'Признак is_freeze не найден'}
        
        kw_in_freeze = kw_data[kw_data['is_freeze'] == 1]
        kw_out_freeze = kw_data[kw_data['is_freeze'] == 0]
        
        return {
            'keyword': keyword,
            'target_position': target_position,
            'n_freeze_records': len(kw_in_freeze),
            'avg_motiv_in_freeze': float(kw_in_freeze['motiv'].mean()) if len(kw_in_freeze) > 0 else 0,
            'avg_delta_in_freeze': float(kw_in_freeze['position_delta_1d'].mean()) if len(kw_in_freeze) > 0 and 'position_delta_1d' in kw_in_freeze.columns else 0,
            'avg_delta_out_freeze': float(kw_out_freeze['position_delta_1d'].mean()) if len(kw_out_freeze) > 0 and 'position_delta_1d' in kw_out_freeze.columns else 0,
            'recommendation': self._get_freeze_recommendation(kw_in_freeze, kw_out_freeze)
        }
    
    def _get_freeze_recommendation(self, kw_in_freeze, kw_out_freeze) -> str:
        if len(kw_in_freeze) < 5:
            return 'Недостаточно данных о поведении ключа во время фризов'
        
        delta_in = kw_in_freeze['position_delta_1d'].mean() if 'position_delta_1d' in kw_in_freeze.columns else 0
        delta_out = kw_out_freeze['position_delta_1d'].mean() if len(kw_out_freeze) > 0 and 'position_delta_1d' in kw_out_freeze.columns else 0
        
        if delta_in >= 0 and delta_out < 0:
            return '⚠️ Во время фризов продвижение не работает — сделайте паузу'
        elif abs(delta_in) < 0.5 and abs(delta_out) > 1:
            return '⏸️ Во время фризов эффект мотива минимален — приостановите продвижение'
        else:
            return '✅ Ключ стабилен во время фризов — можно продолжать продвижение'