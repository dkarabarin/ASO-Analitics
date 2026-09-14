"""
Модуль расчёта требуемого мотива для достижения целевой позиции
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')


class MotivCalculator:
    """Класс для расчёта требуемого мотива"""
    
    def __init__(self, data_path: Path, df: pd.DataFrame = None):
        self.data_path = Path(data_path)
        
        if df is not None:
            self.df = df
        else:
            clean_path = self.data_path / "clean_data.pkl"
            if clean_path.exists():
                self.df = pd.read_pickle(clean_path)
            else:
                self.df = None
        
        self.motiv_efficiency = None
        # Загружаем готовую эффективность если есть
        eff_path = self.data_path / "motiv_efficiency.csv"
        if eff_path.exists():
            self.motiv_efficiency = pd.read_csv(eff_path)
    
    def calculate_efficiency(self) -> pd.DataFrame:
        """Рассчитывает эффективность мотива для всех ключей"""
        if self.df is None:
            return pd.DataFrame()
        
        results = []
        for keyword in self.df['keyword'].unique():
            result = self._calculate_keyword_efficiency(keyword)
            if result:
                results.append(result)
        
        self.motiv_efficiency = pd.DataFrame(results)
        
        if len(self.motiv_efficiency) > 0:
            self.motiv_efficiency.to_csv(self.data_path / "motiv_efficiency.csv", index=False)
        
        return self.motiv_efficiency
    
    def _calculate_keyword_efficiency(self, keyword: str) -> Optional[Dict]:
        """Расчёт эффективности для одного ключа"""
        kw_data = self.df[self.df['keyword'] == keyword].sort_values('date')
        
        if len(kw_data) < 10:
            return None
        
        with_motiv = kw_data[kw_data['motiv'] > 0]
        without_motiv = kw_data[kw_data['motiv'] == 0]
        
        if len(with_motiv) < 5:
            return None
        
        avg_pos_with = with_motiv['position'].mean()
        avg_pos_without = without_motiv['position'].mean() if len(without_motiv) > 0 else avg_pos_with
        avg_motiv = with_motiv['motiv'].mean()
        
        if avg_motiv > 0 and avg_pos_without != -1:
            efficiency = (avg_pos_without - avg_pos_with) / avg_motiv
        else:
            efficiency = 0
        
        best_pos = kw_data[kw_data['position'] != -1]['position'].min()
        best_motiv_rows = kw_data[kw_data['position'] == best_pos]
        best_motiv = best_motiv_rows['motiv'].mean() if len(best_motiv_rows) > 0 else 0
        
        valid = kw_data[(kw_data['position'] != -1) & (kw_data['motiv'] > 0)]
        corr = valid[['motiv', 'position']].corr().iloc[0, 1] if len(valid) > 5 else 0
        
        return {
            'keyword': keyword,
            'avg_pos_with_motiv': round(avg_pos_with, 2),
            'avg_pos_without_motiv': round(avg_pos_without, 2),
            'avg_motiv': round(avg_motiv, 2),
            'max_motiv': float(with_motiv['motiv'].max()),
            'efficiency': round(efficiency, 4),
            'best_position': float(best_pos),
            'best_motiv': round(best_motiv, 2),
            'correlation': round(corr, 4),
            'n_with_motiv': len(with_motiv),
            'n_without_motiv': len(without_motiv),
            'n_total': len(kw_data)
        }
    
    def calculate_required_motiv(self, keyword: str, target_position: int) -> Dict:
        """Рассчитывает требуемый мотив для достижения целевой позиции"""
        if self.df is None:
            return {'error': 'Данные не загружены'}
        
        if self.motiv_efficiency is None or len(self.motiv_efficiency) == 0:
            self.calculate_efficiency()
        
        kw_data = self.df[self.df['keyword'] == keyword]
        
        if len(kw_data) == 0:
            return {'error': f'Ключ "{keyword}" не найден'}
        
        valid_pos = kw_data[kw_data['position'] != -1]
        current_pos = valid_pos['position'].iloc[-1] if len(valid_pos) > 0 else -1
        current_motiv = kw_data['motiv'].iloc[-1] if len(kw_data) > 0 else 0
        
        eff_row = self.motiv_efficiency[self.motiv_efficiency['keyword'] == keyword] if self.motiv_efficiency is not None else pd.DataFrame()
        
        if len(eff_row) > 0:
            eff = eff_row.iloc[0]
            efficiency = eff['efficiency']
            correlation = eff['correlation']
        else:
            efficiency = 1.0
            correlation = 0
        
        position_diff = current_pos - target_position
        
        if position_diff <= 0:
            return {
                'keyword': keyword,
                'current_position': round(current_pos, 1),
                'target_position': target_position,
                'position_diff': 0,
                'current_motiv': round(current_motiv, 2),
                'required_motiv': 0,
                'required_motiv_int': 0,
                'efficiency': round(efficiency, 3),
                'correlation': round(correlation, 3),
                'predicted_position_1d': round(current_pos, 1),
                'predicted_position_7d': round(current_pos, 1),
                'confidence': 'high',
                'n_records': len(kw_data),
                'message': f'✅ Уже на целевой позиции ({current_pos:.1f} ≤ {target_position})',
                'recommendation': 'Поддерживать текущий мотив'
            }
        
        if efficiency > 0.1:
            required_motiv = position_diff / efficiency
        else:
            required_motiv = position_diff * 0.5
        
        required_motiv = min(max(required_motiv, 1), 30)
        
        if efficiency > 0.1:
            predicted_1d = current_pos - (required_motiv * efficiency)
            predicted_7d = current_pos - (required_motiv * efficiency * 7 * 0.3)
        else:
            predicted_1d = current_pos - required_motiv * 0.5
            predicted_7d = current_pos - required_motiv * 0.5 * 7 * 0.3
        
        if len(kw_data) > 100 and efficiency > 0.5 and abs(correlation) > 0.3:
            confidence = 'high'
        elif len(kw_data) > 50 and efficiency > 0.2:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        if required_motiv <= 2:
            recommendation = 'Небольшое увеличение мотива (по правилу R5)'
        elif required_motiv <= 5:
            recommendation = 'Умеренное увеличение мотива (по правилу R9)'
        elif required_motiv <= 10:
            recommendation = 'Значительное увеличение мотива (по правилу R8)'
        else:
            recommendation = 'Агрессивное продвижение (по правилу R12)'
        
        return {
            'keyword': keyword,
            'current_position': round(current_pos, 1),
            'target_position': target_position,
            'position_diff': round(position_diff, 1),
            'current_motiv': round(current_motiv, 2),
            'required_motiv': round(required_motiv, 2),
            'required_motiv_int': int(np.ceil(required_motiv)),
            'efficiency': round(efficiency, 3),
            'correlation': round(correlation, 3),
            'predicted_position_1d': round(predicted_1d, 1),
            'predicted_position_7d': round(predicted_7d, 1),
            'confidence': confidence,
            'n_records': len(kw_data),
            'message': f'Для достижения позиции {target_position} требуется мотив ≈ {int(np.ceil(required_motiv))}',
            'recommendation': recommendation
        }
    
    def get_keywords_list(self) -> List[str]:
        """Получить список всех ключевых слов"""
        if self.df is None:
            return []
        return sorted(self.df['keyword'].unique().tolist())
    
    def get_keyword_info(self, keyword: str) -> Dict:
        """Получить информацию о ключевом слове"""
        if self.df is None:
            return {'error': 'Данные не загружены'}
        
        kw_data = self.df[self.df['keyword'] == keyword]
        
        if len(kw_data) == 0:
            return {'error': f'Ключ "{keyword}" не найден'}
        
        valid_pos = kw_data[kw_data['position'] != -1]
        
        return {
            'keyword': keyword,
            'n_records': len(kw_data),
            'current_position': float(valid_pos['position'].iloc[-1]) if len(valid_pos) > 0 else -1,
            'current_motiv': float(kw_data['motiv'].iloc[-1]) if len(kw_data) > 0 else 0,
            'avg_position': float(valid_pos['position'].mean()) if len(valid_pos) > 0 else -1,
            'avg_motiv': float(kw_data['motiv'].mean()),
            'min_position': float(valid_pos['position'].min()) if len(valid_pos) > 0 else -1,
            'max_position': float(valid_pos['position'].max()) if len(valid_pos) > 0 else -1,
            'max_motiv': float(kw_data['motiv'].max()),
            'organic_avg': float(kw_data['organic_us'].mean()),
            'activation_avg': float(kw_data['activation_us'].mean()),
            'period_start': kw_data['date'].min().strftime('%Y-%m-%d'),
            'period_end': kw_data['date'].max().strftime('%Y-%m-%d')
        }