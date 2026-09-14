"""
Модуль загрузки и предобработки данных
Шаг 1: Загрузка Excel, парсинг, дедупликация, создание признаков
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Класс для загрузки и предобработки данных из Excel"""
    
    def __init__(self, data_path: Path):
        """
        Args:
            data_path: Путь к папке с данными
        """
        self.data_path = Path(data_path)
        self.df = None
        self.files = {
            2024: self.data_path / "Sound Amplifier_2024.xlsx",
            2025: self.data_path / "Sound Amplifier_2025.xlsx",
            2026: self.data_path / "Sound Amplifier_2026.xlsx"
        }
    
    def parse_excel_file(self, file_path: Path, year: int) -> pd.DataFrame:
        """
        Парсинг Excel файла с учётом структуры:
        - Строка 0: даты
        - Строка 1: заголовки (ALL)
        - Строка 2: ДАННЫЕ US (органика и активации) - ОБЩИЕ ДЛЯ ВСЕХ КЛЮЧЕЙ
        - Строка 3: заголовки данных (мотив/позиция)
        - Строка 4+: ключевые слова в колонке C (индекс 2) + мотив и позиция
        """
        print(f"\nПарсинг файла: {file_path.name}")
        df_raw = pd.read_excel(file_path, header=None)
        
        # =====================================================
        # 1. ИЗВЛЕЧЕНИЕ ДАТ ИЗ СТРОКИ 0
        # =====================================================
        date_cols = {}
        for col_idx, val in df_raw.iloc[0, :].items():
            if isinstance(val, (pd.Timestamp, datetime)):
                date_cols[col_idx] = val
            elif isinstance(val, str) and val.startswith('202'):
                try:
                    date_cols[col_idx] = pd.to_datetime(val)
                except:
                    pass
        
        print(f"  - Найдено дат: {len(date_cols)}")
        
        # =====================================================
        # 2. ПАРСИНГ ДАННЫХ US ИЗ СТРОКИ 2
        #    ОРГАНИКА И АКТИВАЦИИ ОБЩИЕ ДЛЯ ВСЕХ КЛЮЧЕЙ В ДЕНЬ
        # =====================================================
        us_organic_cols = {}
        us_activation_cols = {}
        us_row = 2  # ← СТРОКА 2: ДАННЫЕ US
        
        if us_row < len(df_raw):
            # Проходим по всем колонкам с датами
            for col_idx, date in date_cols.items():
                if col_idx < len(df_raw.columns):
                    val = df_raw.iloc[us_row, col_idx]
                    try:
                        # Проверяем, есть ли значение
                        if not pd.isna(val) and str(val).strip() not in ['', '–', '-']:
                            num_val = float(val)
                            
                            # Проверяем соседнюю колонку (следующую в паре)
                            next_col = col_idx + 1
                            if next_col < len(df_raw.columns):
                                next_val = df_raw.iloc[us_row, next_col]
                                next_has_data = not pd.isna(next_val) and str(next_val).strip() not in ['', '–', '-']
                                
                                # Если в следующей колонке есть данные - значит текущая это АКТИВАЦИЯ
                                # а следующая - ОРГАНИКА
                                if next_has_data:
                                    try:
                                        next_num_val = float(next_val)
                                        us_activation_cols[date] = num_val
                                        us_organic_cols[date_cols[next_col]] = next_num_val
                                    except:
                                        us_organic_cols[date] = num_val
                                else:
                                    # Если в следующей нет данных - значит это ОРГАНИКА
                                    us_organic_cols[date] = num_val
                            else:
                                us_organic_cols[date] = num_val
                    except:
                        pass
        
        print(f"  - Найдено органики US: {len(us_organic_cols)}")
        print(f"  - Найдено активаций US: {len(us_activation_cols)}")
        
        # =====================================================
        # 3. ПАРСИНГ КЛЮЧЕВЫХ СЛОВ (НАЧИНАЮТСЯ С СТРОКИ 4)
        # =====================================================
        keyword_start_row = 4  # ← СТРОКА 4: НАЧАЛО КЛЮЧЕВЫХ СЛОВ
        keyword_end_row = len(df_raw)
        
        for idx in range(keyword_start_row, min(keyword_start_row + 500, len(df_raw))):
            val = df_raw.iloc[idx, 2]
            if pd.isna(val) or str(val).strip() == '':
                keyword_end_row = idx
                break
            if isinstance(val, str):
                val_lower = val.lower()
                if 'лимит' in val_lower or 'итого' in val_lower or 'total' in val_lower:
                    keyword_end_row = idx
                    break
        
        print(f"  - Диапазон ключевых слов: {keyword_start_row} - {keyword_end_row}")
        
        # =====================================================
        # 4. СБОР ДАННЫХ ДЛЯ КАЖДОГО КЛЮЧЕВОГО СЛОВА И ДАТЫ
        # =====================================================
        keywords_data = []
        
        for row_idx in range(keyword_start_row, keyword_end_row):
            keyword = df_raw.iloc[row_idx, 2]
            if pd.isna(keyword) or str(keyword).strip() == '':
                continue
            keyword_str = str(keyword).strip()
            if len(keyword_str) < 2:
                continue
            
            for col_idx, date in date_cols.items():
                if col_idx + 1 >= len(df_raw.columns):
                    continue
                
                motiv_val = df_raw.iloc[row_idx, col_idx]
                position_val = df_raw.iloc[row_idx, col_idx + 1]
                
                has_motiv = not pd.isna(motiv_val) and str(motiv_val).strip() not in ['', '–', '-']
                has_position = not pd.isna(position_val) and str(position_val).strip() not in ['', '–', '-']
                
                if not has_motiv and not has_position:
                    continue
                
                try:
                    motiv = float(motiv_val) if has_motiv else 0
                except:
                    motiv = 0
                try:
                    position = float(position_val) if has_position else -1
                except:
                    position = -1
                
                # Получаем органику и активации для этой даты
                organic = us_organic_cols.get(date, 0)
                activation = us_activation_cols.get(date, 0)
                
                keywords_data.append({
                    'date': date,
                    'keyword': keyword_str,
                    'motiv': motiv,
                    'position': position,
                    'year': year,
                    'organic_us': organic,
                    'activation_us': activation
                })
        
        df_result = pd.DataFrame(keywords_data)
        print(f"  - Загружено {len(df_result):,} записей")
        print(f"  - Уникальных ключей: {df_result['keyword'].nunique()}")
        
        if len(df_result) > 0:
            print(f"  - Период: {df_result['date'].min()} - {df_result['date'].max()}")
            # Проверяем наличие органики и активаций
            organic_sum = df_result['organic_us'].sum()
            activation_sum = df_result['activation_us'].sum()
            if organic_sum > 0 or activation_sum > 0:
                print(f"  - Суммарная органика US: {organic_sum:.0f}")
                print(f"  - Суммарные активации US: {activation_sum:.0f}")
        
        return df_result
    
    def load_and_preprocess(self) -> pd.DataFrame:
        """Загрузка всех файлов, объединение, дедупликация и создание признаков"""
        print("\n" + "="*80)
        print("ШАГ 1: ЗАГРУЗКА, ПРЕДОБРАБОТКА И ДЕДУПЛИКАЦИЯ ДАННЫХ")
        print("="*80)
        
        all_dfs = []
        for year, file_path in self.files.items():
            if file_path.exists():
                df_year = self.parse_excel_file(file_path, year)
                if len(df_year) > 0:
                    all_dfs.append(df_year)
            else:
                print(f"\n⚠️ Файл {file_path} не найден")
        
        if not all_dfs:
            raise ValueError("Не найдено ни одного файла с данными!")
        
        # Объединение
        df = pd.concat(all_dfs, ignore_index=True)
        
        print("\n" + "="*60)
        print("ДО ДЕДУПЛИКАЦИИ:")
        print(f"  - Всего записей: {len(df):,}")
        print(f"  - Уникальных ключей: {df['keyword'].nunique()}")
        print(f"  - Период: {df['date'].min()} - {df['date'].max()}")
        print("="*60)
        
        # =====================================================
        # ДЕДУПЛИКАЦИЯ
        # =====================================================
        print("\n" + "="*60)
        print("ДЕДУПЛИКАЦИЯ ДАННЫХ")
        print("="*60)
        
        # Проверка пересечений по годам
        print("\nПроверка пересечений по периодам:")
        for i, (year1, file1) in enumerate(self.files.items()):
            for j, (year2, file2) in enumerate(self.files.items()):
                if i < j and i < len(all_dfs) and j < len(all_dfs):
                    df1 = all_dfs[i]
                    df2 = all_dfs[j]
                    if len(df1) > 0 and len(df2) > 0:
                        overlap = set(df1['date'].dt.date).intersection(set(df2['date'].dt.date))
                        if overlap:
                            print(f"  {year1} и {year2}: пересечение {len(overlap)} дней")
        
        print(f"\nДо дедупликации: {len(df):,} записей")
        
        # Считаем дубликаты
        duplicates = df.duplicated(subset=['date', 'keyword'], keep=False).sum()
        print(f"Найдено дублирующихся записей: {duplicates:,}")
        
        # Оставляем первую запись (из более раннего года)
        df = df.drop_duplicates(subset=['date', 'keyword'], keep='first')
        print(f"✅ Удалены дубликаты (оставлена первая запись)")
        
        # Проверяем результат
        print(f"После дедупликации: {len(df):,} записей")
        unique_check = df.duplicated(subset=['date', 'keyword']).sum()
        print(f"Осталось дубликатов: {unique_check}")
        
        # Сортировка
        df = df.sort_values(['keyword', 'date']).reset_index(drop=True)
        
        print("\n" + "="*60)
        print("ИТОГО ПОСЛЕ ДЕДУПЛИКАЦИИ:")
        print(f"  - Всего записей: {len(df):,}")
        print(f"  - Уникальных ключей: {df['keyword'].nunique()}")
        print(f"  - Период: {df['date'].min()} - {df['date'].max()}")
        print("="*60)
        
        # =====================================================
        # ПРЕДОБРАБОТКА
        # =====================================================
        print("\nОБРАБОТКА ПРОПУСКОВ:")
        df['motiv'] = df['motiv'].fillna(0)
        df['organic_us'] = df['organic_us'].fillna(0)
        df['activation_us'] = df['activation_us'].fillna(0)
        df['position'] = df['position'].fillna(-1)
        print("  + Все пропуски обработаны")
        
        # Создание признаков
        print("\nСОЗДАНИЕ ПРИЗНАКОВ:")
        
        # Временные признаки
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        print("  + Временные признаки добавлены")
        
        # Лаговые признаки
        for lag in [1, 2, 3, 7, 14]:
            df[f'motiv_lag_{lag}'] = df.groupby('keyword')['motiv'].shift(lag).fillna(0)
            df[f'position_lag_{lag}'] = df.groupby('keyword')['position'].shift(lag).fillna(-1)
        print("  + Лаговые признаки добавлены")
        
        # Скользящие средние
        for window in [3, 7, 14]:
            df[f'motiv_ma_{window}'] = df.groupby('keyword')['motiv'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()).fillna(0)
            df[f'position_ma_{window}'] = df.groupby('keyword')['position'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()).fillna(-1)
            df[f'motiv_std_{window}'] = df.groupby('keyword')['motiv'].transform(
                lambda x: x.rolling(window, min_periods=1).std()).fillna(0)
        print("  + Скользящие средние добавлены")
        
        # Изменения
        df['motiv_change'] = df.groupby('keyword')['motiv'].diff().fillna(0)
        df['position_change'] = df.groupby('keyword')['position'].diff().fillna(0)
        df['motiv_pct_change'] = df.groupby('keyword')['motiv'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        df['position_pct_change'] = df.groupby('keyword')['position'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        print("  + Изменения добавлены")
        
        # Сценарии изменения мотива
        def classify_motiv_scenario(row):
            change = row.get('motiv_pct_change', 0)
            if change >= 0.3:
                return 'sharp_growth'
            elif change <= -0.3:
                return 'sharp_decline'
            elif change < 0:
                return 'gentle_decline'
            elif change > 0:
                return 'gentle_growth'
            else:
                return 'stable'
        
        df['motiv_scenario'] = df.apply(classify_motiv_scenario, axis=1)
        print("  + Сценарии изменения мотива добавлены")
        
        # Целевые позиции
        df['target_40'] = (df['position'] <= 40).astype(int)
        df['target_20'] = (df['position'] <= 20).astype(int)
        df['target_10'] = (df['position'] <= 10).astype(int)
        
        # Изменение позиции через N дней
        for h in [3, 7, 14]:
            df[f'position_delta_{h}d'] = df.groupby('keyword')['position'].shift(-h) - df['position']
            df[f'position_delta_{h}d'] = df[f'position_delta_{h}d'].fillna(0)
        print("  + Метрики роста добавлены")
        
        self.df = df
        
        # Сохранение
        self.save_data()
        
        print("\n" + "="*80)
        print("✅ ШАГ 1 ЗАВЕРШЕН УСПЕШНО!")
        print("="*80)
        
        return df
    
    def save_data(self):
        """Сохранение данных в CSV и Pickle"""
        if self.df is not None:
            self.df.to_csv(self.data_path / "clean_data.csv", index=False)
            self.df.to_pickle(self.data_path / "clean_data.pkl")
            print(f"\n✅ Данные сохранены:")
            print(f"  - CSV: {self.data_path / 'clean_data.csv'}")
            print(f"  - Pickle: {self.data_path / 'clean_data.pkl'}")
    
    def load_clean_data(self) -> pd.DataFrame:
        """Загрузка сохранённых данных"""
        pkl_path = self.data_path / "clean_data.pkl"
        if pkl_path.exists():
            self.df = pd.read_pickle(pkl_path)
            print(f"✅ Данные загружены из {pkl_path}")
            return self.df
        else:
            print(f"⚠️ Файл {pkl_path} не найден")
            return None