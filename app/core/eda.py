"""
Модуль разведочного анализа данных (EDA) v2.0
Шаг 2: Статистика, визуализации, корреляции
Включает: график фризов и iOS обновлений, прогноз на 1 день
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class EDAAnalyzer:
    """Класс для разведочного анализа данных (v2.0)"""
    
    def __init__(self, data_path: Path, df: pd.DataFrame = None):
        """
        Args:
            data_path: Путь к папке с данными
            df: DataFrame с данными (если None, загружается из clean_data.pkl)
        """
        self.data_path = Path(data_path)
        self.plots_path = self.data_path / "plots"
        self.plots_path.mkdir(exist_ok=True)
        
        if df is not None:
            self.df = df
        else:
            pkl_path = self.data_path / "clean_data.pkl"
            if pkl_path.exists():
                self.df = pd.read_pickle(pkl_path)
            else:
                self.df = None
        
        self.results = {}
        self.top20_keywords = []
        self.df_top20_unique = None
        self.corr_matrix = None
    
    def run_eda(self) -> dict:
        """Запуск полного EDA"""
        try:
            print("\n" + "="*80)
            print("ШАГ 2: РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ (EDA)")
            print("="*80)
            
            if self.df is None:
                raise ValueError("Данные не загружены")
            
            df = self.df
            
            print(f"\nРазмер данных: {len(df):,} записей")
            print(f"Уникальных ключей: {df['keyword'].nunique()}")
            print(f"Всего признаков: {len(df.columns)}")
            print(f"Период: {df['date'].min()} - {df['date'].max()}")
            
            # 1. Общая статистика
            self._general_statistics(df)
            
            # 2. Распределение мотива и позиций
            self._distribution_analysis(df)
            
            # 3. Топ-20 ключевых слов
            self._top_keywords_analysis(df)
            
            # 4. Корреляционный анализ
            self._correlation_analysis(df)
            
            # 5. Сценарии изменения мотива
            self._scenario_analysis(df)
            
            # 6. Генерация графиков
            self._generate_plots(df)
            
            # 7. Выводы
            self._generate_insights(df)
            
            print("\n" + "="*80)
            print("✅ ШАГ 2 ЗАВЕРШЕН УСПЕШНО!")
            print("="*80)
            
            return self.results
            
        except Exception as e:
            print(f"❌ Ошибка в EDA: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _general_statistics(self, df: pd.DataFrame):
        """Общая статистика"""
        print("\n" + "="*60)
        print("1. ОБЩАЯ СТАТИСТИКА")
        print("="*60)
        
        print("\nСтатистика по числовым колонкам:")
        print(df[['motiv', 'position', 'organic_us', 'activation_us']].describe())
        
        print("\nУникальные значения по ключевым колонкам:")
        print(f"  - Количество ключевых слов: {df['keyword'].nunique()}")
        print(f"  - Количество дат: {df['date'].nunique()}")
        print(f"  - Количество лет: {df['year'].nunique()}")
        print(f"  - Всего признаков: {len(df.columns)}")
        
        # Проверка новых признаков
        new_features = ['is_freeze', 'is_ios_update_day', 'position_delta_1d']
        has_new = all(f in df.columns for f in new_features)
        
        if has_new:
            print(f"  - ✅ Данные содержат признаки фризов и iOS обновлений")
            print(f"  - Записей в фризах: {df['is_freeze'].sum():,}")
            print(f"  - Дней с iOS: {df['is_ios_update_day'].sum()}")
        else:
            print(f"  - ⚠️ Данные НЕ содержат новые признаки (фризы, iOS, прогноз 1д)")
        
        self.results['general_stats'] = {
            'total_records': int(len(df)),
            'unique_keywords': int(df['keyword'].nunique()),
            'unique_dates': int(df['date'].nunique()),
            'years': [int(y) for y in df['year'].unique().tolist()],
            'total_features': int(len(df.columns)),
            'period_start': df['date'].min().strftime('%Y-%m-%d'),
            'period_end': df['date'].max().strftime('%Y-%m-%d'),
            'has_events': bool(has_new)
        }
    
    def _distribution_analysis(self, df: pd.DataFrame):
        """Распределение мотива и позиций"""
        print("\n" + "="*60)
        print("2. РАСПРЕДЕЛЕНИЕ МОТИВА И ПОЗИЦИЙ")
        print("="*60)
        
        # Мотив
        print("\nСтатистика по мотиву:")
        motiv_pos = df[df['motiv'] > 0]['motiv']
        print(f"  - Записей с мотивом > 0: {len(motiv_pos):,} ({len(motiv_pos)/len(df)*100:.1f}%)")
        print(f"  - Записей без мотива: {(df['motiv'] == 0).sum():,} ({(df['motiv'] == 0).sum()/len(df)*100:.1f}%)")
        print(f"  - Средний мотив (все записи): {df['motiv'].mean():.2f}")
        print(f"  - Средний мотив (только > 0): {motiv_pos.mean():.2f}")
        print(f"  - Медианный мотив (только > 0): {motiv_pos.median():.2f}")
        print(f"  - Максимальный мотив: {df['motiv'].max():.0f}")
        
        # Позиции
        print("\nСтатистика по позициям:")
        valid_pos_mask = df['position'] != -1
        position_valid_df = df[valid_pos_mask]
        position_values = position_valid_df['position']
        
        print(f"  - Записей с позицией: {len(position_valid_df):,} ({len(position_valid_df)/len(df)*100:.1f}%)")
        print(f"  - Записей без позиции: {(df['position'] == -1).sum():,} ({(df['position'] == -1).sum()/len(df)*100:.1f}%)")
        if len(position_values) > 0:
            print(f"  - Средняя позиция: {position_values.mean():.2f}")
            print(f"  - Медианная позиция: {position_values.median():.2f}")
            print(f"  - Минимальная позиция: {position_values.min():.0f}")
            print(f"  - Максимальная позиция: {position_values.max():.0f}")
        
        self.results['distribution'] = {
            'motiv_positive_count': int(len(motiv_pos)),
            'motiv_positive_pct': float(len(motiv_pos)/len(df)*100),
            'motiv_mean_all': float(df['motiv'].mean()),
            'motiv_mean_positive': float(motiv_pos.mean()) if len(motiv_pos) > 0 else 0,
            'motiv_median_positive': float(motiv_pos.median()) if len(motiv_pos) > 0 else 0,
            'position_mean': float(position_values.mean()) if len(position_values) > 0 else 0,
            'position_median': float(position_values.median()) if len(position_values) > 0 else 0
        }
    
    def _top_keywords_analysis(self, df: pd.DataFrame):
        """Анализ топ-20 ключевых слов"""
        print("\n" + "="*60)
        print("3. ТОП-20 КЛЮЧЕВЫХ СЛОВ")
        print("="*60)
        
        top20 = df['keyword'].value_counts().head(20)
        print("\nТоп-20 по количеству записей:")
        for i, (keyword, count) in enumerate(top20.items(), 1):
            print(f"  {i:2d}. {keyword:35s} - {count:5d} записей")
        
        # Статистика по топ-20
        top20_keywords = top20.index.tolist()
        df_top20 = df[df['keyword'].isin(top20_keywords)]
        
        # Определяем колонки для агрегации
        agg_dict = {
            'motiv': 'mean',
            'position': 'first',
            'organic_us': 'mean',
            'activation_us': 'mean',
            'motiv_scenario': 'first'
        }
        
        # Добавляем position_delta_1d, если есть
        if 'position_delta_1d' in df_top20.columns:
            agg_dict['position_delta_1d'] = 'first'
        
        df_top20_unique = df_top20.groupby(['keyword', 'date']).agg(agg_dict).reset_index()
        
        print("\nСтатистика по топ-20 ключам:")
        stats_top20 = df_top20_unique.groupby('keyword').agg({
            'motiv': ['mean', 'max', 'sum'],
            'position': ['mean', 'min', 'max']
        }).round(2)
        stats_top20.columns = ['motiv_mean', 'motiv_max', 'motiv_sum', 
                               'position_mean', 'position_min', 'position_max']
        stats_top20 = stats_top20.sort_values('motiv_sum', ascending=False)
        
        print("\n  {:<30s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
            'Ключ', 'Мотив_сред', 'Мотив_макс', 'Мотив_сумма', 
            'Позиция_сред', 'Позиция_мин', 'Позиция_макс'
        ))
        print("  " + "-"*100)
        for keyword, row in stats_top20.head(20).iterrows():
            print("  {:<30s} {:>12.2f} {:>12.0f} {:>12.0f} {:>12.2f} {:>12.0f} {:>12.0f}".format(
                keyword[:30], 
                row['motiv_mean'], 
                row['motiv_max'], 
                row['motiv_sum'],
                row['position_mean'],
                row['position_min'],
                row['position_max']
            ))
        
        self.results['top20'] = {
            'keywords': top20.to_dict(),
            'stats': stats_top20.head(20).to_dict()
        }
        
        # Сохраняем для дальнейшего использования
        self.top20_keywords = top20_keywords
        self.df_top20_unique = df_top20_unique
    
    def _correlation_analysis(self, df: pd.DataFrame):
        """Корреляционный анализ"""
        print("\n" + "="*60)
        print("4. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
        print("="*60)
        
        if self.df_top20_unique is None:
            print("  ⚠️ Нет данных для корреляционного анализа")
            return
        
        df_top20_unique = self.df_top20_unique
        
        # Корреляция мотива и позиции
        print("\nКорреляция мотива и позиции:")
        try:
            keyword_corr = df_top20_unique.groupby('keyword').apply(
                lambda x: x[['motiv', 'position']].corr().iloc[0, 1] if len(x) > 1 else np.nan
            ).dropna()
            
            if len(keyword_corr) > 0:
                print(f"  - Средняя корреляция: {keyword_corr.mean():.3f}")
                print(f"  - Медианная корреляция: {keyword_corr.median():.3f}")
                print(f"  - Максимальная корреляция: {keyword_corr.max():.3f}")
                print(f"  - Минимальная корреляция: {keyword_corr.min():.3f}")
        except Exception as e:
            print(f"  ⚠️ Ошибка при расчёте корреляции мотива и позиции: {e}")
        
        # Корреляция между ключевыми словами
        print("\nКорреляция между ключевыми словами (на основе позиций):")
        try:
            pivot_position = df_top20_unique.pivot_table(
                index='date', 
                columns='keyword', 
                values='position'
            ).dropna(axis=1, how='all')
            
            if pivot_position.shape[1] > 1:
                corr_matrix = pivot_position.corr()
                self.corr_matrix = corr_matrix
                
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                corr_values = upper_tri.stack()
                corr_values = corr_values[~np.isnan(corr_values)]
                
                if len(corr_values) > 0:
                    print(f"  - Средняя корреляция между ключами: {corr_values.mean():.3f}")
                    print(f"  - Максимальная корреляция: {corr_values.max():.3f}")
                    print(f"  - Минимальная корреляция: {corr_values.min():.3f}")
                    
                    print("\n  Топ-5 пар ключей с максимальной корреляцией:")
                    corr_pairs = corr_values.sort_values(ascending=False).head(5)
                    for (k1, k2), corr in corr_pairs.items():
                        print(f"    - {k1[:20]} ↔ {k2[:20]}: {corr:.3f}")
        except Exception as e:
            print(f"  ⚠️ Ошибка при расчёте корреляции между ключами: {e}")
    
    def _scenario_analysis(self, df: pd.DataFrame):
        """Анализ сценариев изменения мотива"""
        print("\n" + "="*60)
        print("5. СЦЕНАРИИ ИЗМЕНЕНИЯ МОТИВА")
        print("="*60)
        
        scenario_names = {
            'stable': 'стабильный',
            'sharp_growth': 'резкий рост',
            'sharp_decline': 'резкий спад',
            'gentle_growth': 'плавный рост',
            'gentle_decline': 'плавный спад'
        }
        
        if 'motiv_scenario' in df.columns:
            scenario_counts = df['motiv_scenario'].value_counts()
            print("\nРаспределение сценариев:")
            for scenario, count in scenario_counts.items():
                name = scenario_names.get(scenario, scenario)
                print(f"  - {name}: {count:,} записей ({count/len(df)*100:.1f}%)")
        
        # Эффективность сценариев (прогноз на 1 день)
        target_col = 'position_delta_1d'
        
        if target_col in df.columns and 'motiv_scenario' in df.columns:
            print(f"\nЭффективность сценариев ({target_col}):")
            df_scenario_effect = df.dropna(subset=[target_col, 'motiv_scenario'])
            
            if len(df_scenario_effect) > 0:
                scenario_effect = df_scenario_effect.groupby('motiv_scenario').agg({
                    target_col: ['mean', 'std', 'count']
                }).round(3)
                scenario_effect.columns = ['mean_delta', 'std_delta', 'count']
                
                print("\n  {:<20s} {:>15s} {:>15s} {:>10s}".format(
                    'Сценарий', 'Среднее Δ', 'Стд. откл.', 'Кол-во'
                ))
                print("  " + "-"*65)
                for scenario, row in scenario_effect.iterrows():
                    name = scenario_names.get(scenario, scenario)
                    print("  {:<20s} {:>15.3f} {:>15.3f} {:>10.0f}".format(
                        name, row['mean_delta'], row['std_delta'], row['count']
                    ))
    
    def _generate_plots(self, df: pd.DataFrame):
        """Генерация всех графиков"""
        print("\n" + "="*60)
        print("6. ГЕНЕРАЦИЯ ГРАФИКОВ")
        print("="*60)
        
        try:
            # 1. Общий обзор EDA
            self._plot_eda_overview(df)
            
            # 2. Тепловая карта корреляций
            self._plot_correlation_heatmap()
            
            # 3. Временные ряды топ-5
            self._plot_timeseries_top5(df)
            
            # 4. Ежемесячный анализ
            self._plot_monthly_analysis(df)
            
            # 5. Распределение мотива
            self._plot_motiv_distribution(df)
            
            # 6. Распределение позиций
            self._plot_position_distribution(df)
            
            # 7. Топ-20 ключей
            self._plot_top20_keywords(df)
            
            # 8. Мотив vs Позиция
            self._plot_motiv_vs_position(df)
            
            # 🆕 9. Фризы и iOS обновления
            if 'is_freeze' in df.columns:
                self._plot_freeze_and_updates(df)
            
            print("  + Все графики сгенерированы")
        except Exception as e:
            print(f"  ⚠️ Ошибка при генерации графиков: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_eda_overview(self, df: pd.DataFrame):
        """График обзора EDA"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Анализ мотива и позиций', fontsize=16, fontweight='bold')
            
            # 1. Распределение мотива
            ax = axes[0, 0]
            motiv_positive = df[df['motiv'] > 0]['motiv']
            if len(motiv_positive) > 0:
                motiv_positive.hist(bins=30, edgecolor='black', alpha=0.7, color='steelblue', ax=ax)
                ax.axvline(motiv_positive.mean(), color='red', linestyle='--', 
                          label=f'Средний: {motiv_positive.mean():.2f}')
                ax.legend()
            ax.set_title('Распределение мотива (только > 0)')
            ax.set_xlabel('Мотив')
            ax.set_ylabel('Частота')
            ax.grid(True, alpha=0.3)
            
            # 2. Распределение позиций
            ax = axes[0, 1]
            position_valid = df[df['position'] != -1]['position']
            if len(position_valid) > 0:
                position_valid.hist(bins=50, edgecolor='black', alpha=0.7, color='orange', ax=ax)
                ax.axvline(position_valid.mean(), color='red', linestyle='--', 
                          label=f'Средняя: {position_valid.mean():.2f}')
                ax.axvline(position_valid.median(), color='blue', linestyle='--', 
                          label=f'Медиана: {position_valid.median():.2f}')
                ax.legend()
            ax.set_title('Распределение позиций')
            ax.set_xlabel('Позиция')
            ax.set_ylabel('Частота')
            ax.grid(True, alpha=0.3)
            
            # 3. Топ-20 ключей
            ax = axes[1, 0]
            top20 = df['keyword'].value_counts().head(20)
            if len(top20) > 0:
                top20.plot(kind='barh', ax=ax, color='green')
            ax.set_title('Топ-20 ключевых слов по количеству записей')
            ax.set_xlabel('Количество записей')
            ax.set_ylabel('Ключевое слово')
            ax.grid(True, alpha=0.3)
            
            # 4. Мотив vs Позиция
            ax = axes[1, 1]
            stats_top20 = df.groupby('keyword').agg({
                'motiv': 'mean',
                'position': 'mean'
            }).dropna()
            stats_top20 = stats_top20[stats_top20['position'] != -1]
            if len(stats_top20) > 0:
                ax.scatter(stats_top20['motiv'], stats_top20['position'], 
                           alpha=0.6, s=80, color='purple')
                ax.set_title('Средний мотив vs Средняя позиция')
                ax.set_xlabel('Средний мотив')
                ax.set_ylabel('Средняя позиция')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plots_path / 'eda_overview.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  + График сохранен: eda_overview.png")
        except Exception as e:
            print(f"  ⚠️ Ошибка при создании eda_overview.png: {e}")
    
    def _plot_correlation_heatmap(self):
        """Тепловая карта корреляций"""
        try:
            if not hasattr(self, 'corr_matrix') or self.corr_matrix is None:
                print("  ⚠️ Нет данных для тепловой карты корреляций")
                return
            
            if self.corr_matrix.shape[0] <= 1:
                return
            
            fig, ax = plt.subplots(figsize=(14, 12))
            
            if self.corr_matrix.shape[0] > 15 and len(self.top20_keywords) > 0:
                top_keys = self.top20_keywords[:min(15, len(self.top20_keywords))]
                corr_subset = self.corr_matrix.loc[top_keys, top_keys]
            else:
                corr_subset = self.corr_matrix
            
            mask = np.triu(np.ones_like(corr_subset, dtype=bool))
            sns.heatmap(corr_subset, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
                        center=0, square=True, ax=ax, cbar_kws={'label': 'Корреляция'})
            ax.set_title('Корреляция позиций между ключевыми словами', fontsize=14)
            plt.tight_layout()
            plt.savefig(self.plots_path / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  + Тепловая карта сохранена: correlation_heatmap.png")
        except Exception as e:
            print(f"  ⚠️ Ошибка при создании correlation_heatmap.png: {e}")
    
    def _plot_timeseries_top5(self, df: pd.DataFrame):
        """Временные ряды топ-5 ключей"""
        try:
            top20 = df['keyword'].value_counts().head(20)
            top5 = top20.head(5).index.tolist()
            df_top5 = df[df['keyword'].isin(top5)]
            
            fig, axes = plt.subplots(2, 1, figsize=(16, 12))
            fig.suptitle('Динамика мотива и позиций (топ-5 ключей)', fontsize=16, fontweight='bold')
            
            # Мотив по времени
            ax = axes[0]
            for keyword in top5:
                keyword_data = df_top5[df_top5['keyword'] == keyword]
                ax.plot(keyword_data['date'], keyword_data['motiv'], label=keyword, alpha=0.7)
            ax.set_title('Динамика мотива (топ-5 ключей)')
            ax.set_xlabel('Дата')
            ax.set_ylabel('Мотив')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Позиции по времени
            ax = axes[1]
            for keyword in top5:
                keyword_data = df_top5[df_top5['keyword'] == keyword]
                keyword_data = keyword_data[keyword_data['position'] != -1]
                if len(keyword_data) > 0:
                    ax.plot(keyword_data['date'], keyword_data['position'], label=keyword, alpha=0.7)
            ax.set_title('Динамика позиций (топ-5 ключей)')
            ax.set_xlabel('Дата')
            ax.set_ylabel('Позиция')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.invert_yaxis()
            
            plt.tight_layout()
            plt.savefig(self.plots_path / 'timeseries_top5.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  + Временные ряды сохранены: timeseries_top5.png")
        except Exception as e:
            print(f"  ⚠️ Ошибка при создании timeseries_top5.png: {e}")
    
    def _plot_monthly_analysis(self, df: pd.DataFrame):
        """Ежемесячный анализ"""
        try:
            print("\n  + Построение ежемесячного анализа...")
            
            # Подготовка данных
            df_monthly = df.groupby(df['date'].dt.to_period('M')).agg({
                'motiv': ['mean', 'sum', 'count'],
                'position': 'mean'
            }).reset_index()
            df_monthly.columns = ['period', 'motiv_mean', 'motiv_sum', 'motiv_count', 'position_mean']
            df_monthly['period_str'] = df_monthly['period'].astype(str)
            
            # Доля дней с мотивом
            monthly_days_data = []
            for period, group in df.groupby(df['date'].dt.to_period('M')):
                total_days = group['date'].nunique()
                days_with_motiv = group[group['motiv'] > 0]['date'].nunique()
                monthly_days_data.append({
                    'period': period,
                    'total_days': total_days,
                    'days_with_motiv': days_with_motiv
                })
            
            monthly_days = pd.DataFrame(monthly_days_data)
            monthly_days['period_str'] = monthly_days['period'].astype(str)
            monthly_days['days_ratio'] = (monthly_days['days_with_motiv'] / monthly_days['total_days']) * 100
            
            # Активные ключи
            monthly_active_data = []
            for period, group in df[df['motiv'] > 0].groupby(df['date'].dt.to_period('M')):
                active_keywords = group['keyword'].nunique()
                monthly_active_data.append({
                    'period': period,
                    'active_keywords': active_keywords
                })
            
            monthly_active = pd.DataFrame(monthly_active_data)
            monthly_active['period_str'] = monthly_active['period'].astype(str)
            
            # Графики
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Ежемесячный анализ мотива и позиций', fontsize=16, fontweight='bold')
            
            # График 1: Средний мотив + Средняя позиция
            ax1 = axes[0, 0]
            ax2 = ax1.twinx()
            ax1.bar(df_monthly['period_str'], df_monthly['motiv_mean'], 
                    alpha=0.7, color='steelblue', label='Средний мотив')
            ax1.set_xlabel('Месяц')
            ax1.set_ylabel('Средний мотив', color='steelblue')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            ax2.plot(df_monthly['period_str'], df_monthly['position_mean'], 
                     color='red', marker='o', linewidth=2, label='Средняя позиция')
            ax2.set_ylabel('Средняя позиция', color='red')
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            ax1.set_title('Средний мотив и средняя позиция по месяцам')
            
            # График 2: Суммарный мотив
            ax = axes[0, 1]
            ax.bar(df_monthly['period_str'], df_monthly['motiv_sum'], 
                   alpha=0.7, color='green', label='Суммарный мотив')
            ax.set_xlabel('Месяц')
            ax.set_ylabel('Суммарный мотив')
            ax.set_title('Суммарный мотив по месяцам')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            if df_monthly['motiv_sum'].max() > 0:
                max_sum = df_monthly['motiv_sum'].max()
                for i, v in enumerate(df_monthly['motiv_sum']):
                    ax.text(i, v + max_sum * 0.02, f'{int(v)}', ha='center', va='bottom', fontsize=8)
            
            # График 3: Доля дней с мотивом
            ax = axes[1, 0]
            ax.bar(monthly_days['period_str'], monthly_days['days_ratio'], 
                   alpha=0.7, color='coral')
            ax.set_xlabel('Месяц')
            ax.set_ylabel('Доля дней с мотивом > 0 (%)')
            ax.set_title('Доля дней с мотивом > 0 по месяцам')
            ax.tick_params(axis='x', rotation=45)
            ax.axhline(y=30, color='red', linestyle='--', linewidth=2, label='30% порог')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 105)
            for i, v in enumerate(monthly_days['days_ratio']):
                ax.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
            
            # График 4: Активные ключи
            ax = axes[1, 1]
            ax.bar(monthly_active['period_str'], monthly_active['active_keywords'], 
                   alpha=0.7, color='purple')
            ax.set_xlabel('Месяц')
            ax.set_ylabel('Количество активных ключей')
            ax.set_title('Количество ключей с мотивом > 0 по месяцам')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            for i, v in enumerate(monthly_active['active_keywords']):
                ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(self.plots_path / 'monthly_analysis.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  + Ежемесячный анализ сохранен: monthly_analysis.png")
        except Exception as e:
            print(f"  ⚠️ Ошибка при создании monthly_analysis.png: {e}")
    
    def _plot_motiv_distribution(self, df: pd.DataFrame):
        """График распределения мотива"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            motiv_positive = df[df['motiv'] > 0]['motiv']
            if len(motiv_positive) > 0:
                motiv_positive.hist(bins=30, edgecolor='black', alpha=0.7, color='steelblue', ax=ax)
                ax.axvline(motiv_positive.mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Средний: {motiv_positive.mean():.2f}')
                ax.axvline(motiv_positive.median(), color='blue', linestyle='--', 
                          linewidth=2, label=f'Медиана: {motiv_positive.median():.2f}')
                ax.legend()
            ax.set_title('Распределение мотива (только > 0)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Мотив')
            ax.set_ylabel('Частота')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plots_path / 'motiv_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  + График сохранен: motiv_distribution.png")
        except Exception as e:
            print(f"  ⚠️ Ошибка при создании motiv_distribution.png: {e}")
    
    def _plot_position_distribution(self, df: pd.DataFrame):
        """График распределения позиций"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            position_valid = df[df['position'] != -1]['position']
            if len(position_valid) > 0:
                position_valid.hist(bins=50, edgecolor='black', alpha=0.7, color='orange', ax=ax)
                ax.axvline(position_valid.mean(), color='red', linestyle='--', 
                          linewidth=2, label=f'Средняя: {position_valid.mean():.2f}')
                ax.axvline(position_valid.median(), color='blue', linestyle='--', 
                          linewidth=2, label=f'Медиана: {position_valid.median():.2f}')
                ax.legend()
            ax.set_title('Распределение позиций', fontsize=14, fontweight='bold')
            ax.set_xlabel('Позиция')
            ax.set_ylabel('Частота')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.plots_path / 'position_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  + График сохранен: position_distribution.png")
        except Exception as e:
            print(f"  ⚠️ Ошибка при создании position_distribution.png: {e}")
    
    def _plot_top20_keywords(self, df: pd.DataFrame):
        """График топ-20 ключевых слов"""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            top20 = df['keyword'].value_counts().head(20)
            if len(top20) > 0:
                top20.plot(kind='barh', ax=ax, color='green', alpha=0.8)
                ax.invert_yaxis()
                for i, (keyword, count) in enumerate(top20.items()):
                    ax.text(count + 2, i, str(count), va='center', fontsize=9)
            ax.set_title('Топ-20 ключевых слов по количеству записей', fontsize=14, fontweight='bold')
            ax.set_xlabel('Количество записей')
            ax.set_ylabel('Ключевое слово')
            ax.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig(self.plots_path / 'top20_keywords.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  + График сохранен: top20_keywords.png")
        except Exception as e:
            print(f"  ⚠️ Ошибка при создании top20_keywords.png: {e}")
    
    def _plot_motiv_vs_position(self, df: pd.DataFrame):
        """График мотив vs позиция"""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            stats = df.groupby('keyword').agg({
                'motiv': 'mean',
                'position': 'mean'
            }).dropna()
            stats = stats[stats['position'] != -1]
            
            if len(stats) > 0:
                scatter = ax.scatter(stats['motiv'], stats['position'], 
                                    alpha=0.6, s=80, c=stats['position'], 
                                    cmap='viridis', edgecolors='black', linewidth=0.5)
                ax.set_xlabel('Средний мотив')
                ax.set_ylabel('Средняя позиция')
                ax.set_title('Средний мотив vs Средняя позиция по ключам', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax, label='Средняя позиция')
                
                top10 = df['keyword'].value_counts().head(10).index.tolist()
                for keyword in top10:
                    if keyword in stats.index:
                        ax.annotate(keyword[:20], 
                                   (stats.loc[keyword, 'motiv'], stats.loc[keyword, 'position']),
                                   fontsize=8, alpha=0.7, ha='right')
            
            plt.tight_layout()
            plt.savefig(self.plots_path / 'motiv_vs_position.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  + График сохранен: motiv_vs_position.png")
        except Exception as e:
            print(f"  ⚠️ Ошибка при создании motiv_vs_position.png: {e}")
    
    def _plot_freeze_and_updates(self, df: pd.DataFrame):
        """🆕 График фризов и iOS обновлений"""
        try:
            print("\n  + Построение графика фризов и iOS обновлений...")
            
            fig, axes = plt.subplots(3, 1, figsize=(18, 14))
            fig.suptitle('Влияние фризов и обновлений iOS на позиции', fontsize=16, fontweight='bold')
            
            # Агрегация по дням
            agg_dict = {
                'position': lambda x: x[x != -1].mean(),
                'motiv': 'mean',
                'is_freeze': 'max'
            }
            if 'is_ios_update_day' in df.columns:
                agg_dict['is_ios_update_day'] = 'max'
            
            df_daily = df.groupby('date').agg(agg_dict).reset_index()
            
            freeze_days = df_daily[df_daily['is_freeze'] == 1]['date']
            update_days = df_daily[df_daily['is_ios_update_day'] == 1]['date'] if 'is_ios_update_day' in df_daily.columns else pd.Series([])
            
            # График 1: Средняя позиция
            ax = axes[0]
            ax.plot(df_daily['date'], df_daily['position'],
                    color='steelblue', linewidth=1.5, label='Средняя позиция', alpha=0.7)
            
            for fd in freeze_days:
                ax.axvspan(fd, fd + pd.Timedelta(days=1), alpha=0.2, color='red', zorder=0)
            
            for ud in update_days:
                ax.axvline(ud, color='green', linestyle='--', alpha=0.6, linewidth=1.5)
            
            ax.set_title('Средняя позиция с отметками фризов (красный) и обновлений iOS (зелёный)')
            ax.set_xlabel('Дата')
            ax.set_ylabel('Средняя позиция')
            ax.invert_yaxis()
            ax.legend(['Средняя позиция', 'Фризы', 'iOS обновления'], loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # График 2: Средний мотив
            ax = axes[1]
            ax.plot(df_daily['date'], df_daily['motiv'],
                    color='purple', linewidth=1.5, label='Средний мотив', alpha=0.7)
            
            for fd in freeze_days:
                ax.axvspan(fd, fd + pd.Timedelta(days=1), alpha=0.2, color='red', zorder=0)
            
            for ud in update_days:
                ax.axvline(ud, color='green', linestyle='--', alpha=0.6, linewidth=1.5)
            
            ax.set_title('Средний мотив с отметками фризов и обновлений')
            ax.set_xlabel('Дата')
            ax.set_ylabel('Средний мотив')
            ax.legend(['Средний мотив', 'Фризы', 'iOS обновления'], loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # График 3: Количество активных ключей
            ax = axes[2]
            df_active = df[df['motiv'] > 0].groupby('date')['keyword'].nunique().reset_index()
            df_active.columns = ['date', 'active_keywords']
            
            ax.plot(df_active['date'], df_active['active_keywords'],
                    color='green', linewidth=1.5, label='Активных ключей', alpha=0.7)
            
            for fd in freeze_days:
                ax.axvspan(fd, fd + pd.Timedelta(days=1), alpha=0.2, color='red', zorder=0)
            
            for ud in update_days:
                ax.axvline(ud, color='green', linestyle='--', alpha=0.6, linewidth=1.5)
            
            ax.set_title('Количество активных ключей')
            ax.set_xlabel('Дата')
            ax.set_ylabel('Активных ключей')
            ax.legend(['Активные ключи', 'Фризы', 'iOS обновления'], loc='upper right')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plots_path / 'freeze_and_updates.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  + График фризов сохранён: freeze_and_updates.png")
        except Exception as e:
            print(f"  ⚠️ Ошибка при создании freeze_and_updates.png: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_insights(self, df: pd.DataFrame):
        """Генерация ключевых выводов"""
        print("\n" + "="*60)
        print("7. КЛЮЧЕВЫЕ ВЫВОДЫ EDA")
        print("="*60)
        
        top20 = df['keyword'].value_counts().head(20)
        position_valid = df[df['position'] != -1]['position']
        
        avg_position = position_valid.mean() if len(position_valid) > 0 else 0
        top_keyword = top20.index[0] if len(top20) > 0 else "N/A"
        top_keyword_count = top20.iloc[0] if len(top20) > 0 else 0
        top3_pct = (top20.iloc[:3].sum() / len(df) * 100) if len(top20) >= 3 else 0
        
        # Данные по событиям
        events_text = ""
        if 'is_freeze' in df.columns:
            events_text = f"""
4. СОБЫТИЯ:
   - Записей в фризах: {df['is_freeze'].sum():,} ({df['is_freeze'].sum()/len(df)*100:.1f}%)
   - Дней с iOS обновлениями: {df['is_ios_update_day'].sum() if 'is_ios_update_day' in df.columns else 0}
   - Записей в неделю после iOS: {df['is_week_after_ios_update'].sum() if 'is_week_after_ios_update' in df.columns else 0:,}
"""
        
        insights_text = f"""
1. ОБЩАЯ СТАТИСТИКА:
   - Всего записей: {len(df):,}
   - Уникальных ключей: {df['keyword'].nunique()}
   - Всего признаков: {len(df.columns)}
   - Период: {df['date'].min().strftime('%Y-%m-%d')} - {df['date'].max().strftime('%Y-%m-%d')}
   - Средний мотив: {df['motiv'].mean():.2f}
   - Средняя позиция: {avg_position:.2f}

2. ТОП-КЛЮЧИ:
   - Самый частый ключ: "{top_keyword}" ({top_keyword_count:,} записей)
   - Топ-3 ключа занимают {top3_pct:.1f}% всех записей

3. РЕКОМЕНДАЦИИ:
   - Для дальнейшего анализа выделить топ-20 ключей
   - Изучить корреляции между ключами
   - Проанализировать влияние сценариев на позиции
{events_text}
"""
        print(insights_text)
        
        self.results['insights'] = {
            'text': insights_text,
            'top_keyword': top_keyword,
            'top_keyword_count': int(top_keyword_count),
            'recommendations': [
                'Выделить топ-20 ключей для дальнейшего анализа',
                'Изучить корреляции между ключами',
                'Проанализировать влияние сценариев на позиции',
                'Учесть фризы и iOS обновления при планировании'
            ]
        }
    
    def get_plots_list(self) -> list:
        """Получить список всех доступных графиков"""
        plots = []
        for f in self.plots_path.glob("*.png"):
            plots.append({
                "name": f.name,
                "url": f"/plots/{f.name}",
                "size": f.stat().st_size,
                "modified": f.stat().st_mtime
            })
        return plots