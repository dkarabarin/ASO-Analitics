"""
Модуль бизнес-кластеризации
Шаг 3: Группировка по позициям и объёму мотива, якорные слова
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ClusterAnalyzer:
    """Класс для бизнес-кластеризации (не K-Means)"""
    
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
            self.df = pd.read_pickle(self.data_path / "clean_data.pkl")
        
        self.keyword_stats = None
        self.df_with_clusters = None
        self.anchor_df = None
        self.cluster_stats = None
    
    def run_clustering(self) -> pd.DataFrame:
        """Запуск кластеризации"""
        print("\n" + "="*80)
        print("ШАГ 3: КЛАСТЕРИЗАЦИЯ ПО ПОЗИЦИЯМ И ОБЪЁМУ МОТИВА")
        print("="*80)
        
        df = self.df
        
        print(f"\nРазмер данных: {len(df):,} записей")
        print(f"Уникальных ключей: {df['keyword'].nunique()}")
        
        print("\nГруппировка ключевых слов по:")
        print("  - Позиции: топ-10, топ-50, выше 50")
        print("  - Объёму мотива: высокий, средний, низкий, нулевой")
        
        # 1. Агрегация данных по ключам
        self._aggregate_keywords(df)
        
        # 2. Категоризация по позиции
        self._categorize_by_position()
        
        # 3. Категоризация по объёму мотива
        self._categorize_by_motiv()
        
        # 4. Создание кластеров
        self._create_clusters()
        
        # 5. Якорные слова
        self._find_anchor_words()
        
        # 6. Визуализация
        self._plot_clusters()
        
        # 7. Динамика кластеров
        self._plot_cluster_timeline()
        
        # 8. Сохранение
        self._save_results()
        
        # 9. Выводы
        self._generate_insights()
        
        print("\n" + "="*80)
        print("✅ ШАГ 3 ЗАВЕРШЕН УСПЕШНО!")
        print("="*80)
        
        return self.df_with_clusters
    
    def _aggregate_keywords(self, df: pd.DataFrame):
        """Агрегация данных по ключам"""
        print("\n" + "="*60)
        print("1. АГРЕГАЦИЯ ДАННЫХ ПО КЛЮЧАМ")
        print("="*60)
        
        keyword_stats = df.groupby('keyword').agg({
            'position': ['mean', 'min', 'max', 'count'],
            'motiv': ['mean', 'sum', 'count']
        }).round(2)
        keyword_stats.columns = ['pos_mean', 'pos_min', 'pos_max', 'pos_count', 
                                 'motiv_mean', 'motiv_sum', 'motiv_count']
        keyword_stats = keyword_stats.reset_index()
        
        print(f"  - Всего ключей: {len(keyword_stats)}")
        print(f"  - Ключей с позицией: {(keyword_stats['pos_mean'] != -1).sum()}")
        print(f"  - Ключей с мотивом > 0: {(keyword_stats['motiv_sum'] > 0).sum()}")
        
        self.keyword_stats = keyword_stats
    
    def _categorize_by_position(self):
        """Категоризация по позиции"""
        print("\n" + "="*60)
        print("2. КАТЕГОРИЗАЦИЯ ПО ПОЗИЦИИ")
        print("="*60)
        
        def position_category(pos_mean):
            if pos_mean <= 10:
                return 'top10'
            elif pos_mean <= 50:
                return 'top50'
            else:
                return 'above50'
        
        self.keyword_stats['pos_category'] = self.keyword_stats['pos_mean'].apply(position_category)
        
        print("\nРаспределение по категориям позиции:")
        pos_dist = self.keyword_stats['pos_category'].value_counts().sort_index()
        for cat, count in pos_dist.items():
            print(f"  - {cat}: {count} ключей ({count/len(self.keyword_stats)*100:.1f}%)")
        
        self.pos_dist = pos_dist
    
    def _categorize_by_motiv(self):
        """Категоризация по объёму мотива"""
        print("\n" + "="*60)
        print("3. КАТЕГОРИЗАЦИЯ ПО ОБЪЁМУ МОТИВА")
        print("="*60)
        
        def motiv_category(motiv_sum):
            if motiv_sum == 0:
                return 'zero'
            elif motiv_sum < 100:
                return 'low'
            elif motiv_sum < 1000:
                return 'medium'
            else:
                return 'high'
        
        self.keyword_stats['motiv_category'] = self.keyword_stats['motiv_sum'].apply(motiv_category)
        
        print("\nРаспределение по категориям мотива:")
        motiv_dist = self.keyword_stats['motiv_category'].value_counts().sort_index()
        for cat, count in motiv_dist.items():
            print(f"  - {cat}: {count} ключей ({count/len(self.keyword_stats)*100:.1f}%)")
        
        print("\nСтатистика по категориям мотива:")
        for cat in ['zero', 'low', 'medium', 'high']:
            subset = self.keyword_stats[self.keyword_stats['motiv_category'] == cat]
            if len(subset) > 0:
                print(f"  - {cat}: средний мотив={subset['motiv_mean'].mean():.2f}, "
                      f"суммарный мотив={subset['motiv_sum'].sum():.0f}")
        
        self.motiv_dist = motiv_dist
    
    def _create_clusters(self):
        """Создание кластеров"""
        print("\n" + "="*60)
        print("4. СОЗДАНИЕ КЛАСТЕРОВ")
        print("="*60)
        
        self.keyword_stats['cluster'] = (
            self.keyword_stats['pos_category'] + '_' + self.keyword_stats['motiv_category']
        )
        
        # Сохраняем кластеры
        df_clusters = self.keyword_stats[['keyword', 'cluster', 'pos_category', 'motiv_category']]
        self.df_with_clusters = self.df.merge(df_clusters, on='keyword', how='left')
        
        print("\nРаспределение ключей по кластерам:")
        cluster_key_counts = df_clusters['cluster'].value_counts().sort_index()
        for cluster, count in cluster_key_counts.items():
            print(f"  {cluster:20s}: {count:3d} ключей")
        
        # Статистика по кластерам
        print("\n" + "="*60)
        print("СТАТИСТИКА ПО КЛАСТЕРАМ")
        print("="*60)
        
        cluster_stats = self.df_with_clusters.groupby('cluster').agg({
            'motiv': ['mean', 'sum', 'max'],
            'position': ['mean', 'min', 'max'],
            'keyword': 'count'
        }).round(2)
        cluster_stats.columns = ['motiv_mean', 'motiv_sum', 'motiv_max', 
                                 'position_mean', 'position_min', 'position_max', 
                                 'keyword_count']
        cluster_stats = cluster_stats.sort_values('position_mean')
        
        print("\n  {:<15s} {:>10s} {:>12s} {:>10s} {:>12s} {:>12s} {:>12s}".format(
            'Кластер', 'Ключей', 'Мотив_сред', 'Мотив_сумма', 'Позиция_сред', 'Позиция_мин', 'Позиция_макс'
        ))
        print("  " + "-"*85)
        
        for cluster, row in cluster_stats.iterrows():
            print("  {:<15s} {:>10.0f} {:>12.2f} {:>10.0f} {:>12.2f} {:>12.0f} {:>12.0f}".format(
                cluster[:15],
                row['keyword_count'],
                row['motiv_mean'],
                row['motiv_sum'],
                row['position_mean'],
                row['position_min'],
                row['position_max']
            ))
        
        self.cluster_stats = cluster_stats
        self.df_clusters = df_clusters
    
    def _find_anchor_words(self):
        """Поиск якорных слов"""
        print("\n" + "="*60)
        print("5. ЯКОРНЫЕ СЛОВА (топ-3 по частоте в каждом кластере)")
        print("="*60)
        
        anchor_results = []
        
        for cluster in self.df_clusters['cluster'].unique():
            keywords_in_cluster = self.df_clusters[self.df_clusters['cluster'] == cluster]['keyword'].tolist()
            if not keywords_in_cluster:
                continue
            
            freq = self.df[self.df['keyword'].isin(keywords_in_cluster)]['keyword'].value_counts()
            top3 = freq.head(3)
            
            print(f"\n  Кластер {cluster} ({len(keywords_in_cluster)} ключей):")
            for kw, cnt in top3.items():
                kw_data = self.df[self.df['keyword'] == kw]
                pos_mean = kw_data['position'].mean()
                motiv_sum = kw_data['motiv'].sum()
                
                freq_score = cnt / len(keywords_in_cluster)
                motiv_score = motiv_sum / (kw_data['motiv'].sum() + 1)
                anchor_score = (freq_score * 0.5 + motiv_score * 0.5)
                
                print(f"    - {kw}: позиция={pos_mean:.1f}, мотив={motiv_sum:.0f}, частота={cnt}, score={anchor_score:.3f}")
                
                anchor_results.append({
                    'cluster': cluster,
                    'anchor_word': kw,
                    'motiv_mean': kw_data['motiv'].mean(),
                    'motiv_sum': motiv_sum,
                    'position_mean': pos_mean,
                    'anchor_score': anchor_score
                })
        
        # Сохраняем
        self.anchor_df = pd.DataFrame(anchor_results)
        if not self.anchor_df.empty:
            self.anchor_df = self.anchor_df.sort_values(['cluster', 'anchor_score'], ascending=[True, False])
            self.anchor_df.to_csv(self.data_path / "anchor_words.csv", index=False)
            print(f"\n  + Якорные слова сохранены: {self.data_path / 'anchor_words.csv'}")
    
    def _plot_clusters(self):
        """Визуализация кластеров"""
        print("\n" + "="*60)
        print("6. ВИЗУАЛИЗАЦИЯ КЛАСТЕРОВ")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Анализ кластеров по позициям и мотиву', fontsize=16, fontweight='bold')
        
        # 1. Распределение кластеров
        ax = axes[0, 0]
        cluster_sizes = self.df_clusters['cluster'].value_counts().sort_index()
        colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_sizes)))
        bars = ax.barh(cluster_sizes.index, cluster_sizes.values, color=colors, alpha=0.7)
        ax.set_xlabel('Количество ключей')
        ax.set_title('Размер кластеров')
        ax.grid(True, alpha=0.3)
        
        for i, (cluster, count) in enumerate(cluster_sizes.items()):
            ax.text(count + 0.5, i, str(count), va='center', fontsize=9)
        
        # 2. Средняя позиция по кластерам
        ax = axes[0, 1]
        cluster_pos = self.cluster_stats['position_mean'].sort_index()
        bars = ax.bar(cluster_pos.index, cluster_pos.values, color='coral', alpha=0.7)
        ax.set_xlabel('Кластер')
        ax.set_ylabel('Средняя позиция')
        ax.set_title('Средняя позиция по кластерам')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Граница топ-50')
        ax.axhline(y=10, color='green', linestyle='--', linewidth=2, label='Граница топ-10')
        ax.legend()
        
        for i, (cluster, pos) in enumerate(cluster_pos.items()):
            ax.text(i, pos + 2, f'{pos:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Средний мотив по кластерам
        ax = axes[1, 0]
        cluster_motiv = self.cluster_stats['motiv_mean'].sort_index()
        bars = ax.bar(cluster_motiv.index, cluster_motiv.values, color='green', alpha=0.7)
        ax.set_xlabel('Кластер')
        ax.set_ylabel('Средний мотив')
        ax.set_title('Средний мотив по кластерам')
        ax.grid(True, alpha=0.3)
        
        for i, (cluster, motiv) in enumerate(cluster_motiv.items()):
            ax.text(i, motiv + 0.1, f'{motiv:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 4. Матрица кластеров
        ax = axes[1, 1]
        scatter = ax.scatter(
            self.cluster_stats['position_mean'], 
            self.cluster_stats['motiv_mean'],
            s=self.cluster_stats['keyword_count'] * 15,
            alpha=0.6,
            c=range(len(self.cluster_stats)),
            cmap='viridis'
        )
        ax.set_xlabel('Средняя позиция')
        ax.set_ylabel('Средний мотив')
        ax.set_title('Позиция vs Мотив по кластерам (размер = количество ключей)')
        ax.grid(True, alpha=0.3)
        
        for idx, (cluster, row) in enumerate(self.cluster_stats.iterrows()):
            ax.annotate(cluster[:12], (row['position_mean'], row['motiv_mean']), 
                        fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(self.plots_path / 'cluster_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  + График кластеров сохранен: {self.plots_path / 'cluster_analysis.png'}")
    
    def _plot_cluster_timeline(self):
        """Динамика кластеров по времени"""
        print("\n" + "="*60)
        print("7. ДИНАМИКА КЛАСТЕРОВ ПО ВРЕМЕНИ")
        print("="*60)
        
        df_monthly_clusters = self.df_with_clusters.groupby([
            self.df_with_clusters['date'].dt.to_period('M'),
            'cluster'
        ]).agg({
            'motiv': 'mean',
            'position': 'mean',
            'keyword': 'count'
        }).reset_index()
        df_monthly_clusters.columns = ['month', 'cluster', 'motiv_mean', 'position_mean', 'count']
        
        # Топ-5 кластеров
        top_clusters = self.df_clusters['cluster'].value_counts().head(5).index.tolist()
        df_top_clusters = df_monthly_clusters[df_monthly_clusters['cluster'].isin(top_clusters)]
        
        if len(df_top_clusters) > 0:
            fig, axes = plt.subplots(2, 1, figsize=(16, 10))
            fig.suptitle('Динамика кластеров по месяцам', fontsize=16, fontweight='bold')
            
            # Позиции
            ax = axes[0]
            for cluster in top_clusters:
                data = df_top_clusters[df_top_clusters['cluster'] == cluster]
                ax.plot(data['month'].astype(str), data['position_mean'], 
                        marker='o', label=cluster, linewidth=2)
            ax.set_xlabel('Месяц')
            ax.set_ylabel('Средняя позиция')
            ax.set_title('Динамика средней позиции по кластерам')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            # Мотив
            ax = axes[1]
            for cluster in top_clusters:
                data = df_top_clusters[df_top_clusters['cluster'] == cluster]
                ax.plot(data['month'].astype(str), data['motiv_mean'], 
                        marker='s', label=cluster, linewidth=2)
            ax.set_xlabel('Месяц')
            ax.set_ylabel('Средний мотив')
            ax.set_title('Динамика среднего мотива по кластерам')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.plots_path / 'cluster_timeline.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  + Динамика кластеров сохранена: {self.plots_path / 'cluster_timeline.png'}")
    
    def _save_results(self):
        """Сохранение результатов"""
        print("\n" + "="*60)
        print("8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        print("="*60)
        
        self.df_with_clusters.to_pickle(self.data_path / "data_with_clusters.pkl")
        self.df_clusters.to_csv(self.data_path / "cluster_keywords.csv", index=False)
        
        print(f"  + Данные с кластерами: {self.data_path / 'data_with_clusters.pkl'}")
        print(f"  + Список ключей по кластерам: {self.data_path / 'cluster_keywords.csv'}")
        print(f"  + Якорные слова: {self.data_path / 'anchor_words.csv'}")
    
    def _generate_insights(self):
        """Генерация выводов"""
        print("\n" + "="*60)
        print("9. КЛЮЧЕВЫЕ ВЫВОДЫ ПО КЛАСТЕРИЗАЦИИ")
        print("="*60)
        
        print(f"""
1. РАСПРЕДЕЛЕНИЕ ПО КЛАСТЕРАМ:
   - Всего кластеров: {len(self.cluster_stats)}
   - Всего ключей: {len(self.df_clusters)}

2. КАТЕГОРИИ ПОЗИЦИЙ:
   - top10: {self.pos_dist.get('top10', 0)} ключей
   - top50: {self.pos_dist.get('top50', 0)} ключей  
   - above50: {self.pos_dist.get('above50', 0)} ключей

3. КАТЕГОРИИ МОТИВА:
   - zero: {self.motiv_dist.get('zero', 0)} ключей
   - low: {self.motiv_dist.get('low', 0)} ключей
   - medium: {self.motiv_dist.get('medium', 0)} ключей
   - high: {self.motiv_dist.get('high', 0)} ключей

4. ЛУЧШИЙ КЛАСТЕР (по позиции):
   - {self.cluster_stats['position_mean'].idxmin()} 
   - Средняя позиция: {self.cluster_stats['position_mean'].min():.2f}
   - Ключей: {self.cluster_stats.loc[self.cluster_stats['position_mean'].idxmin(), 'keyword_count']:.0f}

5. ХУДШИЙ КЛАСТЕР (по позиции):
   - {self.cluster_stats['position_mean'].idxmax()}
   - Средняя позиция: {self.cluster_stats['position_mean'].max():.2f}
   - Ключей: {self.cluster_stats.loc[self.cluster_stats['position_mean'].idxmax(), 'keyword_count']:.0f}

6. РЕКОМЕНДАЦИИ:
   - Для продвижения в топ-10 используйте кластеры с high мотивом
   - Для быстрого теста используйте якорные слова из каждого кластера
   - Следите за динамикой кластеров по месяцам
""")
    
    def get_cluster_info(self, cluster: str) -> dict:
        """Получить информацию о кластере"""
        if self.cluster_stats is None:
            return None
        
        if cluster not in self.cluster_stats.index:
            return None
        
        row = self.cluster_stats.loc[cluster]
        return {
            'cluster': cluster,
            'keywords_count': int(row['keyword_count']),
            'motiv_mean': float(row['motiv_mean']),
            'motiv_sum': float(row['motiv_sum']),
            'position_mean': float(row['position_mean']),
            'position_min': float(row['position_min']),
            'position_max': float(row['position_max'])
        }
    
    def get_anchor_words(self, cluster: str = None, top_n: int = 5) -> pd.DataFrame:
        """Получить якорные слова"""
        if self.anchor_df is None:
            return pd.DataFrame()
        
        if cluster:
            return self.anchor_df[self.anchor_df['cluster'] == cluster].head(top_n)
        else:
            return self.anchor_df.groupby('cluster').head(top_n)