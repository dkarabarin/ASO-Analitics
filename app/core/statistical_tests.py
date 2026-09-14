"""
Модуль статистических тестов v2.0
Шаг 5: Проверка гипотез H1-H8 с прогнозом на 1 день
Включает: H7 (фризы), H8 (iOS обновления), эффективность мотива
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, kruskal, ttest_rel, ttest_ind
import warnings
warnings.filterwarnings('ignore')


class StatisticalTester:
    """Класс для статистических тестов по кластерам (v2.0)"""
    
    def __init__(self, data_path: Path, df: pd.DataFrame = None):
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
        
        self.start_date = pd.to_datetime('2026-06-20')
        self.df_test = None
        self.clusters = []
        self.all_hypotheses = {'H1': {}, 'H2': {}, 'H3': {}, 'H4': {}, 'H5': {}, 'H6': {}, 'H7': {}, 'H8': {}}
        
        self.scenario_names = {
            'stable': 'стабильный',
            'sharp_growth': 'резкий рост',
            'sharp_decline': 'резкий спад',
            'gentle_growth': 'плавный рост',
            'gentle_decline': 'плавный спад'
        }
        
        # 🆕 Целевая метрика — прогноз на 1 день
        self.TARGET_COL = 'position_delta_1d'
    
    def run_tests(self) -> dict:
        """Запуск всех статистических тестов"""
        print("\n" + "="*80)
        print("ШАГ 5: СТАТИСТИЧЕСКИЕ ТЕСТЫ (ПРОГНОЗ НА 1 ДЕНЬ)")
        print("="*80)
        
        if self.df is None:
            raise ValueError("Данные не загружены")
        
        df = self.df
        
        print(f"\nРазмер данных: {len(df):,} записей")
        print(f"Уникальных ключей: {df['keyword'].nunique()}")
        print(f"Всего признаков: {len(df.columns)}")
        
        # Проверка целевой переменной
        if self.TARGET_COL not in df.columns:
            raise ValueError(f"Целевая переменная {self.TARGET_COL} не найдена. Перезапустите Шаг 1!")
        
        print(f"\n🎯 Целевая метрика: {self.TARGET_COL} (прогноз на 1 день)")
        
        # Фильтрация
        self._filter_data()
        
        # Проверка всех гипотез
        self._test_all_hypotheses()
        
        # Сводная таблица
        self._create_summary_table()
        
        # Визуализация
        self._plot_results()
        
        # Сохранение
        self._save_results()
        
        print("\n" + "="*80)
        print("✅ ШАГ 5 ЗАВЕРШЕН УСПЕШНО!")
        print("="*80)
        
        return self.all_hypotheses
    
    def _filter_data(self):
        """Фильтрация данных после 20.06.2026"""
        print("\n" + "="*60)
        print("ФИЛЬТРАЦИЯ ДАННЫХ")
        print("="*60)
        
        self.df_test = self.df[self.df['date'] >= self.start_date].copy()
        
        print(f"\n  Исходный размер данных: {len(self.df):,} записей")
        print(f"  Данных с 20.06.2026: {len(self.df_test):,} записей")
        
        if len(self.df_test) > 0:
            print(f"  Период: {self.df_test['date'].min()} - {self.df_test['date'].max()}")
            print(f"  Уникальных ключей: {self.df_test['keyword'].nunique()}")
        else:
            print("\n  ⚠️ Нет данных после 20.06.2026! Используем последние 30% данных")
            self.df_test = self.df.sort_values('date').tail(int(len(self.df) * 0.3))
            print(f"  Взято последних 30% данных: {len(self.df_test):,} записей")
        
        # Кластеры (если есть)
        if 'cluster' in self.df_test.columns:
            self.clusters = [c for c in self.df_test['cluster'].unique() 
                            if c != 'nan' and not pd.isna(c)]
            print(f"\n  Активных кластеров: {len(self.clusters)}")
        else:
            self.clusters = []
            print(f"\n  Кластеры отсутствуют (не критично)")
    
    def _test_all_hypotheses(self):
        """Проверка всех гипотез"""
        print("\n" + "="*60)
        print("ПРОВЕРКА ГИПОТЕЗ")
        print("="*60)
        
        # H1: Похожие запросы
        print("\n" + "─"*50)
        print("H1: Похожие запросы (корреляция > 0.7)")
        print("─"*50)
        result_h1 = self._test_hypothesis_1()
        if result_h1:
            self.all_hypotheses['H1']['global'] = result_h1
            print(f"\n  Средняя корреляция: {result_h1['mean_corr']:.3f}")
            print(f"  Максимальная: {result_h1['max_corr']:.3f}")
            print(f"  p-value: {result_h1['p_value']:.4f}")
            print(f"  Результат: {result_h1['result']}")
        
        # H2: Согласованность сценариев
        print("\n" + "─"*50)
        print("H2: Согласованность сценариев мотива")
        print("─"*50)
        result_h2 = self._test_hypothesis_2()
        if result_h2:
            self.all_hypotheses['H2']['global'] = result_h2
            print(f"  Сценариев: {result_h2['n_scenarios']}")
            print(f"  H-статистика: {result_h2['h_stat']:.4f}")
            print(f"  p-value: {result_h2['p_value']:.4f}")
            print(f"  Результат: {result_h2['result']}")
        
        # H3: Эффективность сценариев
        print("\n" + "─"*50)
        print("H3: Эффективность сценариев")
        print("─"*50)
        result_h3 = self._test_hypothesis_3()
        if result_h3:
            self.all_hypotheses['H3']['global'] = result_h3
            print(f"  Лучший сценарий: {result_h3['best_scenario']['name']}")
            print(f"  Среднее Δ: {result_h3['best_scenario']['mean']:.3f}")
            print(f"  p-value: {result_h3['p_value']:.4f}")
            print(f"  Результат: {result_h3['result']}")
        
        # H4: Влияние фризов (мотив=0)
        print("\n" + "─"*50)
        print("H4: Влияние фризов (мотив = 0)")
        print("─"*50)
        result_h4 = self._test_hypothesis_4()
        if result_h4:
            self.all_hypotheses['H4']['global'] = result_h4
            print(f"  Записей без мотива: {result_h4['n_zero']:,}")
            print(f"  Записей с мотивом: {result_h4['n_nonzero']:,}")
            print(f"  Δ с мотивом: {result_h4['mean_delta_nonzero']:.3f}")
            print(f"  Δ без мотива: {result_h4['mean_delta_zero']:.3f}")
            print(f"  p-value: {result_h4['p_value']:.4f}")
            print(f"  Результат: {result_h4['result']}")
        
        # H5: Влияние мотива
        print("\n" + "─"*50)
        print("H5: Влияние мотива на позицию")
        print("─"*50)
        result_h5 = self._test_hypothesis_5()
        if result_h5:
            self.all_hypotheses['H5']['global'] = result_h5
            print(f"  Pearson: r = {result_h5['pearson_corr']:.4f}, p = {result_h5['p_pearson']:.4f}")
            print(f"  Spearman: r = {result_h5['spearman_corr']:.4f}, p = {result_h5['p_spearman']:.4f}")
            print(f"  Результат: {result_h5['result']}")
        
        # H6: Эффективность мотива
        print("\n" + "─"*50)
        print("H6: Эффективность мотива > 0")
        print("─"*50)
        result_h6 = self._test_hypothesis_6()
        if result_h6:
            self.all_hypotheses['H6']['global'] = result_h6
            print(f"  Ключей: {result_h6['n_keywords']}")
            print(f"  Средняя эффективность: {result_h6['mean_efficiency']:.3f}")
            print(f"  Положительных: {result_h6['positive_count']} ({result_h6['positive_pct']:.1f}%)")
            print(f"  p-value: {result_h6['p_value']:.4f}")
            print(f"  Результат: {result_h6['result']}")
        
        # 🆕 H7: Влияние фризов (события)
        print("\n" + "─"*50)
        print("H7: Влияние фризов (события) 🆕")
        print("─"*50)
        result_h7 = self._test_hypothesis_7()
        if result_h7:
            self.all_hypotheses['H7']['global'] = result_h7
            print(f"  Записей во фризе: {result_h7['n_in_freeze']:,}")
            print(f"  Записей вне фриза: {result_h7['n_out_freeze']:,}")
            print(f"  Δ во фризе: {result_h7['mean_in_freeze']:.3f}")
            print(f"  Δ вне фриза: {result_h7['mean_out_freeze']:.3f}")
            print(f"  p-value: {result_h7['p_value']:.4f}")
            print(f"  Результат: {result_h7['result']}")
        
        # 🆕 H8: Влияние iOS обновлений
        print("\n" + "─"*50)
        print("H8: Влияние iOS обновлений 🆕")
        print("─"*50)
        result_h8 = self._test_hypothesis_8()
        if result_h8:
            self.all_hypotheses['H8']['global'] = result_h8
            print(f"  Записей за неделю после iOS: {result_h8['n_week_after']:,}")
            print(f"  Записей в обычные дни: {result_h8['n_normal']:,}")
            print(f"  Δ после iOS: {result_h8['mean_week_after']:.3f}")
            print(f"  Δ обычные: {result_h8['mean_normal']:.3f}")
            print(f"  p-value: {result_h8['p_value']:.4f}")
            print(f"  Результат: {result_h8['result']}")
    
    def _test_hypothesis_1(self):
        """H1: Корреляция позиций похожих ключей > 0.7"""
        top_keywords = self.df_test['keyword'].value_counts().head(20).index.tolist()
        df_top = self.df_test[self.df_test['keyword'].isin(top_keywords)]
        
        pivot = df_top.pivot_table(index='date', columns='keyword', values='position').dropna(axis=1, how='all')
        
        if pivot.shape[1] < 2:
            return None
        
        pivot = pivot.fillna(pivot.mean())
        corr_matrix = pivot.corr()
        
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        corr_values = upper_tri.stack().values
        corr_values = corr_values[~np.isnan(corr_values)]
        
        if len(corr_values) == 0:
            return None
        
        t_stat, p_value = ttest_ind(corr_values, [0.7] * len(corr_values), alternative='less')
        
        return {
            'n_keywords': len(top_keywords),
            'n_pairs': len(corr_values),
            'mean_corr': float(np.mean(corr_values)),
            'median_corr': float(np.median(corr_values)),
            'max_corr': float(np.max(corr_values)),
            'min_corr': float(np.min(corr_values)),
            't_stat': float(t_stat),
            'p_value': float(p_value),
            'hypothesis_accepted': bool(p_value < 0.05),
            'result': '✅ ПОДТВЕРЖДЕНА' if p_value < 0.05 else '❌ НЕ ПОДТВЕРЖДЕНА'
        }
    
    def _test_hypothesis_2(self):
        """H2: Согласованность сценариев"""
        df_analysis = self.df_test.dropna(subset=[self.TARGET_COL])
        
        if len(df_analysis) < 50:
            return None
        
        groups = [group[self.TARGET_COL].values for name, group in df_analysis.groupby('motiv_scenario') 
                  if len(group) >= 10]
        
        if len(groups) < 2:
            return None
        
        h_stat, p_value = kruskal(*groups)
        
        return {
            'n_scenarios': len(groups),
            'h_stat': float(h_stat),
            'p_value': float(p_value),
            'hypothesis_accepted': bool(p_value < 0.05),
            'result': '✅ ПОДТВЕРЖДЕНА' if p_value < 0.05 else '❌ НЕ ПОДТВЕРЖДЕНА'
        }
    
    def _test_hypothesis_3(self):
        """H3: Эффективность сценариев"""
        df_scenarios = self.df_test[self.df_test['motiv_scenario'] != 'stable'].dropna(subset=[self.TARGET_COL])
        
        if len(df_scenarios) < 20:
            return None
        
        groups = []
        scenario_stats = []
        
        for scenario, group in df_scenarios.groupby('motiv_scenario'):
            if len(group) >= 5:
                groups.append(group[self.TARGET_COL].values)
                scenario_stats.append({
                    'scenario': scenario,
                    'name': self.scenario_names.get(scenario, scenario),
                    'mean': float(group[self.TARGET_COL].mean()),
                    'median': float(group[self.TARGET_COL].median()),
                    'std': float(group[self.TARGET_COL].std()),
                    'count': int(len(group))
                })
        
        if len(groups) < 2:
            return None
        
        h_stat, p_value = kruskal(*groups)
        
        scenario_df = pd.DataFrame(scenario_stats)
        best_scenario = scenario_df.loc[scenario_df['mean'].idxmin()].to_dict()
        
        return {
            'n_records': int(len(df_scenarios)),
            'n_scenarios': len(groups),
            'h_stat': float(h_stat),
            'p_value': float(p_value),
            'best_scenario': best_scenario,
            'hypothesis_accepted': bool(p_value < 0.05),
            'result': '✅ ПОДТВЕРЖДЕНА' if p_value < 0.05 else '❌ НЕ ПОДТВЕРЖДЕНА'
        }
    
    def _test_hypothesis_4(self):
        """H4: Влияние фризов (мотив = 0)"""
        df_zero = self.df_test[self.df_test['motiv'] == 0]
        df_nonzero = self.df_test[self.df_test['motiv'] > 0]
        
        if len(df_zero) < 20 or len(df_nonzero) < 20:
            return None
        
        delta_zero = df_zero.dropna(subset=[self.TARGET_COL])[self.TARGET_COL]
        delta_nonzero = df_nonzero.dropna(subset=[self.TARGET_COL])[self.TARGET_COL]
        
        if len(delta_zero) < 10 or len(delta_nonzero) < 10:
            return None
        
        stat, p_value = mannwhitneyu(delta_zero, delta_nonzero, alternative='two-sided')
        
        n1, n2 = len(delta_zero), len(delta_nonzero)
        mean1, mean2 = delta_zero.mean(), delta_nonzero.mean()
        std1, std2 = delta_zero.std(), delta_nonzero.std()
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        return {
            'n_zero': int(len(df_zero)),
            'n_nonzero': int(len(df_nonzero)),
            'mean_delta_zero': float(mean1),
            'mean_delta_nonzero': float(mean2),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'hypothesis_accepted': bool(p_value < 0.05),
            'result': '✅ ПОДТВЕРЖДЕНА' if p_value < 0.05 else '❌ НЕ ПОДТВЕРЖДЕНА'
        }
    
    def _test_hypothesis_5(self):
        """H5: Влияние мотива на позицию"""
        df_analysis = self.df_test.dropna(subset=[self.TARGET_COL])
        
        if len(df_analysis) < 50:
            return None
        
        pearson_corr, p_pearson = pearsonr(df_analysis['motiv'], df_analysis[self.TARGET_COL])
        spearman_corr, p_spearman = spearmanr(df_analysis['motiv'], df_analysis[self.TARGET_COL])
        
        hypothesis_accepted = p_pearson < 0.05 or p_spearman < 0.05
        
        return {
            'n_records': int(len(df_analysis)),
            'pearson_corr': float(pearson_corr),
            'p_pearson': float(p_pearson),
            'spearman_corr': float(spearman_corr),
            'p_spearman': float(p_spearman),
            'hypothesis_accepted': bool(hypothesis_accepted),
            'result': '✅ ПОДТВЕРЖДЕНА' if hypothesis_accepted else '❌ НЕ ПОДТВЕРЖДЕНА'
        }
    
    def _test_hypothesis_6(self):
        """H6: Эффективность мотива"""
        eff_path = self.data_path / "motiv_efficiency.csv"
        if not eff_path.exists():
            return None
        
        eff_df = pd.read_csv(eff_path)
        eff_values = eff_df['efficiency'].dropna()
        eff_values = eff_values[(eff_values > -10) & (eff_values < 50)]
        
        if len(eff_values) < 10:
            return None
        
        t_stat, p_value = ttest_ind(eff_values, [0] * len(eff_values), alternative='greater')
        positive_count = (eff_values > 0).sum()
        
        return {
            'n_keywords': int(len(eff_values)),
            'mean_efficiency': float(eff_values.mean()),
            'median_efficiency': float(eff_values.median()),
            'positive_count': int(positive_count),
            'positive_pct': float(positive_count / len(eff_values) * 100),
            't_stat': float(t_stat),
            'p_value': float(p_value),
            'hypothesis_accepted': bool(p_value < 0.05 and eff_values.mean() > 0),
            'result': '✅ ПОДТВЕРЖДЕНА' if (p_value < 0.05 and eff_values.mean() > 0) else '❌ НЕ ПОДТВЕРЖДЕНА'
        }
    
    def _test_hypothesis_7(self):
        """H7: Влияние фризов (события)"""
        if 'is_freeze' not in self.df_test.columns:
            return None
        
        df_analysis = self.df_test.dropna(subset=[self.TARGET_COL])
        
        in_freeze = df_analysis[df_analysis['is_freeze'] == 1][self.TARGET_COL]
        out_freeze = df_analysis[df_analysis['is_freeze'] == 0][self.TARGET_COL]
        
        if len(in_freeze) < 10 or len(out_freeze) < 10:
            return None
        
        stat, p_value = mannwhitneyu(in_freeze, out_freeze, alternative='two-sided')
        
        n1, n2 = len(in_freeze), len(out_freeze)
        mean1, mean2 = in_freeze.mean(), out_freeze.mean()
        std1, std2 = in_freeze.std(), out_freeze.std()
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        return {
            'n_in_freeze': int(len(in_freeze)),
            'n_out_freeze': int(len(out_freeze)),
            'mean_in_freeze': float(mean1),
            'mean_out_freeze': float(mean2),
            'diff': float(mean1 - mean2),
            'cohens_d': float(cohens_d),
            'p_value': float(p_value),
            'hypothesis_accepted': bool(p_value < 0.05),
            'result': '✅ ПОДТВЕРЖДЕНА' if p_value < 0.05 else '❌ НЕ ПОДТВЕРЖДЕНА'
        }
    
    def _test_hypothesis_8(self):
        """H8: Влияние iOS обновлений"""
        if 'is_week_after_ios_update' not in self.df_test.columns:
            return None
        
        df_analysis = self.df_test.dropna(subset=[self.TARGET_COL])
        
        week_after = df_analysis[df_analysis['is_week_after_ios_update'] == 1][self.TARGET_COL]
        normal_days = df_analysis[df_analysis['is_week_after_ios_update'] == 0][self.TARGET_COL]
        
        if len(week_after) < 10 or len(normal_days) < 10:
            return None
        
        stat, p_value = mannwhitneyu(week_after, normal_days, alternative='two-sided')
        
        n1, n2 = len(week_after), len(normal_days)
        mean1, mean2 = week_after.mean(), normal_days.mean()
        std1, std2 = week_after.std(), normal_days.std()
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        return {
            'n_week_after': int(len(week_after)),
            'n_normal': int(len(normal_days)),
            'mean_week_after': float(mean1),
            'mean_normal': float(mean2),
            'diff': float(mean1 - mean2),
            'cohens_d': float(cohens_d),
            'p_value': float(p_value),
            'hypothesis_accepted': bool(p_value < 0.05),
            'result': '✅ ПОДТВЕРЖДЕНА' if p_value < 0.05 else '❌ НЕ ПОДТВЕРЖДЕНА'
        }
    
    def _create_summary_table(self):
        """Сводная таблица гипотез"""
        print("\n" + "="*60)
        print("СВОДНАЯ ТАБЛИЦА ГИПОТЕЗ")
        print("="*60)
        
        summary = []
        hypothesis_names = {
            'H1': 'Похожие запросы',
            'H2': 'Согласованность сценариев',
            'H3': 'Эффективность сценариев',
            'H4': 'Влияние фризов (мотив=0)',
            'H5': 'Влияние мотива',
            'H6': 'Эффективность мотива',
            'H7': 'Влияние фризов (события) 🆕',
            'H8': 'Влияние iOS обновлений 🆕'
        }
        
        for h_id, results in self.all_hypotheses.items():
            if 'global' in results:
                r = results['global']
                summary.append({
                    'Гипотеза': f"{h_id}: {hypothesis_names.get(h_id, h_id)}",
                    'Метрика': self._get_metric(h_id, r),
                    'p-value': f"{r.get('p_value', 0):.4f}",
                    'Результат': r.get('result', 'N/A')
                })
        
        self.summary_df = pd.DataFrame(summary)
        
        if len(self.summary_df) > 0:
            print("\n" + self.summary_df.to_string(index=False))
            
            accepted = (self.summary_df['Результат'].str.contains('✅')).sum()
            total = len(self.summary_df)
            print(f"\n📊 ИТОГО: {accepted}/{total} гипотез подтверждено ({accepted/total*100:.1f}%)")
        
        # Сохранение
        self.summary_df.to_csv(self.data_path / "hypotheses_summary_1day.csv", index=False)
        print(f"\n  + Сводная таблица сохранена: {self.data_path / 'hypotheses_summary_1day.csv'}")
    
    def _get_metric(self, h_id, r):
        """Метрика для гипотезы"""
        if h_id == 'H1':
            return f"Ср.корр: {r.get('mean_corr', 0):.3f}"
        elif h_id == 'H2':
            return f"Сценариев: {r.get('n_scenarios', 0)}"
        elif h_id == 'H3':
            return f"Лучший: {r.get('best_scenario', {}).get('name', 'N/A')}"
        elif h_id == 'H4':
            return f"Δ: {r.get('mean_delta_nonzero', 0):.3f} vs {r.get('mean_delta_zero', 0):.3f}"
        elif h_id == 'H5':
            return f"Spearman: {r.get('spearman_corr', 0):.3f}"
        elif h_id == 'H6':
            return f"Ср.эфф: {r.get('mean_efficiency', 0):.3f}"
        elif h_id == 'H7':
            return f"Δ фриз: {r.get('mean_in_freeze', 0):.3f}"
        elif h_id == 'H8':
            return f"Δ iOS: {r.get('mean_week_after', 0):.3f}"
        return 'N/A'
    
    def _plot_results(self):
        """Визуализация результатов"""
        print("\n" + "="*60)
        print("ВИЗУАЛИЗАЦИЯ")
        print("="*60)
        
        # 8 гипотез - 2x4
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        fig.suptitle('Результаты статистических тестов (прогноз на 1 день)', fontsize=16, fontweight='bold')
        
        # H1
        ax = axes[0, 0]
        if 'H1' in self.all_hypotheses and 'global' in self.all_hypotheses['H1']:
            h1 = self.all_hypotheses['H1']['global']
            ax.bar(['Средняя', 'Медиана', 'Максимум'], 
                   [h1['mean_corr'], h1['median_corr'], h1['max_corr']],
                   color=['steelblue', 'orange', 'green'], alpha=0.7, edgecolor='black')
            ax.axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='Порог 0.7')
            ax.set_title(f"H1: Корреляции\n{h1['result']}", fontsize=10)
            ax.set_ylabel('Корреляция')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # H2
        ax = axes[0, 1]
        if 'H2' in self.all_hypotheses and 'global' in self.all_hypotheses['H2']:
            h2 = self.all_hypotheses['H2']['global']
            ax.bar(['H-статистика'], [h2['h_stat']], color='steelblue', alpha=0.7)
            ax.set_title(f"H2: Согласованность\n{h2['result']}\np={h2['p_value']:.4f}", fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # H3
        ax = axes[0, 2]
        if 'H3' in self.all_hypotheses and 'global' in self.all_hypotheses['H3']:
            h3 = self.all_hypotheses['H3']['global']
            best = h3['best_scenario']
            ax.bar([best['name']], [best['mean']], color='green', alpha=0.7, edgecolor='black')
            ax.set_title(f"H3: Лучший сценарий\n{best['name']} ({best['mean']:.3f})", fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # H4
        ax = axes[0, 3]
        if 'H4' in self.all_hypotheses and 'global' in self.all_hypotheses['H4']:
            h4 = self.all_hypotheses['H4']['global']
            ax.bar(['С мотивом', 'Без мотива'], 
                   [h4['mean_delta_nonzero'], h4['mean_delta_zero']],
                   color=['green', 'red'], alpha=0.7, edgecolor='black')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.set_title(f"H4: Фризы (мотив=0)\n{h4['result']}", fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # H5
        ax = axes[1, 0]
        if 'H5' in self.all_hypotheses and 'global' in self.all_hypotheses['H5']:
            h5 = self.all_hypotheses['H5']['global']
            ax.bar(['Pearson', 'Spearman'], 
                   [h5['pearson_corr'], h5['spearman_corr']],
                   color=['steelblue', 'orange'], alpha=0.7, edgecolor='black')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.set_title(f"H5: Влияние мотива\n{h5['result']}", fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # H6
        ax = axes[1, 1]
        if 'H6' in self.all_hypotheses and 'global' in self.all_hypotheses['H6']:
            h6 = self.all_hypotheses['H6']['global']
            ax.bar(['Средняя\nэффективность'], [h6['mean_efficiency']], 
                   color='purple', alpha=0.7, edgecolor='black')
            ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
            ax.set_title(f"H6: Эффективность\n{h6['result']}", fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # H7 (НОВАЯ)
        ax = axes[1, 2]
        if 'H7' in self.all_hypotheses and 'global' in self.all_hypotheses['H7']:
            h7 = self.all_hypotheses['H7']['global']
            ax.bar(['Во фризе', 'Вне фриза'], 
                   [h7['mean_in_freeze'], h7['mean_out_freeze']],
                   color=['purple', 'steelblue'], alpha=0.7, edgecolor='black')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.set_title(f"H7: Фризы (события) 🆕\n{h7['result']}", fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # H8 (НОВАЯ)
        ax = axes[1, 3]
        if 'H8' in self.all_hypotheses and 'global' in self.all_hypotheses['H8']:
            h8 = self.all_hypotheses['H8']['global']
            ax.bar(['После iOS', 'Обычные'], 
                   [h8['mean_week_after'], h8['mean_normal']],
                   color=['orange', 'steelblue'], alpha=0.7, edgecolor='black')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.set_title(f"H8: iOS обновления 🆕\n{h8['result']}", fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_path / 'hypotheses_1day.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  + График сохранён: {self.plots_path / 'hypotheses_1day.png'}")
    
    def _save_results(self):
        """Сохранение"""
        pass  # Уже сохранено в _create_summary_table
    
    def get_hypothesis_results(self) -> dict:
        return self.all_hypotheses
    
    def get_summary(self) -> pd.DataFrame:
        return self.summary_df