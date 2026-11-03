import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import warnings
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Independence
import math

sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
warnings.filterwarnings('ignore')

plt.style.use('ggplot')
plt.rcParams['axes.unicode_minus'] = False

analysis_dir = os.path.join('demo_results', 'asso_3_qcomp')
os.makedirs(analysis_dir, exist_ok=True)

try:
    data_file = os.path.join('demo_data', 'df_association.xlsx')
    df_all = pd.read_excel(data_file)
    print(f"Data shape: {df_all.shape}")

    if 'clin_pregnancy_trimester' not in df_all.columns:
        print("Error: 'clin_pregnancy_trimester' column missing")
        sys.exit(1)

    pregnancy_stages = df_all['clin_pregnancy_trimester'].unique()
    print(f"Pregnancy trimesters: {pregnancy_stages}")

    demo_confounders = []
    demo_file = os.path.join('demo_results', 'asso_1_single factor', 'summary', 'demographic_significant_variables.xlsx')
    try:
        if os.path.exists(demo_file):
            df_demo = pd.read_excel(demo_file)
            demo_confounders = df_demo['variable_raw'].tolist()
        else:
            print(f"Warning: demographic confounders file not found: {demo_file}")
    except Exception as e:
        print(f"Error reading demographic confounders file: {e}")

    clinical_confounders = []
    clinical_file = os.path.join('demo_results', 'asso_1_single factor', 'summary', 'clinical_significant_variables.xlsx')
    try:
        if os.path.exists(clinical_file):
            df_clinical = pd.read_excel(clinical_file)
            clinical_confounders = df_clinical['variable_raw'].tolist()
        else:
            print(f"Warning: clinical confounders file not found: {clinical_file}")
    except Exception as e:
        print(f"Error reading clinical confounders file: {e}")

    confounders = list(set(demo_confounders + clinical_confounders))

    if not confounders:
        print("Warning: confounders list is empty, using default confounders")
        demo_confounders = ["epi_age", "epi_education_level", "epi_bmi"]
        clinical_confounders = ["clin_cesarean_section"]
        confounders = demo_confounders + clinical_confounders

    confounders_dir = analysis_dir
    df_confounders = pd.DataFrame({
        'variable_raw': confounders,
        'variable_type': ['demographic' if var in demo_confounders else 'clinical' for var in confounders]
    })
    confounders_file = os.path.join(confounders_dir, 'confounding_variables.xlsx')
    df_confounders.to_excel(confounders_file, index=False)
    print(f"Confounders: {confounders}")

    pfas_vars = [col for col in df_all.columns if col.startswith('pfas_')]
    print(f"Number of PFAS variables: {len(pfas_vars)}")

    metabolic_vars = []

    all_vars = pfas_vars + confounders + ['clin_pregnancy_trimester']

    missing_threshold = 0.4
    cv_threshold = 0.1

    stage_filter_results = {}
    all_selected_vars = set(['clin_pregnancy_trimester'] + confounders)

    for stage in pregnancy_stages:
        df_stage = df_all[df_all['clin_pregnancy_trimester'] == stage].copy()

        stage_results = {'filtered_vars': [], 'missing_rate_vars': [], 'low_cv_vars': [], 'selected_vars': []}

        for var in all_vars:
            if var == 'clin_pregnancy_trimester':
                stage_results['selected_vars'].append(var)
                continue
            if var not in df_stage.columns:
                stage_results['filtered_vars'].append(var)
                continue
            if not pd.api.types.is_numeric_dtype(df_stage[var]):
                stage_results['selected_vars'].append(var)
                continue

            missing_rate = df_stage[var].isna().mean()
            if missing_rate > missing_threshold:
                stage_results['missing_rate_vars'].append(var)
                continue

            mean = df_stage[var].mean()
            std = df_stage[var].std()
            if mean == 0 or pd.isna(mean) or pd.isna(std):
                stage_results['selected_vars'].append(var)
                continue

            cv = abs(std / mean)
            if cv < cv_threshold:
                stage_results['low_cv_vars'].append(var)
                continue

            stage_results['selected_vars'].append(var)

        stage_filter_results[stage] = stage_results
        all_selected_vars.update(stage_results['selected_vars'])

        selected_pfas = [var for var in pfas_vars if var in stage_results['selected_vars']]
        print(f"{stage}: selected {len(stage_results['selected_vars'])}/{len(all_vars)} variables "
              f"(PFAS {len(selected_pfas)}/{len(pfas_vars)})")

    selected_vars = list(all_selected_vars)
    print(f"\nTotal selected variables across all trimesters: {len(selected_vars)}/{len(all_vars)}")

    filter_log_file = os.path.join(analysis_dir, 'variable_filter_log.txt')
    with open(filter_log_file, 'w', encoding='utf-8') as f:
        for stage, results in stage_filter_results.items():
            f.write(f"\n{'-'*50}\n{stage} filtering results:\n")
            f.write(f"  Variables before filtering: {len(all_vars)}\n")
            f.write(f"  High missing rate variables: {len(results['missing_rate_vars'])}\n")
            f.write(f"  Low CV variables: {len(results['low_cv_vars'])}\n")
            f.write(f"  Selected variables: {len(results['selected_vars'])}\n\n")

    required_vars = ['clin_pregnancy_trimester', 'TD']
    missing_vars = [var for var in required_vars if var not in df_all.columns]
    if missing_vars:
        print(f"Error: missing required variables: {missing_vars}")
        sys.exit(1)

    non_feature_vars = ['id_enrollment', 'clin_pregnancy_trimester', 'clin_examination_date',
                         'clin_last_menstrual_period', 'TD']

    for stage in pregnancy_stages:
        print(f"\n{'-'*50}\nAnalyzing trimester: {stage}")

        stage_dir = os.path.join(analysis_dir, f"{stage}")
        os.makedirs(stage_dir, exist_ok=True)

        df_stage = df_all[df_all['clin_pregnancy_trimester'] == stage].copy()
        print(f"{stage} data shape: {df_stage.shape}")

        df_analysis = pd.DataFrame()
        df_analysis['TD'] = df_stage['TD']

        feature_vars = []
        for var in selected_vars:
            if var in non_feature_vars:
                continue
            if var in df_stage.columns:
                feature_vars.append(var)
                df_analysis[var] = df_stage[var]
        print(f"{stage} feature variable count: {len(feature_vars)}")

        stage_data_file = os.path.join(stage_dir, f"{stage}_analysis_data.xlsx")
        df_analysis.to_excel(stage_data_file, index=False)

        df_std = df_analysis.copy()

        std_vars = []
        for col in feature_vars:
            if col == 'TD':
                continue
            if col not in df_analysis.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df_analysis[col]):
                continue
            if df_analysis[col].count() < 10:
                continue
            if df_analysis[col].std() == 0:
                continue

            try:
                scaler = StandardScaler()
                non_null_mask = ~df_analysis[col].isna()
                values = df_analysis.loc[non_null_mask, col].values.reshape(-1, 1)
                scaled_values = scaler.fit_transform(values).flatten()
                df_std.loc[non_null_mask, col + '_std'] = scaled_values
                std_vars.append(col + '_std')
            except Exception as e:
                print(f"Error standardizing {col}: {e}")

        std_file = os.path.join(stage_dir, f"{stage}_standardized.xlsx")
        df_std.to_excel(std_file, index=False)

        analysis_vars = std_vars if std_vars else [col for col in feature_vars if col in df_std.columns and col != 'TD']
        analysis_vars = [var for var in analysis_vars if not var.startswith('TD')]

        analysis_results = []

        print(f"\nAnalyzing PFAS mixture effect for {stage} using quantile-based g-computation (qcomp)...")

        for i, var in enumerate(analysis_vars):
            if var not in df_std.columns:
                continue
            if df_std[var].count() < 10:
                continue

            result = {
                'variable': var, 'p_value': None, 'coefficient': None, 'std_error': None,
                'ci_lower': None, 'ci_upper': None, 'HR': None, 'HR_ci_lower': None,
                'HR_ci_upper': None, 'HR_IQR': None, 'HR_IQR_ci_lower': None, 'HR_IQR_ci_upper': None,
                'Q1': None, 'Q2': None, 'Q3': None, 'IQR': None
            }

            try:
                model_data = df_std.dropna(subset=[var, 'TD'])
                if len(model_data) < 10:
                    continue

                var_raw = var[:-4] if var.endswith('_std') else var
                result['variable_raw'] = var_raw

                q1 = model_data[var].quantile(0.25)
                q2 = model_data[var].quantile(0.50)
                q3 = model_data[var].quantile(0.75)
                iqr = q3 - q1
                result['Q1'], result['Q2'], result['Q3'], result['IQR'] = q1, q2, q3, iqr

                min_days = model_data['TD'].min()
                if min_days < 0:
                    model_data['TD_positive'] = model_data['TD'] - min_days + 1
                else:
                    model_data['TD_positive'] = model_data['TD'] + 1

                available_confounders = [c for c in confounders if c in model_data.columns]

                if i == 0:
                    var_confounders_file = os.path.join(stage_dir, f'{stage}_var_analysis_confounders.xlsx')
                    pd.DataFrame({'variable_raw': available_confounders}).to_excel(var_confounders_file, index=False)

                if available_confounders:
                    X = sm.add_constant(model_data[[var] + available_confounders])
                else:
                    X = sm.add_constant(model_data[var])
                y = model_data['TD_positive']

                family = Poisson()
                groups = np.arange(len(X))
                model = GEE(y, X, groups, family=family, cov_struct=Independence())
                model_result = model.fit()

                result['p_value'] = model_result.pvalues[var] if var in model_result.pvalues else None
                result['coefficient'] = model_result.params[var] if var in model_result.params else None
                result['std_error'] = model_result.bse[var] if var in model_result.bse else None

                if result['coefficient'] is not None and result['std_error'] is not None:
                    result['ci_lower'] = result['coefficient'] - 1.96 * result['std_error']
                    result['ci_upper'] = result['coefficient'] + 1.96 * result['std_error']
                    result['HR'] = math.exp(result['coefficient'])
                    result['HR_ci_lower'] = math.exp(result['ci_lower'])
                    result['HR_ci_upper'] = math.exp(result['ci_upper'])
                    result['HR_IQR'] = math.exp(result['coefficient'] * iqr)
                    result['HR_IQR_ci_lower'] = math.exp(result['ci_lower'] * iqr)
                    result['HR_IQR_ci_upper'] = math.exp(result['ci_upper'] * iqr)

                    print(f"  {var}: coef={result['coefficient']:.4f}, HR={result['HR']:.4f}, "
                          f"p={result['p_value']:.4f}")

                analysis_results.append(result)

            except Exception as e:
                print(f"  Error analyzing variable {var}: {e}")

        if analysis_results:
            results_df = pd.DataFrame(analysis_results)
            results_df = results_df.sort_values('p_value').reset_index(drop=True)

            if 'variable_raw' not in results_df.columns:
                results_df['variable_raw'] = results_df['variable'].apply(lambda x: x[:-4] if x.endswith('_std') else x)

            cols = results_df.columns.tolist()
            cols.remove('variable_raw')
            cols = ['variable_raw'] + cols
            results_df = results_df[cols]

            results_file = os.path.join(stage_dir, f'{stage}_regression_results.xlsx')
            results_df.to_excel(results_file, index=False)
            print(f"{stage} regression results saved to: {results_file}")

            try:
                print(f"\nRunning quantile-based g-computation (qcomp) for {stage}...")

                pfas_analysis_vars = [var for var in analysis_vars if var.split('_')[0] in pfas_vars or var[:-4] in pfas_vars]

                if len(pfas_analysis_vars) > 0:
                    print(f"Found {len(pfas_analysis_vars)} PFAS variables for qcomp analysis")

                    qcomp_data = df_std.dropna(subset=['TD'] + pfas_analysis_vars).copy()

                    if len(qcomp_data) < 10:
                        print(f"Insufficient data points for qcomp analysis ({len(qcomp_data)} < 10)")
                    else:
                        print(f"qcomp analysis sample size: {len(qcomp_data)}")

                        q = 4
                        qt = QuantileTransformer(n_quantiles=q, output_distribution='uniform')
                        qcomp_data_quantiles = qcomp_data.copy()

                        for var in pfas_analysis_vars:
                            non_null_mask = ~qcomp_data[var].isna()
                            if non_null_mask.sum() > q:
                                values = qcomp_data.loc[non_null_mask, var].values.reshape(-1, 1)
                                quantile_values = qt.fit_transform(values).flatten() * q
                                qcomp_data_quantiles.loc[non_null_mask, f"{var}_q"] = quantile_values

                        q_vars = [f"{var}_q" for var in pfas_analysis_vars if f"{var}_q" in qcomp_data_quantiles.columns]

                        if len(q_vars) > 0:
                            qcomp_data_quantiles['mixture_index'] = qcomp_data_quantiles[q_vars].sum(axis=1)

                            min_days = qcomp_data_quantiles['TD'].min()
                            if min_days < 0:
                                qcomp_data_quantiles['TD_positive'] = qcomp_data_quantiles['TD'] - min_days + 1
                            else:
                                qcomp_data_quantiles['TD_positive'] = qcomp_data_quantiles['TD'] + 1

                            available_confounders = [c for c in confounders if c in qcomp_data_quantiles.columns]

                            stage_confounders_file = os.path.join(stage_dir, f'{stage}_available_confounders.xlsx')
                            pd.DataFrame({'variable_raw': available_confounders}).to_excel(stage_confounders_file, index=False)

                            if available_confounders:
                                X = sm.add_constant(qcomp_data_quantiles[['mixture_index'] + available_confounders])
                            else:
                                X = sm.add_constant(qcomp_data_quantiles[['mixture_index']])

                            y = qcomp_data_quantiles['TD_positive']

                            family = Poisson()
                            groups = np.arange(len(X))
                            model = GEE(y, X, groups, family=family, cov_struct=Independence())
                            model_result = model.fit()

                            mixture_coef = model_result.params.get('mixture_index')
                            mixture_p = model_result.pvalues.get('mixture_index')
                            mixture_se = model_result.bse.get('mixture_index')

                            if mixture_coef is not None and mixture_se is not None:
                                mixture_ci_lower = mixture_coef - 1.96 * mixture_se
                                mixture_ci_upper = mixture_coef + 1.96 * mixture_se
                                mixture_hr = math.exp(mixture_coef)

                                print(f"Mixture index coefficient: {mixture_coef:.4f}, HR: {mixture_hr:.4f}, "
                                      f"p-value: {mixture_p:.4f}")

                                print("\nComputing weight contribution of each PFAS variable to the mixture effect...")
                                pfas_results = results_df[results_df['variable'].isin(pfas_analysis_vars)].copy()

                                if not pfas_results.empty:
                                    positive_coefs = pfas_results[pfas_results['coefficient'] > 0]['coefficient'].sum()
                                    negative_coefs = pfas_results[pfas_results['coefficient'] < 0]['coefficient'].sum()

                                    pfas_results['weight_positive'] = 0.0
                                    pfas_results['weight_negative'] = 0.0

                                    if positive_coefs > 0:
                                        pfas_results.loc[pfas_results['coefficient'] > 0, 'weight_positive'] = \
                                            pfas_results.loc[pfas_results['coefficient'] > 0, 'coefficient'] / positive_coefs

                                    if negative_coefs < 0:
                                        pfas_results.loc[pfas_results['coefficient'] < 0, 'weight_negative'] = \
                                            pfas_results.loc[pfas_results['coefficient'] < 0, 'coefficient'] / negative_coefs

                                    qcomp_results = {
                                        'mixture_coefficient': mixture_coef,
                                        'mixture_std_error': mixture_se,
                                        'mixture_p_value': mixture_p,
                                        'mixture_HR': np.exp(mixture_coef),
                                        'mixture_HR_ci_lower': np.exp(mixture_ci_lower),
                                        'mixture_HR_ci_upper': np.exp(mixture_ci_upper),
                                        'mixture_index_mean': qcomp_data_quantiles['mixture_index'].mean(),
                                        'mixture_index_std': qcomp_data_quantiles['mixture_index'].std(),
                                        'sample_size': len(qcomp_data_quantiles),
                                        'confounders_count': len(available_confounders),
                                        'confounders_list': ', '.join(available_confounders) if available_confounders else 'None'
                                    }

                                    qcomp_df = pd.DataFrame([qcomp_results])
                                    qcomp_file = os.path.join(stage_dir, f'{stage}_qcomp_results.xlsx')
                                    qcomp_df.to_excel(qcomp_file, index=False)
                                    print(f"{stage} qcomp results saved to: {qcomp_file}")

                                    weights_file = os.path.join(stage_dir, f'{stage}_qcomp_weights.xlsx')
                                    pfas_results[['variable_raw', 'coefficient', 'p_value', 'weight_positive', 'weight_negative']].to_excel(
                                        weights_file, index=False)
                                    print(f"{stage} PFAS weights saved to: {weights_file}")

                                    try:
                                        plt.figure(figsize=(12, 8))
                                        all_weights = pd.DataFrame()
                                        all_weights['variable'] = pfas_results['variable_raw']
                                        all_weights['weight'] = pfas_results['weight_positive'] + pfas_results['weight_negative']
                                        all_weights['direction'] = np.where(all_weights['weight'] >= 0, 'Positive', 'Negative')
                                        all_weights = all_weights.sort_values('weight', key=abs, ascending=False)

                                        colors = {'Positive': 'green', 'Negative': 'red'}
                                        sns.barplot(x='weight', y='variable', hue='direction', palette=colors, data=all_weights)

                                        plt.title(f'{stage}: PFAS variable weight contribution to mixture effect')
                                        plt.xlabel('Weight contribution')
                                        plt.ylabel('PFAS variable')
                                        plt.axvline(x=0, color='gray', linestyle='--')
                                        plt.tight_layout()

                                        weights_plot_file = os.path.join(stage_dir, f'{stage}_qcomp_weights_plot.png')
                                        plt.savefig(weights_plot_file, dpi=300, bbox_inches='tight')
                                        plt.close()
                                        print(f"{stage} PFAS weight contribution plot saved to: {weights_plot_file}")
                                    except Exception as e:
                                        print(f"Error plotting weight contribution: {e}")
            except Exception as e:
                print(f"Error running qcomp analysis: {e}")

    try:
        print("\nSummarizing qcomp results across all trimesters...")

        all_qcomp_results = []
        all_qcomp_weights = []

        for stage in pregnancy_stages:
            qcomp_file = os.path.join(analysis_dir, stage, f'{stage}_qcomp_results.xlsx')
            weights_file = os.path.join(analysis_dir, stage, f'{stage}_qcomp_weights.xlsx')

            if os.path.exists(qcomp_file):
                df_qcomp = pd.read_excel(qcomp_file)
                df_qcomp['trimester'] = stage
                all_qcomp_results.append(df_qcomp)

            if os.path.exists(weights_file):
                df_weights = pd.read_excel(weights_file)
                df_weights['trimester'] = stage
                all_qcomp_weights.append(df_weights)

        if all_qcomp_results:
            df_all_qcomp = pd.concat(all_qcomp_results, ignore_index=True)
            summary_file = os.path.join(analysis_dir, 'qcomp_results_summary.xlsx')
            df_all_qcomp.to_excel(summary_file, index=False)
            print(f"qcomp results summary saved to: {summary_file}")

        if all_qcomp_weights:
            df_all_weights = pd.concat(all_qcomp_weights, ignore_index=True)

            pivot_positive = df_all_weights.pivot_table(
                index='variable_raw', columns='trimester', values='weight_positive', fill_value=0)
            pivot_negative = df_all_weights.pivot_table(
                index='variable_raw', columns='trimester', values='weight_negative', fill_value=0)

            weights_summary_file = os.path.join(analysis_dir, 'qcomp_weights_summary.xlsx')
            with pd.ExcelWriter(weights_summary_file) as writer:
                df_all_weights.to_excel(writer, sheet_name='raw_data', index=False)
                pivot_positive.to_excel(writer, sheet_name='positive_weights')
                pivot_negative.to_excel(writer, sheet_name='negative_weights')
            print(f"PFAS weights summary saved to: {weights_summary_file}")

            try:
                pivot_all = pivot_positive + pivot_negative
                plt.figure(figsize=(12, 10))
                sns.heatmap(pivot_all, cmap='RdBu_r', center=0, annot=True, fmt='.2f')
                plt.title('PFAS variable weight contribution across trimesters')
                plt.tight_layout()

                heatmap_file = os.path.join(analysis_dir, 'qcomp_weights_heatmap.png')
                plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"PFAS weight contribution heatmap saved to: {heatmap_file}")
            except Exception as e:
                print(f"Error plotting weight contribution heatmap: {e}")
    except Exception as e:
        print(f"Error summarizing qcomp results: {e}")

    print(f"\nqcomp-based PFAS mixture effect analysis completed. Results saved to: {analysis_dir}")

    log_file = os.path.join(analysis_dir, 'variable_groups_info.txt')
    with open(log_file, 'w', encoding='utf-8') as f:
        selected_pfas = [var for var in pfas_vars if var in selected_vars]
        f.write(f"PFAS variables ({len(pfas_vars)}):\n" + '\n'.join(pfas_vars) + '\n\n')
        f.write(f"Selected PFAS variables ({len(selected_pfas)}):\n" + '\n'.join(selected_pfas) + '\n\n')

except FileNotFoundError:
    print("Error: file not found")
    sys.exit(1)
except Exception as e:
    print(f"Error processing data: {e}")
    sys.exit(1)
