import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
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

analysis_dir = os.path.join('demo_results', 'asso_2_multiple factor')
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
    print(f"Confounders: {confounders}")

    pfas_vars = [col for col in df_all.columns if col.startswith('pfas_')]
    print(f"Number of PFAS variables: {len(pfas_vars)}")

    metabolic_vars = [col for col in df_all.columns if col.startswith('metabo_')]
    print(f"Number of metabolite variables: {len(metabolic_vars)}")

    all_vars = pfas_vars + metabolic_vars + confounders + ['clin_pregnancy_trimester']

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
        selected_metabolic = [var for var in metabolic_vars if var in stage_results['selected_vars']]
        print(f"{stage}: selected {len(stage_results['selected_vars'])}/{len(all_vars)} variables "
              f"(PFAS {len(selected_pfas)}/{len(pfas_vars)}, metabolite {len(selected_metabolic)}/{len(metabolic_vars)})")

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
        if 'id_enrollment' in df_stage.columns:
            if 'id_enrollment' in df_std.columns:
                df_std = df_std.drop('id_enrollment', axis=1)
            df_std = pd.concat([df_stage['id_enrollment'].reset_index(drop=True),
                                 df_std.reset_index(drop=True)], axis=1)

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

        available_confounders = [c for c in confounders if c in df_std.columns]
        confounders_file = os.path.join(stage_dir, f'{stage}_available_confounders.xlsx')
        pd.DataFrame({'confounders': available_confounders}).to_excel(confounders_file, index=False)

        analysis_results = []

        for var in analysis_vars:
            if var not in df_std.columns:
                continue
            if df_std[var].count() < 10:
                continue

            result = {
                'variable': var, 'p_value': None, 'coefficient': None, 'std_error': None,
                'ci_lower': None, 'ci_upper': None, 'HR': None, 'HR_ci_lower': None,
                'HR_ci_upper': None, 'HR_IQR': None, 'HR_IQR_ci_lower': None, 'HR_IQR_ci_upper': None,
                'Q1': None, 'Q2': None, 'Q3': None, 'IQR': None, 'confounders': None
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

                available_confounders_var = [c for c in confounders if c in model_data.columns]
                if available_confounders_var:
                    result['confounders'] = available_confounders_var
                    X = sm.add_constant(model_data[[var] + available_confounders_var])
                else:
                    result['confounders'] = []
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
                          f"HR_IQR={result['HR_IQR']:.4f}, p={result['p_value']:.4f}")

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

            confounders_by_var_file = os.path.join(stage_dir, f'{stage}_confounders_by_variable.xlsx')
            confounders_by_var = []
            for _, row in results_df.iterrows():
                if isinstance(row['confounders'], list):
                    for confounder in row['confounders']:
                        confounders_by_var.append({'variable': row['variable'], 'confounder': confounder})
            if confounders_by_var:
                pd.DataFrame(confounders_by_var).to_excel(confounders_by_var_file, index=False)

            try:
                sig_results = results_df[results_df['p_value'] < 0.05].head(15)

                if not sig_results.empty:
                    var_names = sig_results['variable_raw'].tolist()
                    p_values = sig_results['p_value'].values

                    plt.figure(figsize=(12, 8))
                    plt.axvline(x=1, color='gray', linestyle='--')

                    hr_iqrs = sig_results['HR_IQR'].values
                    hr_iqr_ci_lowers = sig_results['HR_IQR_ci_lower'].values
                    hr_iqr_ci_uppers = sig_results['HR_IQR_ci_upper'].values

                    y_pos = np.arange(len(var_names))
                    plt.errorbar(hr_iqrs, y_pos, xerr=[hr_iqrs - hr_iqr_ci_lowers, hr_iqr_ci_uppers - hr_iqrs],
                                 fmt='o', capsize=5, elinewidth=2, markeredgewidth=2)

                    for i, (hr_iqr_val, p_val) in enumerate(zip(hr_iqrs, p_values)):
                        plt.text(hr_iqr_val + 0.1, i, f"HR_IQR={hr_iqr_val:.2f}, p={p_val:.4f}", va='center')

                    plt.yticks(y_pos, var_names)
                    plt.xlabel('HR_IQR (95% CI)')
                    plt.title(f'{stage}: significant factors of TD (adjusted GEE Poisson regression)')
                    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
                    plt.tight_layout()

                    forest_file = os.path.join(stage_dir, f'{stage}_forest_plot.png')
                    plt.savefig(forest_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"{stage} forest plot saved to: {forest_file}")
            except Exception as e:
                print(f"Error plotting forest plot for {stage}: {e}")

    print(f"\nMultivariable GEE Poisson regression analysis completed. Results saved to: {analysis_dir}")

    log_file = os.path.join(analysis_dir, 'variable_groups_info.txt')
    with open(log_file, 'w', encoding='utf-8') as f:
        selected_pfas = [var for var in pfas_vars if var in selected_vars]
        selected_metabolic = [var for var in metabolic_vars if var in selected_vars]

        f.write(f"PFAS variables ({len(pfas_vars)}):\n" + '\n'.join(pfas_vars) + '\n\n')
        f.write(f"Selected PFAS variables ({len(selected_pfas)}):\n" + '\n'.join(selected_pfas) + '\n\n')
        f.write(f"Metabolite variables ({len(metabolic_vars)}):\n" + '\n'.join(metabolic_vars) + '\n\n')
        f.write(f"Selected metabolite variables ({len(selected_metabolic)}):\n" + '\n'.join(selected_metabolic) + '\n\n')
        f.write(f"Confounders ({len(confounders)}):\n" + '\n'.join(confounders) + '\n')

    confounders_summary_file = os.path.join(analysis_dir, 'confounders_summary.xlsx')
    pd.DataFrame({'confounders': confounders}).to_excel(confounders_summary_file, index=False)

    stage_confounders = {}
    for stage in pregnancy_stages:
        stage_dir = os.path.join(analysis_dir, stage)
        confounders_file = os.path.join(stage_dir, f'{stage}_available_confounders.xlsx')
        if os.path.exists(confounders_file):
            try:
                df_stage_confounders = pd.read_excel(confounders_file)
                stage_confounders[stage] = df_stage_confounders['confounders'].tolist()
            except Exception as e:
                print(f"Error reading {stage} confounders file: {e}")
                stage_confounders[stage] = []
        else:
            stage_confounders[stage] = []

    all_stage_confounders = set()
    for confounders_list in stage_confounders.values():
        all_stage_confounders.update(confounders_list)

    comparison_data = []
    for confounder in sorted(all_stage_confounders):
        row = {'confounder': confounder}
        for stage in pregnancy_stages:
            row[stage] = 'Yes' if confounder in stage_confounders[stage] else 'No'
        comparison_data.append(row)

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_file = os.path.join(analysis_dir, 'confounders_comparison.xlsx')
        comparison_df.to_excel(comparison_file, index=False)

except FileNotFoundError:
    print("Error: file not found")
    sys.exit(1)
except Exception as e:
    print(f"Error processing data: {e}")
    sys.exit(1)
