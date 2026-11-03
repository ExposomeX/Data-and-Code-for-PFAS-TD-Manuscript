import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

plt.rcParams['font.size'] = 12

prediction_dir = os.path.join('demo_results', 'pred_4_SG_T2')
os.makedirs(prediction_dir, exist_ok=True)

data_file = os.path.join('demo_data', 'df_prediction_T2.xlsx')
df = pd.read_excel(data_file)
print(f"Data shape: {df.shape}")

target_var = "TD"
print(f"Target variable summary:\n{df[target_var].describe()}")

missing_count = df[target_var].isna().sum()
if missing_count > 0:
    df = df.dropna(subset=[target_var])
    print(f"Dropped rows with missing target. Data shape: {df.shape}")

all_columns = df.columns.tolist()

try:
    pop_start, pop_end = "clin_systolic_blood_pressure", "epi_gravidity"
    start_idx, end_idx = all_columns.index(pop_start), all_columns.index(pop_end)
    population_vars = all_columns[start_idx:end_idx + 1]
except ValueError:
    population_vars = []
print(f"Population variables: {len(population_vars)}")

pfas_vars = [col for col in df.columns if col.startswith('pfas_')]
print(f"PFAS variables: {len(pfas_vars)}")

metabolite_vars = [col for col in df.columns if col.startswith('metabo_') and not col.endswith('_pfas')]
print(f"Metabolite variables: {len(metabolite_vars)}")

try:
    path_start, path_end = "Arginine biosynthesis_pfas", "One carbon pool by folate_pfas"
    start_idx, end_idx = all_columns.index(path_start), all_columns.index(path_end)
    pathway_vars = all_columns[start_idx:end_idx + 1]
except ValueError:
    pathway_vars = [col for col in df.columns if (("metabolism" in col.lower() or "biosynthesis" in col.lower()) and col.endswith("_pfas"))]
print(f"Pathway index variables: {len(pathway_vars)}")

metabolite_idx_vars = [col for col in df.columns if (col.startswith('C') and col.endswith('_pfas'))]
print(f"Metabolite index variables: {len(metabolite_idx_vars)}")


def evaluate_model(y_true, y_pred, gamma=1.0):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    within_10d = float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true)) <= 10) * 100)

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    y_true_mean, y_pred_mean = np.mean(y_true_arr), np.mean(y_pred_arr)
    denominator = np.sum((y_true_arr - y_true_mean) ** 2)
    if denominator > 0:
        beta = np.sum((y_true_arr - y_true_mean) * (y_pred_arr - y_pred_mean)) / denominator
        F = np.degrees(np.arctan(np.abs((beta - 1) / (beta + 1))))
    else:
        F = np.nan
    w2 = r2 * np.exp(-gamma * F / 90) if not np.isnan(F) else np.nan

    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'within_10d': within_10d, 'F': F, 'W2': w2}


class StackedGeneralizationModel:
    def __init__(self, base_models_dict, meta_model):
        self.base_models_dict = base_models_dict
        self.meta_model = meta_model
        self.base_models_trained = {}
        self.feature_importances = {}
        self.fold_metrics = {}

    def fit(self, X_dict, y, cv=5):
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        meta_features = np.zeros((len(y), len(self.base_models_dict)))
        self.feature_importances = {name: [] for name in self.base_models_dict.keys()}
        self.fold_metrics = {name: {'mse': [], 'rmse': [], 'mae': [], 'r2': [], 'within_10d': [], 'F': [], 'W2': []}
                              for name in self.base_models_dict.keys()}

        for i, (name, model) in enumerate(self.base_models_dict.items()):
            X = X_dict[name]
            if X.isna().any().any():
                X = X.fillna(X.mean())
            self.base_models_trained[name] = []

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model_clone = clone_model(model)
                model_clone.fit(X_train, y_train)
                val_pred = model_clone.predict(X_val)
                meta_features[val_idx, i] = val_pred
                self.base_models_trained[name].append(model_clone)

                if hasattr(model_clone, 'feature_importances_'):
                    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': model_clone.feature_importances_}).sort_values('Importance', ascending=False)
                    self.feature_importances[name].append(importance_df)

                metrics = evaluate_model(y_val, val_pred)
                for metric_name, metric_value in metrics.items():
                    self.fold_metrics[name][metric_name].append(metric_value)

        self.meta_model.fit(meta_features, y)
        return self

    def predict(self, X_dict):
        meta_features = np.zeros((X_dict[list(X_dict.keys())[0]].shape[0], len(self.base_models_dict)))
        for i, (name, models) in enumerate(self.base_models_trained.items()):
            X = X_dict[name]
            if X.isna().any().any():
                X = X.fillna(X.mean())
            preds = [model.predict(X) for model in models]
            meta_features[:, i] = np.mean(preds, axis=0)
        return self.meta_model.predict(meta_features)

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        for name, models in self.base_models_trained.items():
            model_dir = os.path.join(directory, f"base_model_{name}")
            os.makedirs(model_dir, exist_ok=True)
            for i, model in enumerate(models):
                joblib.dump(model, os.path.join(model_dir, f"fold_{i}.pkl"))
        joblib.dump(self.meta_model, os.path.join(directory, "meta_model.pkl"))


def clone_model(model):
    if isinstance(model, RandomForestRegressor):
        return RandomForestRegressor(
            n_estimators=model.n_estimators, max_depth=model.max_depth,
            min_samples_split=model.min_samples_split, min_samples_leaf=model.min_samples_leaf,
            random_state=model.random_state)
    return model.__class__(**model.get_params())


y = df[target_var].values
X_dict = {
    'population': df[population_vars],
    'pfas': df[pfas_vars],
    'metabolite': df[metabolite_vars],
    'pathway_index': df[pathway_vars],
    'metabolite_index': df[metabolite_idx_vars]
}

indices = np.arange(len(y))
train_indices, test_indices = train_test_split(indices, test_size=0.1, random_state=42)

model_combinations = [
    {'name': 'population', 'groups': ['population']},
    {'name': 'pfas', 'groups': ['pfas']},
    {'name': 'metabolite', 'groups': ['metabolite']},
    {'name': 'pathway_index', 'groups': ['pathway_index']},
    {'name': 'metabolite_index', 'groups': ['metabolite_index']},
    {'name': 'metabolite+pfas', 'groups': ['metabolite', 'pfas']},
    {'name': 'metabolite+metabolite_index', 'groups': ['metabolite', 'metabolite_index']},
    {'name': 'metabolite+pathway_index', 'groups': ['metabolite', 'pathway_index']},
    {'name': 'metabolite+pfas+metabolite_index', 'groups': ['metabolite', 'pfas', 'metabolite_index']},
    {'name': 'metabolite+pfas+pathway_index', 'groups': ['metabolite', 'pfas', 'pathway_index']},
    {'name': 'metabolite+pfas+pathway_index+metabolite_index', 'groups': ['metabolite', 'pfas', 'pathway_index', 'metabolite_index']},
    {'name': 'metabolite+pfas+pathway_index+metabolite_index+population', 'groups': ['metabolite', 'pfas', 'pathway_index', 'metabolite_index', 'population']}
]

results = []
stacked_model = None

for combo in model_combinations:
    print(f"\nTraining model combination: {combo['name']}")

    current_base_models = {group: RandomForestRegressor(n_estimators=100, random_state=42) for group in combo['groups']}
    current_X_dict_train = {group: X_dict[group].iloc[train_indices].copy() for group in combo['groups']}
    current_X_dict_test = {group: X_dict[group].iloc[test_indices].copy() for group in combo['groups']}

    meta_model = RandomForestRegressor(n_estimators=100, random_state=42)
    stacked_model = StackedGeneralizationModel(current_base_models, meta_model)

    y_train = y[train_indices]
    stacked_model.fit(current_X_dict_train, y_train, cv=5)

    model_dir = os.path.join(prediction_dir, f"stacked_model_{combo['name']}")
    stacked_model.save(model_dir)

    y_pred_train = stacked_model.predict(current_X_dict_train)
    metrics_train = evaluate_model(y_train, y_pred_train)

    y_test = y[test_indices]
    y_pred_test = stacked_model.predict(current_X_dict_test)
    metrics_test = evaluate_model(y_test, y_pred_test)
    print(f"  Internal: R2={metrics_train['r2']:.4f}, RMSE={metrics_train['rmse']:.4f} | "
          f"External: R2={metrics_test['r2']:.4f}, RMSE={metrics_test['rmse']:.4f}")

    fold_std_metrics = {}
    for group in combo['groups']:
        fold_std_metrics[group] = {m: np.std(stacked_model.fold_metrics[group][m]) for m in ['mse', 'rmse', 'mae', 'r2', 'within_10d', 'F', 'W2']}

    avg_std = {m: np.mean([fold_std_metrics[group][m] for group in combo['groups']]) for m in ['mse', 'rmse', 'mae', 'r2', 'within_10d', 'F', 'W2']}

    metrics_df = pd.DataFrame({
        'metric': ['MSE', 'RMSE', 'MAE', 'R2', 'Within10Days(%)', 'F(deg)', 'W2(gamma=1)'],
        'internal_value': [metrics_train[k] for k in ['mse', 'rmse', 'mae', 'r2', 'within_10d', 'F', 'W2']],
        'internal_std': [avg_std[k] for k in ['mse', 'rmse', 'mae', 'r2', 'within_10d', 'F', 'W2']],
        'external_value': [metrics_test[k] for k in ['mse', 'rmse', 'mae', 'r2', 'within_10d', 'F', 'W2']]
    })
    metrics_df.to_excel(os.path.join(prediction_dir, f"{combo['name']}_performance.xlsx"), index=False)

    fold_metrics_data = []
    for group in combo['groups']:
        for fold_idx in range(len(stacked_model.fold_metrics[group]['mse'])):
            fold_metrics_data.append({
                'group': group, 'fold': fold_idx + 1,
                'MSE': stacked_model.fold_metrics[group]['mse'][fold_idx],
                'RMSE': stacked_model.fold_metrics[group]['rmse'][fold_idx],
                'MAE': stacked_model.fold_metrics[group]['mae'][fold_idx],
                'R2': stacked_model.fold_metrics[group]['r2'][fold_idx],
                'Within10Days(%)': stacked_model.fold_metrics[group]['within_10d'][fold_idx],
                'F(deg)': stacked_model.fold_metrics[group]['F'][fold_idx],
                'W2(gamma=1)': stacked_model.fold_metrics[group]['W2'][fold_idx]
            })
    pd.DataFrame(fold_metrics_data).to_excel(os.path.join(prediction_dir, f"{combo['name']}_fold_metrics.xlsx"), index=False)

    for label, actual, pred, color in [('internal', y_train, y_pred_train, 'blue'), ('external', y_test, y_pred_test, 'orange')]:
        plt.figure(figsize=(10, 8))
        plt.scatter(actual, pred, alpha=0.5, color=color, label=label)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        r2_val = metrics_train['r2'] if label == 'internal' else metrics_test['r2']
        plt.title(f'{combo["name"]} {label} prediction (R2 = {r2_val:.4f})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(prediction_dir, f"{combo['name']}_{label}_prediction_scatter.png"), dpi=300, bbox_inches='tight')
        plt.close()
        pd.DataFrame({'actual': actual, 'predicted': pred}).to_excel(
            os.path.join(prediction_dir, f"{combo['name']}_{label}_prediction_scatter.xlsx"), index=False)

    with pd.ExcelWriter(os.path.join(prediction_dir, f"{combo['name']}_internal_external_prediction_scatter.xlsx")) as writer:
        pd.DataFrame({'actual': y_train, 'predicted': y_pred_train}).to_excel(writer, sheet_name='internal', index=False)
        pd.DataFrame({'actual': y_test, 'predicted': y_pred_test}).to_excel(writer, sheet_name='external', index=False)

    results.append({
        'combination': combo['name'],
        'internal_MSE': metrics_train['mse'], 'internal_MSE_STD': avg_std['mse'],
        'internal_RMSE': metrics_train['rmse'], 'internal_RMSE_STD': avg_std['rmse'],
        'internal_MAE': metrics_train['mae'], 'internal_MAE_STD': avg_std['mae'],
        'internal_R2': metrics_train['r2'], 'internal_R2_STD': avg_std['r2'],
        'internal_Within10Days': metrics_train['within_10d'], 'internal_Within10Days_STD': avg_std['within_10d'],
        'internal_F': metrics_train['F'], 'internal_F_STD': avg_std['F'],
        'internal_W2': metrics_train['W2'], 'internal_W2_STD': avg_std['W2'],
        'external_MSE': metrics_test['mse'], 'external_RMSE': metrics_test['rmse'],
        'external_MAE': metrics_test['mae'], 'external_R2': metrics_test['r2'],
        'external_Within10Days': metrics_test['within_10d'], 'external_F': metrics_test['F'],
        'external_W2': metrics_test['W2']
    })

    if combo['name'] == 'metabolite+pfas+pathway_index+metabolite_index':
        if hasattr(stacked_model.meta_model, 'feature_importances_'):
            meta_features_names = list(stacked_model.base_models_dict.keys())
            meta_importance = pd.DataFrame({'Feature': meta_features_names, 'Importance': stacked_model.meta_model.feature_importances_}).sort_values('Importance', ascending=False)
            plt.figure(figsize=(10, 6))
            plt.barh(meta_importance['Feature'][::-1], meta_importance['Importance'][::-1], color='orange')
            plt.title('Meta-model feature importance')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(os.path.join(prediction_dir, 'meta_model_feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
            meta_importance.to_excel(os.path.join(prediction_dir, 'meta_model_feature_importance.xlsx'), index=False)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('external_R2', ascending=False)
results_df.to_excel(os.path.join(prediction_dir, "combination_performance_comparison.xlsx"), index=False)

for group_name, importance_list in stacked_model.feature_importances.items():
    if len(importance_list) > 0:
        all_importances = pd.concat(importance_list)
        avg_importance = all_importances.groupby('Feature')['Importance'].mean().reset_index().sort_values('Importance', ascending=False)
        avg_importance.to_excel(os.path.join(prediction_dir, f'{group_name}_feature_importance.xlsx'), index=False)

        top_features = avg_importance.head(10) if group_name in ['metabolite_index', 'metabolite'] else avg_importance.head(20)
        colors = {'metabolite_index': '#FF9999', 'metabolite': '#FF9999', 'pathway_index': '#3366FF',
                  'pfas': '#66b3ff', 'population': '#98FB98'}
        plt.figure(figsize=(12, 8))
        plt.barh(top_features['Feature'][::-1], top_features['Importance'][::-1], color=colors.get(group_name, 'skyblue'))
        plt.title(f'Top feature importance: {group_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(prediction_dir, f'{group_name}_feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()

print("\nAll analyses completed!")
