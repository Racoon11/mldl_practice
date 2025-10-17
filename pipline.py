# churn_pipeline.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# === 1. Кастомный Feature Engineering Transformer ===
class TelecomFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, df):
        df = df.copy()

        # Группировка срока обслуживания
        # df['tenure_group'] = pd.cut(
        #     df['ClientPeriod'],
        #     bins=[-1, 12, 24, 60, 100],
        #     labels=['0-12', '13-24', '25-60', '60+']
        # )

        # Усреднённая месячная плата

        df['TotalSpent'] = df['TotalSpent'].replace(r'^\s*$',
                                                    np.nan, regex=True)
        df['TotalSpent'] = pd.to_numeric(df['TotalSpent'], errors='coerce')

        # Заполним пропуски медианой (можно и средним, но медиана устойчивее к выбросам)
        # df['TotalSpent'].fillna(df['TotalSpent'].median(), inplace=True)
        df.fillna({'TotalSpent': df['TotalSpent'].median()}, inplace=True)

        df['HasInternet'] = (df['HasInternetService'] != 'No').astype(int)
        df['NumAddServices'] = (df[['HasOnlineSecurityService',
                                    'HasOnlineBackup', 'HasDeviceProtection',
                                    'HasTechSupportAccess', 'HasOnlineTV',
                                    'HasMovieSubscription']] == 'Yes').sum(axis=1)
        df['IsLongTerm'] = (df['ClientPeriod'] > 48).astype(int)
        df['AvgMonthlyCharge'] = df['TotalSpent'] / (df['ClientPeriod'] + 1)

        cat_cols = df.select_dtypes(include=['object']).columns.tolist()

        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

        df = df.drop("TotalSpent", axis=1)
        return df


# === 2. Загрузка и разделение данных ===
def load_and_split_data(path: str, test_size: float = 0.2,
                        random_state: int = 42):
    df = pd.read_csv(path)
    y = df['Churn']
    X = df.drop(columns=['Churn'])
    return train_test_split(X, y, test_size=test_size, stratify=y,
                            random_state=random_state)


# === 3. Построение препроцессорного пайплайна ===
def build_preprocessor():
    fe = TelecomFeatureEngineer()
    return fe


def build_full_pipeline(model, use_scaler=True):
    """
    Собирает полный пайплайн: препроцессинг + балансировка + модель.
    Балансировка (SMOTE) применяется только при обучении.
    """
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE

    # Числовые и категориальные признаки определяются динамически после Feature Engineering
    num_pipeline = Pipeline([
        ('scaler', StandardScaler() if use_scaler else 'passthrough')
    ])

    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, lambda df: df.select_dtypes(include=[np.number]).columns.tolist()),
        ('cat', cat_pipeline, lambda df: df.select_dtypes(exclude=[np.number]).columns.tolist())
    ])

    return ImbPipeline([
        ('feature_engineering', TelecomFeatureEngineer()),
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])


# === 4. Обучение и оценка моделей ===
def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'LogisticRegression_L2': LogisticRegression(penalty='l2',
                                                    max_iter=1000,
                                                    random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'KNN': KNeighborsClassifier(n_jobs=-1)
    }

    results = {}
    pipelines = {}

    for name, model in models.items():
        use_scaler = name in ['LogisticRegression_L2', 'KNN']
        pipe = build_full_pipeline(model, use_scaler=use_scaler)
        pipe.fit(X_train, y_train)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        results[name] = auc
        pipelines[name] = pipe
        print(f"{name}: ROC-AUC = {auc:.4f}")

    return results, pipelines


# === 5. Настройка гиперпараметров для лучшей модели ===
def tune_best_model(X_train, y_train, best_model_name):
    if best_model_name == 'RandomForest':
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_dist = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__class_weight': ['balanced', None]
        }
    elif best_model_name == 'LogisticRegression_L2':
        model = LogisticRegression(penalty='l2', max_iter=2000,
                                   random_state=42)
        param_dist = {
            'classifier__C': [0.01, 0.1, 1, 10, 100]
        }
    else:
        return None

    pipe = build_full_pipeline(model,
                               use_scaler=(best_model_name != 'RandomForest'))
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=20,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        random_state=42
    )
    search.fit(X_train, y_train)
    return search


# === 6. Запуск всего пайплайна ===
if __name__ == "__main__":
    # 1. Загрузка
    X_train, X_test, y_train, y_test = load_and_split_data('train.csv')

    # 2. Базовая оценка
    results, pipelines = evaluate_models(X_train, X_test, y_train, y_test)

    # 3. Выбор и настройка лучшей модели
    best_name = max(results, key=results.get)
    best_search = tune_best_model(X_train, y_train, best_name)

    if best_search:
        print(f"\nЛучшие параметры для {best_name}:")
        print(best_search.best_params_)
        print(f"Лучший CV ROC-AUC: {best_search.best_score_:.4f}")

        # Финальная оценка на тесте
        y_pred_proba = best_search.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"Тестовый ROC-AUC после настройки: {test_auc:.4f}")

        # Анализ важности признаков (для деревьев)
        if best_name == 'RandomForest':
            feature_names = best_search.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
            importances = best_search.best_estimator_.named_steps['classifier'].feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            plt.figure(figsize=(10, 6))
            plt.title("Top 10 Feature Importances")
            plt.bar(range(10), importances[indices])
            plt.xticks(range(10), [feature_names[i] for i in indices],
                       rotation=45)
            plt.tight_layout()
            plt.show()
