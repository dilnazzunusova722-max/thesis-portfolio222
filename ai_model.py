"""ИИ-модуль: обучение и применение модели прогноза успешности освоения модуля."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model.joblib"
METRICS_PATH = BASE / "metrics.json"

# Признаки: уровень по теме (0-10), сложность модуля (1-3), совпадение формата (0/1),
# часов в неделю, доля успешных модулей в прошлом (0-1). Цель: success (0/1).
FEATURES = ["topic_score", "difficulty", "format_match", "hours_per_week", "prev_success"]
TARGET = "success"

MODULES = {
    "Алгоритмическое мышление": [("Понятие алгоритма", 1, "видео"), ("Блок-схемы", 2, "практика"), ("Оценка сложности", 3, "текст")],
    "Программирование": [("Переменные и типы", 1, "видео"), ("Циклы и условия", 2, "практика"), ("Рекурсия", 3, "текст")],
    "Данные и БД": [("Таблицы и ключи", 1, "текст"), ("Запросы SELECT", 2, "практика"), ("Соединения JOIN", 3, "видео")],
    "Цифровая грамотность": [("Безопасность в сети", 1, "видео"), ("Работа с таблицами", 2, "практика"), ("Анализ данных", 3, "текст")],
}


def make_synthetic_data(n=3000, seed=42):
    """Синтетические данные для демонстрации. В диссертации замените реальными."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "topic_score": rng.integers(0, 11, n),
        "difficulty": rng.integers(1, 4, n),
        "format_match": rng.integers(0, 2, n),
        "hours_per_week": rng.integers(1, 16, n),
        "prev_success": rng.uniform(0, 1, n).round(2),
    })
    z = (0.6 * (df.topic_score - 3.5 * df.difficulty) + 0.5 + 0.6 * df.format_match
         + 0.08 * df.hours_per_week + 1.5 * (df.prev_success - 0.5) + rng.normal(0, 0.8, n))
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df


def train(df=None):
    """Обучает модель, сохраняет её и метрики. Возвращает (model, metrics)."""
    df = make_synthetic_data() if df is None else df
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.25, random_state=42, stratify=df[TARGET])
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(y_te, proba > 0.5)), 3),
        "roc_auc": round(float(roc_auc_score(y_te, proba)), 3),
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "importances": {f: round(float(i), 3) for f, i in zip(FEATURES, model.feature_importances_)},
    }
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return model, metrics


def load_or_train():
    if MODEL_PATH.exists() and METRICS_PATH.exists():
        return joblib.load(MODEL_PATH), json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return train()


def recommend(model, scores, style, hours, prev_success, target_p=0.7, top_n=6):
    """Ранжирует модули: лучшие те, где прогноз успеха ближе к целевому (зона ближайшего развития)."""
    rows = []
    for topic, mods in MODULES.items():
        for title, diff, fmt in mods:
            rows.append({"Тема": topic, "Модуль": title, "Сложность": diff, "Формат": fmt,
                         "topic_score": scores[topic], "difficulty": diff,
                         "format_match": int(fmt == style), "hours_per_week": hours,
                         "prev_success": prev_success})
    df = pd.DataFrame(rows)
    df["Вероятность успеха"] = model.predict_proba(df[FEATURES])[:, 1].round(2)
    df["_dist"] = (df["Вероятность успеха"] - target_p).abs()
    out = df.sort_values("_dist").head(top_n).reset_index(drop=True)
    out.index += 1
    return out[["Тема", "Модуль", "Сложность", "Формат", "Вероятность успеха"]]




# ======================================================================
# Регрессия: прогноз итогового балла (0-100) по уровню и другим признакам
# CSV с реальными данными: level, hours_per_week, attempts, prev_success, score
# ======================================================================
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

SCORE_FEATURES = ["level", "hours_per_week", "attempts", "prev_success"]
SCORE_TARGET = "score"
SCORE_MODEL_PATH = BASE / "score_model.joblib"
SCORE_METRICS_PATH = BASE / "score_metrics.json"


def make_score_data(n=1500, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "level": rng.uniform(0, 10, n).round(1),
        "hours_per_week": rng.integers(1, 16, n),
        "attempts": rng.integers(1, 6, n),
        "prev_success": rng.uniform(0, 1, n).round(2),
    })
    s = (20 + 5.5 * df.level + 1.0 * df.hours_per_week + 12 * df.prev_success
         + 1.5 * np.log1p(df.attempts) + rng.normal(0, 5, n))
    df[SCORE_TARGET] = s.clip(0, 100).round(1)
    return df


def train_score_model(df=None):
    df = make_score_data() if df is None else df
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[SCORE_FEATURES], df[SCORE_TARGET], test_size=0.25, random_state=42)
    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    metrics = {"mae": round(float(mean_absolute_error(y_te, pred)), 2),
               "r2": round(float(r2_score(y_te, pred)), 3),
               "n_train": int(len(X_tr)), "n_test": int(len(X_te))}
    joblib.dump(model, SCORE_MODEL_PATH)
    SCORE_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return model, metrics


def load_or_train_score():
    if SCORE_MODEL_PATH.exists() and SCORE_METRICS_PATH.exists():
        return joblib.load(SCORE_MODEL_PATH), json.loads(SCORE_METRICS_PATH.read_text(encoding="utf-8"))
    return train_score_model()


# ---------------- Локализация (русский / казахский) ----------------
KK = {
    "Алгоритмическое мышление": "Алгоритмдік ойлау", "Программирование": "Бағдарламалау",
    "Данные и БД": "Деректер және ДҚ", "Цифровая грамотность": "Цифрлық сауаттылық",
    "Понятие алгоритма": "Алгоритм ұғымы", "Блок-схемы": "Блок-сызбалар",
    "Оценка сложности": "Күрделілікті бағалау", "Переменные и типы": "Айнымалылар және типтер",
    "Циклы и условия": "Циклдар мен шарттар", "Рекурсия": "Рекурсия",
    "Таблицы и ключи": "Кестелер мен кілттер", "Запросы SELECT": "SELECT сұраныстары",
    "Соединения JOIN": "JOIN байланыстары", "Безопасность в сети": "Желідегі қауіпсіздік",
    "Работа с таблицами": "Кестелермен жұмыс", "Анализ данных": "Деректерді талдау",
    "видео": "бейне", "практика": "практика", "текст": "мәтін",
}
FORMATS = ["видео", "практика", "текст"]


def tn(value, lang):
    """Перевод названия темы, модуля или формата."""
    return KK.get(value, value) if lang == "kk" else value


def localize(df, lang):
    if lang != "kk":
        return df
    df = df.copy()
    for c in ("Тема", "Модуль", "Формат"):
        df[c] = df[c].map(lambda v: KK.get(v, v))
    return df.rename(columns={"Тема": "Тақырып", "Сложность": "Күрделілік",
                              "Вероятность успеха": "Табыс ықтималдығы"})


if __name__ == "__main__":
    print(train()[1])
    print(train_score_model()[1])
