import pandas as pd
import streamlit as st
from ai_model import SCORE_FEATURES, SCORE_TARGET, load_or_train_score, train_score_model

lang = st.session_state.get("lang", "kk")
L = lambda ru, kk: ru if lang == "ru" else kk

if "score_model" not in st.session_state:
    st.session_state.score_model, st.session_state.score_metrics = load_or_train_score()
model, metrics = st.session_state.score_model, st.session_state.score_metrics

st.title(L("📈 Прогноз результата по уровню", "📈 Деңгей бойынша нәтижені болжау"))
st.write(L("Модель машинного обучения (Gradient Boosting) обучена предсказывать итоговый балл (0–100) "
           "по уровню подготовки, времени занятий, числу попыток и прошлым успехам.",
           "Машиналық оқыту моделі (Gradient Boosting) дайындық деңгейі, оқу уақыты, әрекет саны және "
           "бұрынғы жетістіктер бойынша қорытынды балды (0–100) болжауға үйретілген."))

c1, c2 = st.columns(2)
level = c1.slider(L("Уровень подготовки (0–10)", "Дайындық деңгейі (0–10)"), 0.0, 10.0, 5.0, 0.5)
hours = c1.slider(L("Часов в неделю", "Аптасына сағат"), 1, 15, 5)
attempts = c2.slider(L("Число попыток", "Әрекет саны"), 1, 5, 2)
prev = c2.slider(L("Доля успешных модулей ранее", "Бұрын сәтті модульдер үлесі"), 0.0, 1.0, 0.5, 0.05)

x = pd.DataFrame([[level, hours, attempts, prev]], columns=SCORE_FEATURES)
st.metric(L("Прогнозируемый балл", "Болжамды балл"), f"{float(model.predict(x)[0]):.0f} / 100",
          help=L(f"Средняя ошибка модели ≈ {metrics['mae']} балла", f"Модельдің орташа қатесі ≈ {metrics['mae']} балл"))

curve = pd.DataFrame({"level": [i / 2 for i in range(21)]})
for c, v in zip(SCORE_FEATURES[1:], [hours, attempts, prev]):
    curve[c] = v
curve["score"] = model.predict(curve[SCORE_FEATURES])
st.subheader(L("Как меняется результат при росте уровня", "Деңгей өскенде нәтиже қалай өзгереді"))
st.line_chart(curve.set_index("level")["score"])

with st.expander(L("О модели", "Модель туралы")):
    m1, m2 = st.columns(2)
    m1.metric("MAE", metrics["mae"])
    m2.metric("R²", metrics["r2"])
    st.caption(f"train: {metrics['n_train']} · test: {metrics['n_test']}")

with st.expander(L("Переобучить на своих данных (CSV)", "Өз деректеріңізде қайта оқыту (CSV)")):
    st.caption(", ".join(SCORE_FEATURES + [SCORE_TARGET]))
    up = st.file_uploader("CSV", type=["csv"], key="scsv")
    if up and st.button(L("Обучить", "Оқыту")):
        data = pd.read_csv(up)
        miss = [c for c in SCORE_FEATURES + [SCORE_TARGET] if c not in data.columns]
        if miss:
            st.error((f"Бағандар жоқ: {miss}") if lang == "kk" else (f"Нет столбцов: {miss}"))
        else:
            st.session_state.score_model, st.session_state.score_metrics = train_score_model(data)
            st.rerun()
