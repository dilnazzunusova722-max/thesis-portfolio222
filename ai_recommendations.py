import pandas as pd
import streamlit as st
from ai_model import FEATURES, MODULES, TARGET, load_or_train, localize, recommend, tn, train


T = {
    "ru": dict(title="🤖 ИИ: прогноз успешности и рекомендации", lang="Язык / Тіл",
               intro="Модель машинного обучения (Random Forest) предсказывает вероятность того, что учащийся успешно освоит модуль. "
                     "Рекомендуются модули с вероятностью около 70% — не слишком лёгкие и не слишком сложные (зона ближайшего развития).",
               profile="Профиль учащегося", fmt="Предпочитаемый формат", hours="Часов в неделю",
               prev="Доля успешно пройденных модулей ранее", go="Получить рекомендации",
               result="Рекомендованная траектория", model="О модели", imp="Важность признаков",
               retrain="Переобучить модель на своих данных (CSV)",
               csvhelp="Столбцы: " + ", ".join(FEATURES + [TARGET]),
               btn="Обучить", ok="Модель переобучена", fmts=["видео", "практика", "текст"]),
    "kk": dict(title="🤖 ЖИ: табыстылық болжамы және ұсыныстар", lang="Язык / Тіл",
               intro="Машиналық оқыту моделі (Random Forest) оқушының модульді сәтті меңгеру ықтималдығын болжайды. "
                     "Ықтималдығы шамамен 70% модульдер ұсынылады: тым оңай да, тым қиын да емес (жақын даму аймағы).",
               profile="Оқушы профилі", fmt="Қалаулы формат", hours="Аптасына сағат саны",
               prev="Бұрын сәтті өткен модульдер үлесі", go="Ұсыныстар алу",
               result="Ұсынылған траектория", model="Модель туралы", imp="Белгілердің маңыздылығы",
               retrain="Моделді өз деректеріңізде қайта оқыту (CSV)",
               csvhelp="Бағандар: " + ", ".join(FEATURES + [TARGET]),
               btn="Оқыту", ok="Модель қайта оқытылды", fmts=["видео", "практика", "текст"]),
}
lang = st.session_state.get("lang", "kk")
t = T[lang]

if "model" not in st.session_state:
    st.session_state.model, st.session_state.metrics = load_or_train()
model, metrics = st.session_state.model, st.session_state.metrics

st.title(t["title"])
st.write(t["intro"])

st.subheader(t["profile"])
cols = st.columns(2)
scores = {}
for i, topic in enumerate(MODULES):
    scores[topic] = cols[i % 2].slider(tn(topic, lang), 0, 10, 4)
style = st.radio(t["fmt"], t["fmts"], horizontal=True, format_func=lambda f: tn(f, lang))
hours = st.slider(t["hours"], 1, 15, 5)
prev = st.slider(t["prev"], 0.0, 1.0, 0.5, 0.05)

if st.button(t["go"], type="primary"):
    st.subheader(t["result"])
    st.dataframe(localize(recommend(model, scores, style, hours, prev), lang), use_container_width=True)

with st.expander(t["model"]):
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", metrics["accuracy"])
    c2.metric("ROC-AUC", metrics["roc_auc"])
    st.caption(f"train: {metrics['n_train']} · test: {metrics['n_test']}")
    st.write(t["imp"])
    st.bar_chart(pd.Series(metrics["importances"]))

with st.expander(t["retrain"]):
    st.caption(t["csvhelp"])
    up = st.file_uploader("CSV", type=["csv"])
    if up and st.button(t["btn"]):
        data = pd.read_csv(up)
        miss = [c for c in FEATURES + [TARGET] if c not in data.columns]
        if miss:
            st.error((f"Бағандар жоқ: {miss}") if lang == "kk" else (f"Нет столбцов: {miss}"))
        else:
            st.session_state.model, st.session_state.metrics = train(data)
            st.success(t["ok"])
            st.rerun()
