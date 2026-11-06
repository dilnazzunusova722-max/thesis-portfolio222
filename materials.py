import streamlit as st
from ai_model import FORMATS, MODULES, tn

lang = st.session_state.get("lang", "kk")
L = lambda ru, kk: ru if lang == "ru" else kk
ICON = {"видео": "🎬", "практика": "🛠️", "текст": "📄"}
LEVEL = {1: L("Начальный", "Бастапқы"), 2: L("Средний", "Орта"), 3: L("Продвинутый", "Жоғары")}

st.title(L("🧩 Каталог учебных модулей", "🧩 Оқу модульдері каталогы"))
st.write(L("Эти модули использует адаптивный движок при построении траектории.",
           "Бұл модульдерді адаптивті механизм траектория құру кезінде пайдаланады."))

c1, c2 = st.columns(2)
topics = c1.multiselect(L("Тема", "Тақырып"), list(MODULES), default=list(MODULES), format_func=lambda t: tn(t, lang))
fmts = c2.multiselect(L("Формат", "Формат"), FORMATS, default=FORMATS, format_func=lambda f: tn(f, lang))

for topic in topics:
    st.subheader(tn(topic, lang))
    shown = [(t, d, f) for t, d, f in MODULES[topic] if f in fmts]
    if not shown:
        st.caption(L("Нет модулей по выбранным фильтрам.", "Таңдалған сүзгілер бойынша модуль жоқ."))
    for col, (title, d, fmt) in zip(st.columns(3), shown):
        col.markdown(
            f'<div class="card"><div class="icon">{ICON[fmt]}</div><h4>{tn(title, lang)}</h4>'
            f'<p><span class="tag">{tn(fmt, lang)}</span><span class="tag">{LEVEL[d]}</span></p>'
            f'<p class="stars">{"★" * d}{"☆" * (3 - d)}</p></div>', unsafe_allow_html=True)
