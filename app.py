import streamlit as st

st.set_page_config(page_title="Адаптивті оқыту платформасы", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.hero {padding: 2rem 2.2rem; border-radius: 16px; margin-bottom: 1.2rem; color: #fff;
       background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);}
.hero h1 {color: #fff; font-size: 2rem; margin: 0 0 .6rem 0;}
.hero p {margin: .2rem 0; font-size: 1.05rem; opacity: .95;}
.hero .small {font-size: .9rem; opacity: .8;}
.card {border: 1px solid rgba(128,128,128,.3); border-radius: 14px; padding: 1.1rem 1.2rem;
       margin-bottom: .5rem; height: 100%;}
.card .icon {font-size: 1.8rem;}
.card h4 {margin: .3rem 0 .3rem 0;}
.card p {margin: .2rem 0; opacity: .85;}
.tag {display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: .78rem;
      background: rgba(59,130,246,.15); margin-right: 6px;}
.stars {color: #f59e0b; letter-spacing: 2px;}
</style>
""", unsafe_allow_html=True)

if "lang" not in st.session_state:
    st.session_state.lang = "kk"

with st.sidebar:
    st.radio("Тіл / Язык", ["kk", "ru"], key="lang", horizontal=True,
             format_func=lambda x: {"ru": "Русский", "kk": "Қазақша"}[x])
L = lambda ru, kk: ru if st.session_state.lang == "ru" else kk

nav = st.navigation([
    st.Page("views/home.py", title=L("Главная", "Басты бет"), icon="🏠", url_path="home", default=True),
    st.Page("views/adaptive_test.py", title=L("Адаптивный тест", "Адаптивті тест"), icon="🧭", url_path="test"),
    st.Page("views/ai_recommendations.py", title=L("ИИ-рекомендации", "ЖИ ұсыныстары"), icon="🤖", url_path="ai"),
    st.Page("views/score_prediction.py", title=L("Прогноз результата", "Нәтиже болжамы"), icon="📈", url_path="score"),
    st.Page("views/materials.py", title=L("Учебные модули", "Оқу модульдері"), icon="🧩", url_path="modules"),
])

with st.sidebar:
    st.divider()
    st.caption("**Жунусова Дильназ Жомарткызы**")
    st.caption(L("Магистрант 2 курса · ЕНУ им. Л.Н. Гумилёва", "2 курс магистранты · Л.Н. Гумилев атындағы ЕҰУ"))

nav.run()
