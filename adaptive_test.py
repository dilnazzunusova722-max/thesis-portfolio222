import pandas as pd
import streamlit as st
from ai_model import MODULES, load_or_train, localize, recommend, tn


lang = st.session_state.get("lang", "kk")
L = lambda ru, kk: ru if lang == "ru" else kk

TOPICS = list(MODULES)
N_STEPS = 8

# (тема, сложность, вопрос ru, вопрос kk, варианты ru, варианты kk, индекс верного)
O_OK = ["O(1)", "O(n)", "O(n²)", "O(log n)"]
Q = [
    (TOPICS[0], 1, "Сколько раз выполнится цикл for i in range(3)?", "for i in range(3) циклі неше рет орындалады?", ["2", "3", "4", "0"], ["2", "3", "4", "0"], 1),
    (TOPICS[0], 2, "Сложность линейного поиска в массиве?", "Массивтегі сызықтық іздеудің күрделілігі?", O_OK, O_OK, 1),
    (TOPICS[0], 3, "Сложность сортировки пузырьком в худшем случае?", "Көпіршікті сұрыптаудың ең нашар жағдайдағы күрделілігі?", ["O(n)", "O(n log n)", "O(n²)", "O(log n)"], ["O(n)", "O(n log n)", "O(n²)", "O(log n)"], 2),
    (TOPICS[1], 1, "Какой тип у значения 3.14 в Python?", "Python-да 3.14 мәні қандай типке жатады?", ["int", "float", "str", "bool"], ["int", "float", "str", "bool"], 1),
    (TOPICS[1], 2, "Что выведет print(7 // 2)?", "print(7 // 2) нені шығарады?", ["3.5", "3", "4", "2"], ["3.5", "3", "4", "2"], 1),
    (TOPICS[1], 3, "Что выведет print([i*i for i in range(4)])?", "print([i*i for i in range(4)]) нені шығарады?", ["[0, 1, 4, 9]", "[1, 4, 9, 16]", "[0, 1, 2, 3]", "[0, 2, 4, 6]"], ["[0, 1, 4, 9]", "[1, 4, 9, 16]", "[0, 1, 2, 3]", "[0, 2, 4, 6]"], 0),
    (TOPICS[2], 1, "Какая команда SQL выбирает данные из таблицы?", "Кестеден деректерді таңдайтын SQL пәрмені?", ["SELECT", "INSERT", "DELETE", "CREATE"], ["SELECT", "INSERT", "DELETE", "CREATE"], 0),
    (TOPICS[2], 2, "Что такое первичный ключ?", "Бастапқы кілт дегеніміз не?", ["Уникальный идентификатор записи", "Любой столбец", "Название таблицы", "Тип данных"], ["Жазбаның бірегей идентификаторы", "Кез келген баған", "Кесте атауы", "Деректер типі"], 0),
    (TOPICS[2], 3, "Какой JOIN возвращает только совпавшие строки обеих таблиц?", "Қай JOIN екі кестенің тек сәйкес жолдарын қайтарады?", ["INNER JOIN", "LEFT JOIN", "FULL JOIN", "CROSS JOIN"], ["INNER JOIN", "LEFT JOIN", "FULL JOIN", "CROSS JOIN"], 0),
    (TOPICS[3], 1, "Какой пароль самый надёжный?", "Қай құпиясөз ең сенімді?", ["12345678", "qwerty", "Tz7!rK9#mQ2v", "password"], ["12345678", "qwerty", "Tz7!rK9#mQ2v", "password"], 2),
    (TOPICS[3], 2, "Какая формула Excel суммирует диапазон A1:A5?", "A1:A5 ауқымын қосатын Excel формуласы?", ["=SUM(A1:A5)", "=COUNT(A1:A5)", "=IF(A1:A5)", "=MAX(A1:A5)"], ["=SUM(A1:A5)", "=COUNT(A1:A5)", "=IF(A1:A5)", "=MAX(A1:A5)"], 0),
    (TOPICS[3], 3, "Что такое фишинг?", "Фишинг дегеніміз не?", ["Выманивание данных через поддельные сайты и письма", "Ускорение интернета", "Сжатие файлов", "Резервное копирование"], ["Жалған сайттар мен хаттар арқылы деректерді алдап алу", "Интернетті жылдамдату", "Файлдарды сығу", "Резервтік көшірме жасау"], 0),
]


def reset():
    st.session_state.at = dict(step=0, theta={t: 5.0 for t in TOPICS}, asked=set(), correct=0, log=[])


def target_difficulty(theta):
    return 1 if theta < 3.5 else 2 if theta < 6.5 else 3


if "at" not in st.session_state:
    reset()
S = st.session_state.at

st.title(L("🧭 Адаптивный диагностический тест", "🧭 Адаптивті диагностикалық тест"))
st.write(L(
    "Сложность каждого следующего вопроса подбирается по вашим ответам: верный ответ усложняет задание, "
    "ошибка упрощает. После теста модель машинного обучения строит индивидуальную траекторию.",
    "Әр келесі сұрақтың күрделілігі жауаптарыңызға қарай таңдалады: дұрыс жауап тапсырманы күрделендіреді, "
    "қате жеңілдетеді. Тесттен кейін машиналық оқыту моделі дара траектория құрады."))

if S["step"] < N_STEPS:
    topic = TOPICS[S["step"] % len(TOPICS)]
    d = target_difficulty(S["theta"][topic])
    pool = [i for i, q in enumerate(Q) if q[0] == topic and i not in S["asked"]]
    qi = min(pool, key=lambda i: abs(Q[i][1] - d))
    _, qd, qru, qkk, oru, okk, ans = Q[qi]
    opts = oru if lang == "ru" else okk

    st.progress(S["step"] / N_STEPS)
    st.caption(L(f"Вопрос {S['step']+1} из {N_STEPS} · тема: {tn(topic, lang)} · текущая сложность: {qd} из 3",
                 f"{S['step']+1}-сұрақ / {N_STEPS} · тақырып: {tn(topic, lang)} · ағымдағы күрделілік: {qd} / 3"))
    st.subheader(L(qru, qkk))
    choice = st.radio(L("Ответ", "Жауап"), opts, index=None, key=f"q{S['step']}")
    if st.button(L("Ответить", "Жауап беру"), disabled=choice is None, type="primary"):
        ok = opts.index(choice) == ans
        S["theta"][topic] = min(10, max(0, S["theta"][topic] + (qd if ok else -(4 - qd))))
        S["correct"] += ok
        S["asked"].add(qi)
        S["log"].append({L("Тема", "Тақырып"): tn(topic, lang), L("Сложность", "Күрделілік"): qd,
                         L("Результат", "Нәтиже"): "✅" if ok else "❌"})
        S["step"] += 1
        st.rerun()
else:
    model, _ = load_or_train()
    st.success(L("Тест завершён!", "Тест аяқталды!"))
    st.subheader(L("Ваш профиль знаний", "Білім профиліңіз"))
    st.bar_chart(pd.Series({tn(k, lang): round(v, 1) for k, v in S["theta"].items()}))
    st.dataframe(pd.DataFrame(S["log"]), use_container_width=True)

    c1, c2 = st.columns(2)
    style = c1.radio(L("Предпочитаемый формат", "Қалаулы формат"), ["видео", "практика", "текст"], horizontal=True,
                   format_func=lambda f: tn(f, lang))
    hours = c2.slider(L("Часов в неделю", "Аптасына сағат саны"), 1, 15, 5)
    prev = S["correct"] / N_STEPS

    st.subheader(L("🤖 Траектория, построенная моделью машинного обучения",
                   "🤖 Машиналық оқыту моделі құрған траектория"))
    st.dataframe(localize(recommend(model, S["theta"], style, hours, prev), lang), use_container_width=True)
    st.caption(L("Рекомендуются модули с прогнозом успеха около 70%.",
                 "Табыс болжамы шамамен 70% модульдер ұсынылады."))
    if st.button(L("Пройти заново", "Қайта тапсыру")):
        reset()
        st.rerun()
