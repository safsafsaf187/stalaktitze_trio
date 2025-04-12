import streamlit as st
from streamlit_extras.let_it_rain import rain



# Заголовок приложения
st.image("main.gif", width=1500)




# HTML и CSS для анимированного разноцветного текста
html_code = """
<style>
@keyframes rainbow {
    0% { color: red; }
    14% { color: orange; }
    28% { color: yellow; }
    42% { color: green; }
    57% { color: blue; }
    71% { color: indigo; }
    85% { color: violet; }
    100% { color: red; }
}

.rainbow-text {
    font-size: 3em;
    font-weight: bold;
    text-align: center;
    animation: rainbow 5s infinite;
}
</style>

<div class="rainbow-text">Сегодня ты узнаешь все</div>
"""

# Отображение HTML в Streamlit
st.markdown(html_code, unsafe_allow_html=True)

def eye():
    rain(
        emoji="👁",
        font_size=100,
        falling_speed=6,
        animation_length="infinite",
    )

eye()