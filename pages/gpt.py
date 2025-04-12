import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from streamlit_extras.let_it_rain import rain

# Загрузка дообученной модели и токенизатора
@st.cache_resource
def load_model_and_tokenizer():
    model_name = "./fine_tuned_model2"  # Путь к дообученной модели
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Функция для генерации текста
def generate_conspiracy_theory(prompt, model, tokenizer, max_length=100, temperature=1.0, top_k=50, top_p=0.9):
    device = next(model.parameters()).device  # Определяем устройство (CPU или GPU)
    
    # Токенизация входного запроса
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Генерация текста
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True
    )
    
    # Декодирование результатов
    theory = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return theory

st.image("vadim.gif", width=1000)

# Заголовок приложения
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

<div class="rainbow-text">ПОРА ПРОЗРЕТЬ</div>
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

# Ввод текста
user_input = st.text_area("Введите ваш запрос:", "Например: 'Луна — это искусственный спутник'")

# Параметры генерации
max_length = st.slider("Максимальная длина текста", min_value=50, max_value=500, value=100)
temperature = st.slider("Температура", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
top_k = st.slider("Top-k", min_value=1, max_value=100, value=50)
top_p = st.slider("Top-p", min_value=0.1, max_value=1.0, value=0.9, step=0.1)

# Кнопка для генерации
if st.button("Сгенерировать теорию"):
    with st.spinner("Генерация..."):
        theory = generate_conspiracy_theory(
            prompt=user_input,
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    
    # Вывод результата
    st.subheader("Сгенерированная теория:")
    st.write(theory)


# import streamlit as st
# from theoriesgpt import generate_conspiracy_theory

# # Заголовок приложения
# st.title("Генератор конспирологических теорий")

# # Ввод текста
# user_input = st.text_area("Введите ваш запрос:", "Например: 'Луна — это искусственный спутник'")

# # Параметры генерации
# max_length = st.slider("Максимальная длина текста", min_value=50, max_value=500, value=100)
# num_generations = st.slider("Количество генераций", min_value=1, max_value=5, value=1)
# temperature = st.slider("Температура", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
# top_k = st.slider("Top-k", min_value=1, max_value=100, value=50)
# top_p = st.slider("Top-p", min_value=0.1, max_value=1.0, value=0.9, step=0.1)

# # Кнопка для генерации
# if st.button("Сгенерировать теорию"):
#     with st.spinner("Генерация..."):
#         theories = generate_conspiracy_theory(
#             prompt=user_input,
#             max_length=max_length,
#             num_return_sequences=num_generations,
#             temperature=temperature,
#             top_k=top_k,
#             top_p=top_p
#         )
    
#     # Вывод результатов
#     st.subheader("Сгенерированные теории:")
#     for i, theory in enumerate(theories, 1):
#         st.write(f"**Теория {i}:** {theory}")