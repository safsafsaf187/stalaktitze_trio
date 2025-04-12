import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # Добавлен импорт
from torch import nn
from streamlit_extras.let_it_rain import rain




def eye():
    rain(
        emoji="👁",
        font_size=100,
        falling_speed=6,
        animation_length="infinite",
    )

eye()

# Загрузка модели и токенизатора
class ToxicBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained('cointegrated/rubert-tiny-toxicity')
        for param in self.bert.parameters():
            param.requires_grad = False
        self.bert.classifier = nn.Linear(312, 1)
        for param in self.bert.classifier.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

st.image("maxon.gif", width=1000)
# Инициализация модели и загрузка весов
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ToxicBert()
model.load_state_dict(torch.load('toxic_bert.pt', map_location=torch.device(device)))
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny-toxicity")

# Функция для предсказания токсичности
def predict_toxicity(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        preds = torch.sigmoid(outputs).cpu().numpy()
    return "чел ты токсик" if preds[0][0] > 0.5 else "чел ты неженка"

# Создание интерфейса Streamlit

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

<div class="rainbow-text">КТО ТАМ ПИСАЛ ЕРУНДУ В КОММЕНТАРИЯХ?</div>
"""

# Отображение HTML в Streamlit
st.markdown(html_code, unsafe_allow_html=True)
st.write("проверим на токсичность?")

# Поле для ввода текста
user_input = st.text_area("Текст", "")

# Кнопка для обработки текста
if st.button("Проверить"):
    if user_input.strip() == "":
        st.error("Пожалуйста, введите текст.")
    else:
        result = predict_toxicity(user_input)
        st.write(f"Результат: **{result}**")