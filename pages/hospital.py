import streamlit as st
import torch
import joblib
from sklearn.pipeline import Pipeline
from pymorphy3 import MorphAnalyzer
from nltk.corpus import stopwords
import string
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from streamlit_extras.let_it_rain import rain
from typing import Tuple
import numpy as np
import torch.nn.functional as F  # Добавьте этот импорт


# Функция для анимации дождя
def eye():
    rain(
        emoji="👁",
        font_size=100,
        falling_speed=6,
        animation_length="infinite",
    )

eye()


# Отображение изображения
st.image("savelyi.gif", width=1000)

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

<div class="rainbow-text">ВСЕ ОТЗЫВЫ ПО СВОЕМУ ПОЗИТИВНЫЕ!</div>
"""

# Отображение HTML в Streamlit
st.markdown(html_code, unsafe_allow_html=True)


def padding(review_int: list, seq_len: int) -> np.array:
    """
    Добавляет паддинг к последовательностям.
    :param review_int: Список индексов слов.
    :param seq_len: Максимальная длина последовательности.
    :return: Массив с паддингом.
    """
    features = np.zeros((len(review_int), seq_len), dtype=int)
    for i, review in enumerate(review_int):
        if len(review) <= seq_len:
            zeros = list(np.zeros(seq_len - len(review)))
            new = zeros + review
        else:
            new = review[:seq_len]
        features[i, :] = np.array(new)
    return features

# Определение класса для механизма внимания Bahdanau
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out, h_n):
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        return context_vector, attention_weights


# Определение класса для LSTM с механизмом внимания
class LSTMBahdanauAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
        super(LSTMBahdanauAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.attn = BahdanauAttention(hidden_size)
        self.clf = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(128, num_classes),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        embeddings = self.embedding(x)
        outputs, (h_n, _) = self.lstm(embeddings)
        context, att_weights = self.attn(outputs, h_n.squeeze(0))
        out = self.clf(context)
        return out, att_weights  # Возвращаем два значения


# Загрузка моделей
from typing import Tuple  # Добавьте этот импорт
import numpy as np

@st.cache_resource
def load_models():
    bow_pipeline = None
    tfidf_pipeline = None
    bert_model = None
    lstm_model = None
    tokenizer = None
    word_to_int = None

    try:
        # Загрузка pipeline для Bag-of-Words + Logistic Regression
        bow_pipeline = joblib.load("bow_pipeline.joblib")
    except Exception as e:
        st.error(f"Ошибка при загрузке модели Bag-of-Words: {e}")

    try:
        # Загрузка pipeline для TF-IDF + Logistic Regression
        tfidf_pipeline = joblib.load("tfidf_pipeline.joblib")
    except Exception as e:
        st.error(f"Ошибка при загрузке модели TF-IDF: {e}")

    try:
        # Определение пользовательской модели MyTinyBERT
        class MyTinyBERT(nn.Module):
            def __init__(self):
                super().__init__()
                self.bert = BertModel.from_pretrained("cointegrated/rubert-tiny2")
                for param in self.bert.parameters():
                    param.requires_grad = False
                self.linear = nn.Sequential(
                    nn.Linear(312, 256),
                    nn.Sigmoid(),
                    nn.Dropout(),
                    nn.Linear(256, 2)  # NUMBER_OF_CLASSES = 2
                )

            def forward(self, x):
                bert_out = self.bert(x[0], attention_mask=x[1])
                normed_bert_out = nn.functional.normalize(bert_out.last_hidden_state[:, 0, :])
                out = self.linear(normed_bert_out)
                return out

        # Создаем экземпляр модели BERT
        bert_model = MyTinyBERT()
        bert_model.eval()

        # Загружаем сохраненные веса
        state_dict = torch.load("bert4.pt", map_location=torch.device('cpu'), weights_only=True)
        bert_model.load_state_dict(state_dict)

    except Exception as e:
        st.error(f"Ошибка при загрузке модели BERT: {e}")

    try:
        # Загрузка токенизатора для BERT
        tokenizer = BertTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    except Exception as e:
        st.error(f"Ошибка при загрузке токенизатора BERT: {e}")

    try:
        # Определение модели LSTM
        class BahdanauAttention(nn.Module):
            def __init__(self, hidden_size: int):
                super().__init__()
                self.W_q = nn.Linear(hidden_size, hidden_size)
                self.W_k = nn.Linear(hidden_size, hidden_size)
                self.W_v = nn.Linear(hidden_size, 1)

            def forward(self, keys: torch.Tensor, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                query = self.W_q(query)
                keys = self.W_k(keys)
                energy = self.W_v(F.tanh(query.unsqueeze(1) + keys)).squeeze(-1)
                att_weights = torch.softmax(energy, -1)
                context = torch.bmm(att_weights.unsqueeze(1), keys)
                return context, att_weights

        class LSTMBahdanauAttention(nn.Module):
            def __init__(self, vocab_size, embedding_dim, hidden_size):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
                self.attn = BahdanauAttention(hidden_size)
                self.clf = nn.Sequential(
                    nn.Linear(hidden_size, 128),
                    nn.Dropout(0.2),
                    nn.Tanh(),
                    nn.Linear(128, 1),
                    nn.Dropout(0.1)
                )

            def forward(self, x):
                embeddings = self.embedding(x)
                outputs, (h_n, _) = self.lstm(embeddings)
                context, att_weights = self.attn(outputs, h_n.squeeze(0))
                out = self.clf(context)
                return out, att_weights  # Возвращаем два значения

        # Загрузка словаря для LSTM
        word_to_int = joblib.load("word_to_int.joblib")

        # Создаем экземпляр модели LSTM
        VOCAB_SIZE = len(word_to_int) + 1  # Укажите реальный размер словаря
        EMBEDDING_DIM = 32
        HIDDEN_SIZE = 32
        lstm_model = LSTMBahdanauAttention(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE)
        lstm_model.load_state_dict(torch.load("lstm_model_weights.pt", map_location=torch.device('cpu')))
        lstm_model.eval()

    except Exception as e:
        st.error(f"Ошибка при загрузке модели LSTM: {e}")

    try:
        # Загрузка словаря для LSTM
        word_to_int = joblib.load("word_to_int.joblib")
    except Exception as e:
        st.error(f"Ошибка при загрузке словаря word_to_int: {e}")

    return bow_pipeline, tfidf_pipeline, bert_model, lstm_model, tokenizer, word_to_int


# Функция для очистки текста
def clean_text(text):
    morph = MorphAnalyzer()
    stop_words = set(stopwords.words('russian'))

    if not isinstance(text, str):  # Проверяем, что текст строка
        return ""

    text = text.lower()  # Приводим к нижнему регистру
    text = text.translate(str.maketrans('', '', string.punctuation))  # Удаляем пунктуацию
    tokens = text.split()  # Разбиваем на слова
    tokens = [morph.parse(word)[0].normal_form for word in tokens if word not in stop_words]  # Лемматизация
    tokens = [word for word in tokens if len(word) > 1]  # Убираем слишком короткие слова
    cleaned_text = " ".join(tokens).strip()  # Убираем лишние пробелы
    return cleaned_text if cleaned_text else None


# Предсказание с использованием моделей
def predict_sentiment(text, bow_pipeline, tfidf_pipeline, bert_model, lstm_model, tokenizer, word_to_int):
    predictions = {}

    if bow_pipeline is not None and tfidf_pipeline is not None:
        # Очистка текста
        cleaned_text = clean_text(text)

        # Предсказание с помощью Bag-of-Words + Logistic Regression
        bow_prediction = bow_pipeline.predict([cleaned_text])[0]
        predictions["Bag-of-Words"] = "Позитивный" if bow_prediction == 1 else "Негативный"

        # Предсказание с помощью TF-IDF + Logistic Regression
        tfidf_prediction = tfidf_pipeline.predict([cleaned_text])[0]
        predictions["TF-IDF"] = "Позитивный" if tfidf_prediction == 1 else "Негативный"

    if bert_model is not None and tokenizer is not None:
        # Токенизация текста для BERT
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = bert_model((inputs['input_ids'], inputs['attention_mask']))
        logits = outputs
        probs = torch.softmax(logits, dim=1)
        bert_prediction = torch.argmax(probs, dim=1).item()
        predictions["BERT"] = "Позитивный" if bert_prediction == 1 else "Негативный"

    if lstm_model is not None and word_to_int is not None:
        # Преобразование текста в последовательность индексов для LSTM
        tokens = clean_text(text).split()
        sequence = [word_to_int.get(token, 0) for token in tokens]  # 0 - индекс для неизвестных слов
        sequence = padding([sequence], seq_len=64)[0]  # Применяем паддинг
        sequence = torch.tensor(sequence).unsqueeze(0)  # Добавляем размерность батча
        with torch.no_grad():
            lstm_output, _ = lstm_model(sequence)  # Получаем выход и игнорируем веса внимания
        lstm_prediction = torch.sigmoid(lstm_output).squeeze().item() > 0.5
        predictions["LSTM"] = "Позитивный" if lstm_prediction else "Негативный"

    return predictions
   


# Основная функция Streamlit
def main():
    st.title("Анализ тональности отзывов")
    st.write("Введите текст отзыва, чтобы определить его тональность.")

    # Поле для ввода текста
    user_input = st.text_area("Введите текст отзыва:", "")

    # Кнопка для отправки текста
    if st.button("Определить тональность"):
        if user_input.strip() == "":
            st.error("Пожалуйста, введите текст отзыва.")
        else:
            try:
                # Загрузка моделей и словаря
                bow_pipeline, tfidf_pipeline, bert_model, lstm_model, tokenizer, word_to_int = load_models()

                # Получение предсказаний
                predictions = predict_sentiment(
                    text=user_input,
                    bow_pipeline=bow_pipeline,
                    tfidf_pipeline=tfidf_pipeline,
                    bert_model=bert_model,
                    lstm_model=lstm_model,
                    tokenizer=tokenizer,
                    word_to_int=word_to_int
                )

                # Вывод результатов
                st.subheader("Результаты анализа:")
                for model_name, sentiment in predictions.items():
                    st.write(f"{model_name}: {sentiment}")

            except Exception as e:
                st.error(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()




# import streamlit as st
# import torch
# import joblib
# from sklearn.pipeline import Pipeline
# from pymorphy3 import MorphAnalyzer
# from nltk.corpus import stopwords
# import string
# import torch.nn as nn
# from transformers import BertModel
# from streamlit_extras.let_it_rain import rain


# # Функция для анимации дождя
# def eye():
#     rain(
#         emoji="👁",
#         font_size=100,
#         falling_speed=6,
#         animation_length="infinite",
#     )

# eye()


# # Отображение изображения
# st.image("savelyi.gif", width=1000)

# # HTML и CSS для анимированного разноцветного текста
# html_code = """
# <style>
# @keyframes rainbow {
#     0% { color: red; }
#     14% { color: orange; }
#     28% { color: yellow; }
#     42% { color: green; }
#     57% { color: blue; }
#     71% { color: indigo; }
#     85% { color: violet; }
#     100% { color: red; }
# }

# .rainbow-text {
#     font-size: 3em;
#     font-weight: bold;
#     text-align: center;
#     animation: rainbow 5s infinite;
# }
# </style>

# <div class="rainbow-text">ВСЕ ОТЗЫВЫ ПО СВОЕМУ ПОЗИТИВНЫЕ!</div>
# """

# # Отображение HTML в Streamlit
# st.markdown(html_code, unsafe_allow_html=True)


# # Определение класса для механизма внимания Bahdanau
# class BahdanauAttention(nn.Module):
#     def __init__(self, hidden_size):
#         super(BahdanauAttention, self).__init__()
#         self.attention = nn.Linear(hidden_size, 1)

#     def forward(self, lstm_out, h_n):
#         attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
#         context_vector = torch.sum(attention_weights * lstm_out, dim=1)
#         return context_vector, attention_weights


# # Определение класса для LSTM с механизмом внимания
# class LSTMBahdanauAttention(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes):
#         super(LSTMBahdanauAttention, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
#         self.attn = BahdanauAttention(hidden_size)
#         self.clf = nn.Sequential(
#             nn.Linear(hidden_size, 128),
#             nn.Dropout(0.2),
#             nn.Tanh(),
#             nn.Linear(128, num_classes),
#             nn.Dropout(0.1)
#         )

#     def forward(self, x):
#         embeddings = self.embedding(x)
#         outputs, (h_n, _) = self.lstm(embeddings)
#         context, att_weights = self.attn(outputs, h_n.squeeze(0))
#         out = self.clf(context)
#         return out, att_weights


# # Загрузка моделей
# @st.cache_resource
# def load_models():
#     bow_pipeline = None
#     tfidf_pipeline = None

#     try:
#         # Загрузка pipeline для Bag-of-Words + Logistic Regression
#         bow_pipeline = joblib.load("bow_pipeline.joblib")
#     except Exception as e:
#         st.error(f"Ошибка при загрузке модели Bag-of-Words: {e}")

#     try:
#         # Загрузка pipeline для TF-IDF + Logistic Regression
#         tfidf_pipeline = joblib.load("tfidf_pipeline.joblib")
#     except Exception as e:
#         st.error(f"Ошибка при загрузке модели TF-IDF: {e}")

#     return bow_pipeline, tfidf_pipeline


# # Функция для очистки текста
# def clean_text(text):
#     morph = MorphAnalyzer()
#     stop_words = set(stopwords.words('russian'))

#     if not isinstance(text, str):  # Проверяем, что текст строка
#         return ""

#     text = text.lower()  # Приводим к нижнему регистру
#     text = text.translate(str.maketrans('', '', string.punctuation))  # Удаляем пунктуацию
#     tokens = text.split()  # Разбиваем на слова
#     tokens = [morph.parse(word)[0].normal_form for word in tokens if word not in stop_words]  # Лемматизация
#     tokens = [word for word in tokens if len(word) > 1]  # Убираем слишком короткие слова
#     cleaned_text = " ".join(tokens).strip()  # Убираем лишние пробелы
#     return cleaned_text if cleaned_text else None


# # Предсказание с использованием моделей
# def predict_sentiment(text, bow_pipeline, tfidf_pipeline):
#     if bow_pipeline is None or tfidf_pipeline is None:
#         st.error("Модели не загружены. Пожалуйста, проверьте файлы моделей.")
#         return {}

#     # Очистка текста
#     cleaned_text = clean_text(text)

#     # Предсказание с помощью Bag-of-Words + Logistic Regression
#     bow_prediction = bow_pipeline.predict([cleaned_text])[0]

#     # Предсказание с помощью TF-IDF + Logistic Regression
#     tfidf_prediction = tfidf_pipeline.predict([cleaned_text])[0]

#     # Возвращаем результаты
#     return {
#         "Bag-of-Words": "Позитивный" if bow_prediction == 1 else "Негативный",
#         "TF-IDF": "Позитивный" if tfidf_prediction == 1 else "Негативный"
#     }


# # Основная функция Streamlit
# def main():
#     st.title("Анализ тональности отзывов")
#     st.write("Введите текст отзыва, чтобы определить его тональность.")

#     # Поле для ввода текста
#     user_input = st.text_area("Введите текст отзыва:", "")

#     # Кнопка для отправки текста
#     if st.button("Определить тональность"):
#         if user_input.strip() == "":
#             st.error("Пожалуйста, введите текст отзыва.")
#         else:
#             # Загрузка моделей
#             bow_pipeline, tfidf_pipeline = load_models()

#             # Получение предсказаний
#             predictions = predict_sentiment(user_input, bow_pipeline, tfidf_pipeline)

#             # Вывод результатов
#             st.subheader("Результаты анализа:")
#             for model_name, sentiment in predictions.items():
#                 st.write(f"{model_name}: {sentiment}")


# if __name__ == "__main__":
#     main()


