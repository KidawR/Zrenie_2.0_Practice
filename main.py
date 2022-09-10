import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.layers import Dense, SimpleRNN, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from fuzzy_logic.terms import Term
from fuzzy_logic.variables import FuzzyVariable
from fuzzy_logic.mamdani_fs import MamdaniFuzzySystem
from fuzzy_logic.mf import TriangularMF
# Выгружаем высказывания из файлов
with open('text', 'r', encoding='utf-8') as f:
    texts = f.read()
    texts = texts.replace('\ufeff', '')  # убираем первый невидимый символ

with open('text_true', 'r', encoding='utf-8') as f:
    texts_true = f.readlines()
    texts_true[0] = texts_true[0].replace('\ufeff', '') #убираем первый невидимый символ

with open('text_false', 'r', encoding='utf-8') as f:
    texts_false = f.readlines()
    texts_false[0] = texts_false[0].replace('\ufeff', '') #убираем первый невидимый символ
# Разбивам высказывания на слова
maxWordsCount = 1000
tokenizer = Tokenizer(num_words=maxWordsCount, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
                      lower=True, split=' ', char_level=False)
tokenizer.fit_on_texts([texts])
data = tokenizer.texts_to_sequences([texts])
res = np.array(data[0])

# минимальное количество достраивомых слов
inp_words = 3


n = res.shape[0] - inp_words
X = np.array([res[i:i + inp_words] for i in range(n)])
Y = to_categorical(res[inp_words:], num_classes=maxWordsCount)

# (модель, количество слоев, количство нейронов в слоях, количество нейронов на полносвязном слое)
def Make_Model(model, Count_Layer, Nerons_Layer, Nerons_Last_Layer):
    if Count_Layer > 1:
        for i in range(Count_Layer - 2):
            model.add(SimpleRNN(Nerons_Layer, return_sequences=True))
    model.add(SimpleRNN(Nerons_Layer, activation='tanh'))
    model.add(Dense(Nerons_Last_Layer, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(0.0001))
    return (model)

model = Sequential()
model.add(Embedding(maxWordsCount, 256, input_length = inp_words))
Make_Model(model, 3, 128, maxWordsCount)
# history = model.fit(X, Y, batch_size=32, epochs=50)
# model.save_weights('model1.h5')
model.load_weights("model1.h5")
def Build_Phrase(texts, str_len=10):
    res = texts
    data = tokenizer.texts_to_sequences([texts])[0]
    for i in range(str_len):
        x = data[i: i + inp_words]
        inp = np.expand_dims(x, axis=0)

        pred = model.predict(inp)
        indx = pred.argmax(axis=1)[0]
        data.append(indx)

        res += " " + tokenizer.index_word[indx]  # дописываем строку

    return res

# Ввод предложения которое будет достраиваться

# print("Введите слово:")
# Inp = input("")
# res = Build_Phrase(Inp)
res = Build_Phrase("Не позволяй другим испортить")
print(res)


#_______________________________________оценка предложения______________________________________________________________

texts = texts_true + texts_false
count_true = len(texts_true)
count_false = len(texts_false)
total_lines = count_true + count_false
maxWordsCount = 1000
tokenizer = Tokenizer(num_words=maxWordsCount, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»', lower=True, split=' ', char_level=False)
tokenizer.fit_on_texts(texts)

max_text_len = len(res) + 1
data = tokenizer.texts_to_sequences(texts)
data_pad = pad_sequences(data, maxlen=max_text_len)
X = data_pad
Y = np.array([1]*count_true + [0]*count_false)
indeces = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
X = X[indeces]
Y = Y[indeces]
model = Sequential()
model.add(Embedding(maxWordsCount, 128, input_length=max_text_len))
Make_Model(model, 1, 64, 2)

# history = model.fit(X, Y, batch_size=32, epochs=50)
# model.save_weights('model2.h5')
model.load_weights('model2.h5')

# Ввод слова которому будет даваться синтимент оценка
t = res.lower()

data = tokenizer.texts_to_sequences([t])
data_pad = pad_sequences(data, maxlen=max_text_len)
res_lstm = model.predict(data_pad)
print(res_lstm[0][0])

# Fuzzy-logic
t1 = Term('mf1', TriangularMF(-0.5, 0, 0.5))
t2 = Term('mf2', TriangularMF(0, 0.5, 1))
t3 = Term('mf3', TriangularMF(0.5, 1, 1.5))
input: FuzzyVariable = FuzzyVariable('input1', 0, 1, t1, t2, t3)

output = FuzzyVariable(
    'output', 0, 1,
    Term('mf1', TriangularMF(-0.5, 0, 0.5)),
    Term('mf2', TriangularMF(0, 0.5, 1)),
    Term('mf3', TriangularMF(0.5, 1, 1.5))
)

mf: MamdaniFuzzySystem = MamdaniFuzzySystem([input], [output])
mf.rules.append(mf.parse_rule('if (input1 is mf1) then (output is mf1)'))
mf.rules.append(mf.parse_rule('if (input1 is mf2) then (output is mf2)'))
mf.rules.append(mf.parse_rule('if (input1 is mf3) then (output is mf3)'))
result = mf.calculate({input: res_lstm[0][0]})
print(result)
print(res)
