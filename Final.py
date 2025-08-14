import tensorflow as tf
import pandas as pd
import numpy as np

# 데이터 준비
data = {
    '항생제': [0, 0, 1, 1, 2, 2],
    '플라스틱종류': [0, 1, 0, 1, 0, 1],
    '항생제내성비율': [58.52, 55.46, 67.52, 62.88, 45.23, 55.89]
}

anti = pd.DataFrame(data)

독립 = anti[['항생제', '플라스틱종류']]
종속 = anti[['항생제내성비율']]

# 모델의 구조
X = tf.keras.layers.Input(shape=[2])
H = tf.keras.layers.Dense(8, activation="swish")(X)
H = tf.keras.layers.Dense(8, activation="swish")(H)
H = tf.keras.layers.Dense(8, activation="swish")(H)
H = tf.keras.layers.Dense(8, activation="swish")(H)
Y = tf.keras.layers.Dense(1)(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mean_absolute_error'])

model.summary()

# 모델 학습
model.fit(독립, 종속, epochs=10000)

# 예측
print(model.predict(독립[:5]))
print(종속[:5])
