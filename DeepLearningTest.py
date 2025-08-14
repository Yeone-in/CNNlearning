#얘들아 우리한테는 표가 있어. 근데 그걸을 컴퓨터에 입력을 시켜야 해.
#그걸 도와 주와주는 게 판다스 함수
import pandas as pd
import tensorflow  as tf

#데이터 불러오기
'''
파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
레모네이드 = pd.read_csv(파일경로)
'''

#데이터 모양 확인
print(레모네이드.shape)

# 데이터 칼럼이름 확인
print(레모네이드.columns)

# 독립변수와 종속변수 분리
독립 = 레모네이드[['온도']]
종속 = 레모네이드[['판매량']]
print(독립.shape, 종속.shape)

레모네이드.head()

#자 이번엔 엄청나게 중요한 거를 할겁니다!~ 집중집중~~ 수업전체 핵심입니당
# 모델을 만듭니다.
X = tf.keras.layers.Input(shape=[1])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')

# 모델을 학습시킵니다. fitV
model.fit(독립, 종속, epochs=5000, verbose=0)
model.fit(독립, 종속, epochs=10)

# 모델을 이용합니다.
print(model.predict(독립))
print(model.predict([[45]]))
