import tensorflow as tf
from tensorflow.keras import layers, models

# データの前処理
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # 0-1の範囲に正規化

# ニューラルネットワークの定義
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# モデルのコンパイル
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 訓練
model.fit(x_train, y_train, epochs=5)

# 評価
model.evaluate(x_test, y_test)
