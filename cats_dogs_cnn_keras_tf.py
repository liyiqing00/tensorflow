from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

# データの読み込み
PATH = 'cats_and_dogs_small'

train_dir = os.path.join(PATH, 'train')
test_dir = os.path.join(PATH, 'test')

train_cats_dir = os.path.join(train_dir, 'cats')  # 学習用の猫画像のディレクトリ
train_dogs_dir = os.path.join(train_dir, 'dogs')  # 学習用の犬画像のディレクトリ
test_cats_dir = os.path.join(test_dir, 'cats')  # テスト用の猫画像のディレクトリ
test_dogs_dir = os.path.join(test_dir, 'dogs')  # テスト用の犬画像のディレクトリ

num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_test = len(os.listdir(test_cats_dir))
num_dogs_test = len(os.listdir(test_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_test = num_cats_test + num_dogs_test

# 学習用パラメータ
batch_size = 128  # バッチサイズ
epochs = 25      # エポック数
IMG_HEIGHT = 64   # 画像サイス
IMG_WIDTH = 64

# データの準備
train_image_generator = ImageDataGenerator(rescale=1./255) # 学習データのジェネレータ
test_image_generator = ImageDataGenerator(rescale=1./255) # テストデータのジェネレータ

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=test_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

# モデルの構築
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# モデルのコンパイル
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# モデルの概要
model.summary()

# モデルの学習
model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs
)

# モデルの評価
test_loss, test_acc = model.evaluate(test_data_gen)
print('test_accuracy ', test_acc)


