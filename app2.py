from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# 모델이 없으면 학습 후 저장
MODEL_PATH = 'mnist_model.h5'

def train_and_save_model():
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
    from tensorflow.keras.utils import to_categorical

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
    model.save(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    train_and_save_model()

# 모델 로드
model = load_model(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        file = request.files['file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # 이미지 전처리
        img = Image.open(filepath).convert('L')  # 흑백 변환
        img = ImageOps.invert(img)               # 흰색 바탕, 검은 숫자
        img = img.resize((28, 28))               # MNIST 사이즈
        img = np.array(img).reshape(1, 28, 28, 1) / 255.0

        # 예측
        pred = model.predict(img)
        prediction = np.argmax(pred)

    return render_template("index2.html", prediction=prediction)

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
