import cv2
import imgaug.augmenters as iaa
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# set up functions for data loader
aug1 = iaa.Fliplr(0.5)
aug2 = iaa.AddToBrightness((-30, -20))
aug3 = iaa.LinearContrast((0.6, 0.75))


def get_input1(que):
    t = Tokenizer(filters='')
    que_arr = (pad_sequences(t.texts_to_sequences(
        [que]), maxlen=22, padding='post'))[0]
    return que_arr


def get_input2(img):

    # img = cv2.imread(path)
    img = np.array(img)[:, :, ::-1]

    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected img to be a numpy array, got {type(img)}")

    a = np.random.uniform()
    # if a < 0.25:
    #     img = aug1.augment_image(img)
    # elif a < 0.5:
    #     img = aug2.augment_image(img)
    # elif a < 0.75:
    #     img = aug3.augment_image(img)
    # else:
    #     img = img
    img = cv2.resize(img, (224, 224))

    img = np.array(img)/255.0

    return img


def generate_answer(model, question, img, labelencoder):
    img_input = get_input2(img)
    question_input = get_input1(question)
    predicted = model.predict(
        [np.expand_dims(question_input, axis=0), np.expand_dims(img_input, axis=0)])
    predicted_answer = labelencoder.inverse_transform(
        [np.argmax(predicted)])[0]
    return predicted_answer
