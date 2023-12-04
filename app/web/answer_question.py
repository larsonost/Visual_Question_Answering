import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import nltk
from nltk import word_tokenize
from openai_api import generate_sentence
nltk.download('punkt')

# Load pre-trained VGG16 model and create a model to extract features from the last FC layer
vgg_model = VGG16(weights='imagenet', include_top=True)
features_model = Model(inputs=vgg_model.input,
                       outputs=vgg_model.layers[-1].input)


def get_answer(model, question, img, word_idx, top_answers_classes):
    img = img.resize((224, 224))
    img = np.array(img)

    # Preprocess the input image and extract features
    feature_list = [preprocess_input(
        np.expand_dims(image.img_to_array(img), axis=0))]
    img_features = features_model([feature_list], training=False)

    # Tokenize and encode the question
    tok_list = word_tokenize(question.lower())
    question_sequence = np.reshape(
        [word_idx.get(token, 0) for token in tok_list], (1, len(tok_list)))

    # Load the trained LSTM model and predict the answer
    # loaded_model = load_model('VGG19_LSTM/lstm_coco.h5')
    answers_ids = np.argsort(model.predict(
        [img_features, question_sequence])[0])[::-1]

    # Return the answer
    one_word_answer = top_answers_classes[answers_ids[0]]
    return generate_sentence(question, one_word_answer)
