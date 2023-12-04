from load_model import get_caption_model, get_vqa_model
from answer_question import get_answer
from generate_caption import generate_caption
import tensorflow as tf
import streamlit as st
import pickle
from PIL import Image
import os
from openai import OpenAI

try:
    KEY = st.secrets["openai_key"]
except Exception as e:
    KEY = os.getenv("OPENAI_API_KEY")
CLIENT = OpenAI(api_key=KEY)


def generate_sentence(question, answer, client=CLIENT):
    """Generate a sentence based on a question and answer using the OpenAI API."""
    input = f'Question: {question}\nAnswer: {answer}'
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a sentence assistant. Given a question and a one-word answer, you will convert that one-word answer into a complete sentence based on the question without adding any imaginary content."},
            {"role": "user", "content": input}
        ]
    )
    sentence = completion.choices[0].message
    return sentence.content


@st.cache_resource()
def load_model():
    """Load the caption and VQA models."""
    # load tokenizer
    from_disk = pickle.load(open("app/image_caption/tv_layer_new.pkl", "rb"))
    tokenizer = tf.keras.layers.TextVectorization(max_tokens=from_disk['config']['max_tokens'],
                                                  output_mode='int',
                                                  output_sequence_length=from_disk['config']['output_sequence_length'])
    tokenizer.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    tokenizer.set_weights(from_disk['weights'])

    # Load the caption model
    caption_model = get_caption_model(
        'app/image_caption/sim_weights.h5', tokenizer)

    # Load the list of top answer classes and word-to-index mapping
    with open('VGG19_LSTM/mode_answers.txt', 'r') as f:
        top_answers_classes = [line.strip() for line in f]

    with open('VGG19_LSTM/word_idx', 'rb') as f:
        word_idx = pickle.load(f)
    # load vaq model
    vqa_model = get_vqa_model('VGG19_LSTM/lstm_coco.h5')
    return caption_model, tokenizer, top_answers_classes, word_idx, vqa_model


caption_model, tokenizer, top_answers_classes, word_idx, vqa_model = load_model()


# Set up the logo
logo_image = Image.open('app/images/logo.png')
# Display the logo
st.image(logo_image, use_column_width=True)
st.title("Image Interact")
st.subheader(
    'Get auto-generated captions and answers to your questions about images!')

# Upload image through Streamlit
uploaded_file = st.file_uploader(
    "Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Initialize session state
    if 'captioned' not in st.session_state:
        st.session_state.captioned = False
    if 'answered' not in st.session_state:
        st.session_state.answered = False

    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    if st.button("Generate Caption"):
        with st.spinner("Generating caption results..."):
            # print(uploaded_file.name)
            # Generate captions on the uploaded image
            img = uploaded_file.read()
            img_caption = generate_caption(
                img, caption_model, tokenizer)
        st.session_state.captioned = True
        st.session_state.img_caption = img_caption
    if st.session_state.captioned:
        # Display the caption results
        st.subheader("Auto-generated image caption:")
        st.text(f'{st.session_state.img_caption.capitalize()}.')

    st.divider()
    st.subheader("Visual question answering:")
    # Adding a submit button
    user_question = st.text_input(
        "Ask any question about the image you uploaded: ", placeholder='e.g. What is the color of...')
    vqa_button = st.button("Submit")
    if vqa_button and user_question:
        with st.spinner("Generating answers..."):
            # Make VQA prediction
            img = Image.open(uploaded_file)
            one_word_answer = get_answer(
                vqa_model, user_question, img, word_idx, top_answers_classes)
            vqa_answer = generate_sentence(user_question, one_word_answer)
            st.session_state.answered = True
            st.session_state.vqa_answer = vqa_answer
            print(vqa_answer)

    if vqa_button and user_question == '':
        st.warning('Please enter a question.')
    if st.session_state.answered:
        # Display the VQA vqa_answer
        st.subheader("Auto-generated answer from our VQA model:")
        st.text(st.session_state.vqa_answer)
