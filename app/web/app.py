import streamlit as st
import pickle
from PIL import Image
import tensorflow as tf
from generate_caption import generate_caption
from answer_question import get_answer
from load_model import get_caption_model, get_vqa_model

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
# Initialize session state
if 'vqa_prediction_in_progress' not in st.session_state:
    st.session_state.vqa_prediction_in_progress = False

if uploaded_file is not None:

    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    # if st.session_state.vqa_prediction_in_progress == False:
    with st.spinner("Generating caption results..."):
        # print(uploaded_file.name)
        # Generate captions on the uploaded image
        img = uploaded_file.read()
        img_caption = generate_caption(
            img, caption_model, tokenizer)
    # Display the prediction results
    st.subheader(img_caption)
    st.session_state.vqa_prediction_in_progress = True
    # Allow the user to input a question
    user_question = st.text_input("Ask a question about the image:")

    # Provide VQA results when the user submits a question
    if st.button("Submit") and user_question:
        with st.spinner("Making VQA prediction..."):
            # Make VQA prediction
            img = Image.open(uploaded_file)
            vqa_predictions = get_answer(
                vqa_model, user_question, img, word_idx, top_answers_classes)
        # Display the VQA predictions
        st.subheader("Answer from our VQA model:")
        st.write(vqa_predictions)
