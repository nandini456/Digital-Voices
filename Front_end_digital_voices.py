import streamlit as st
from streamlit_lottie import st_lottie
import pandas as pd
import pickle
import re
import json

# nltk
import nltk

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background-color: #000000;
opacity: 1;
background-image:  radial-gradient(#ffffff 2px, transparent 2px), radial-gradient(#ffffff 2px, #000000 2px);
background-size: 80px 80px;
background-position: 0 0,40px 40px;
}

[data-testid="stSidebar"] .css-1lcbmhc, .css-1d391kg {
    color: black;  /* Text color */
}

/* Add padding to columns for increased gap */
.css-18e3th9 {
    padding: 10px 30px;
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("WELCOME I'M HAILEY, I LL CLASSIFY ALL YOUR REVIEWS INTO POSITIVE OR NEGATIVE")


def load_lottiefile(filepath: str):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("File not found at the specified location.")
        return None
    except json.JSONDecodeError:
        st.error("Invalid JSON format in the file.")
        return None


# Fixing the filepath and using raw string
lottie_hello = load_lottiefile(r"C:\\Users\\user\Desktop\\NANDINI PROJECTS\\Digital Voices\\ani2.json")
lottie_happy = load_lottiefile(r"C:\\Users\\user\\Desktop\\NANDINI PROJECTS\\Digital Voices\\happyface.json")
lottie_sad = load_lottiefile(r"C:\\Users\\user\\Desktop\\NANDINI PROJECTS\\Digital Voices\\crygace.json")

st_lottie(
    lottie_hello,
    speed=1,
    reverse=False,
)

# Defining dictionary containing all emojis with their meanings.
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked', ':-$': 'confused', ':\\': 'annoyed',
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
          ';-)': 'wink', 'O:-)': 'angel', 'O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'}

# Defining set containing all stopwords in english.
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
                'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
                'being', 'below', 'between', 'both', 'by', 'can', 'd', 'did', 'do',
                'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
                'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
                'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
                'into', 'is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
                'me', 'more', 'most', 'my', 'myself', 'now', 'o', 'of', 'on', 'once',
                'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'own', 're',
                's', 'same', 'she', "shes", 'should', "shouldve", 'so', 'some', 'such',
                't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
                'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
                'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was',
                'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom',
                'why', 'will', 'with', 'won', 'y', 'you', "youd", "youll", "youre",
                "youve", 'your', 'yours', 'yourself', 'yourselves']


def preprocess(textdata):
    processedText = []

    # Create Lemmatizer and Stemmer.
    wordLemm = WordNetLemmatizer()

    # Defining regex patterns.
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern = '@[^\s]+'
    alphaPattern = "[^a-zA-Z0-9]"
    sequencePattern = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"

    for review in textdata:
        review = review.lower()

        # Replace all URls with 'URL'
        review = re.sub(urlPattern, ' URL', review)
        # Replace all emojis.
        for emoji in emojis.keys():
            review = review.replace(emoji, "EMOJI" + emojis[emoji])
        # Replace @USERNAME to 'USER'.
        review = re.sub(userPattern, ' USER', review)
        # Replace all non alphabets.
        review = re.sub(alphaPattern, " ", review)
        # Replace 3 or more consecutive letters by 2 letter.
        review = re.sub(sequencePattern, seqReplacePattern, review)
        reviewwords = ''
        for word in review.split():
            # Checking if the word is a stopword.
            # if word not in stopwordlist:
            if len(word) > 1:
                # Lemmatizing the word.
                word = wordLemm.lemmatize(word)
                reviewwords += (word + ' ')

        processedText.append(reviewwords)

    return processedText


def load_models():
    # Load the vectoriser.
    file = open('C:\\Users\\user\\Desktop\\NANDINI PROJECTS\\Digital Voices\\vectoriser-ngram-(1,2).pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()
    # Load the LR Model.
    file = open('C:\\Users\\user\\Desktop\\NANDINI PROJECTS\\Digital Voices\\Sentiment-LR.pickle', 'rb')
    LRmodel = pickle.load(file)
    file.close()

    return vectoriser, LRmodel


def predict(vectoriser, model, text):
    # Predict the sentiment
    textdata = vectoriser.transform(preprocess(text))
    sentiment = model.predict(textdata)

    # Make a list of text with sentiment.
    data = []
    positive_count = 0
    negative_count = 0
    for text, pred in zip(text, sentiment):
        if pred == 0:
            negative_count += 1
        else:
            positive_count += 1
        data.append((text, pred))
    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns=['text', 'sentiment'])
    df = df.replace([0, 1], ["Negative", "Positive"])
    return df, positive_count, negative_count


def main():
    st.title("DIGITAL VOICES")
    st.title("Sentiment Analysis of Online Student Reviews")

    # About section in the sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This application uses a Logistic Regression model to classify the sentiment of online reviews as positive or negative. 
        The model has been trained on a dataset of online reviews. It uses various text preprocessing techniques 
        and natural language processing (NLP) methods to clean and analyze the text data.
        """
    )

    vectoriser, LRmodel = load_models()
    text = st.text_area("Enter text to classify (each line will be treated as a separate input):", height=400)
    if st.button("Classify"):
        text = text.split("\n")
        df, positive_count, negative_count = predict(vectoriser, LRmodel, text)
        st.dataframe(df, width=1000)

        # Display lottie_happy and lottie_sad animations side by side with reduced size
        col1, empty_col, col2 = st.columns([1, 0.2, 1])  # Adjust the ratio to add space

        with col1:
            st_lottie(
                lottie_happy,
                speed=1,
                reverse=False,
                width=300,  # Adjust the width
                height=295,  # Adjust the height
            )
            st.write(f"Number of Positive Sentiments: {positive_count}")

        with col2:
            st_lottie(
                lottie_sad,
                speed=1,
                reverse=False,
                width=300,  # Adjust the width
                height=300,  # Adjust the height
            )
            st.write(f"Number of Negative Sentiments: {negative_count}")


if __name__ == "__main__":
    main()