from fastapi import FastAPI
from pydantic import BaseModel
import json
import pickle
import nltk
import random
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.models import load_model

# Load data
with open('intents.json') as content:
    data = json.load(content)

# Load words 
words = pickle.load(open('words.pkl', 'rb'))

# Load classes
classes = pickle.load(open('classes.pkl', 'rb'))

# Load model
model = load_model('chatbot_model')

factory = StemmerFactory()
stemmer = factory.create_stemmer()

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Wellcome API Zen Bot AI"}

class InputData(BaseModel):
    message: str

@app.post("/chatbot/")
async def chatbot_response(input_data: InputData):
    """
    Endpoint to handle chatbot responses.

    Parameters:
    -----------
    input_data : InputData 
        Input data containing the user's message.

    Returns:
    --------
    dict: Dictionary 
        containing the chatbot's response.
    """
    message = input_data.message
    
    # Check if the user wants to exit
    if message.lower() == 'exit':
        return {"response": "Bot: Goodbye!"}
    
    # Predict intent and get response
    ints = predict_class(message)
    res = get_response(ints, data)
    
    return {"response": res}

def clean_up_classes(sentence):
    """
    Tokenizes and stems the input sentence using Sastrawi stemmer.

    Parameters:
    -----------
    sentence : str 
        Input sentence to be tokenized and stemmed.

    Returns:
    --------
    list: 
        List of tokenized and stemmed words.
    """
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    """
    Converts a sentence into a bag-of-words representation based on the loaded vocabulary.

    Parameters:
    -----------
    sentence : str
        Input sentence to be converted.

    Returns:
    --------
    numpy.ndarray
        Bag-of-words representation of the input sentence.
    """
    sentence_words = clean_up_classes(sentence)
    
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    """
    Predicts the intent class of the input sentence using the loaded model.

    Parameters:
    -----------
    sentence : str 
        Input sentence for intent prediction.

    Returns:
    --------
    list: 
        List of dictionaries containing the predicted intent and its probability.
    """
    bow = bag_of_words(sentence)

    res = model.predict(np.array([bow]))[0]
    
    ERROR_THRESHOLD = 0.25
    
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    
    return return_list

def get_response(intents_list, intents_json):
    """
    Retrieves a random response for the predicted intent from the loaded data.

    Parameters:
    -----------
    intents_list : list 
        List of dictionaries containing the predicted intent and its probability.
    
    intents_json : dict 
        Dictionary containing the intent data loaded from the JSON file.

    Returns:
    --------
    str: 
        Randomly selected response for the predicted intent.
    """
    if not intents_list:
        return "Maaf, saya tidak mengerti pertanyaan Anda."

    tag = intents_list[0]['intent']
    
    matching_intent = next((intent for intent in intents_json['intents'] if intent['tag'] == tag), None)

    if matching_intent:
        result = random.choice(matching_intent['responses'])
    else:
        result = "Maaf, saya tidak mengerti pertanyaan Anda."

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
