# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:35:39 2024

@author: tejaswini appani
"""

import tkinter as tk
from tkinter import scrolledtext
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
import random
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Read data from text file
def read_data_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [line.strip().split(',', maxsplit=1) for line in file]
    return data

# Preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Load data from text file
file_path = r"C:\Users\tejaswini appani\Downloads\data.txt"  # Change this to the path of your text file
data = read_data_from_file(file_path)

# Process data
processed_data = [(preprocess_text(text), info) for text, *info in data]

# Shuffle data for randomness
random.shuffle(processed_data)

# Split data into features (X) and labels (y)
X, y = zip(*processed_data)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train a classifier (Naive Bayes for example)
classifier = MultinomialNB()
classifier.fit(X_vectorized, y)

# Create Tkinter window
window = tk.Tk()
window.title("Chatbot")
window.configure(bg="#1a1a1a")  # Set background color
window.geometry("600x500")  # Set the size of the window

# Create a frame for the conversation
conversation_frame = tk.Frame(window, bg="#1a1a1a")
conversation_frame.pack(pady=10, fill=tk.BOTH, expand=True)

# Create a text area for the conversation
conversation_text = scrolledtext.ScrolledText(conversation_frame, height=20, width=50, bg="#1a1a1a", fg="white",
                                              font=("Arial", 12))
conversation_text.pack(fill=tk.BOTH, expand=True)

# Create a frame for user input
input_frame = tk.Frame(window, bg="#1a1a1a")
input_frame.pack(pady=10, fill=tk.X)

# Create a text entry for user input
user_input_entry = tk.Entry(input_frame, width=50, bg="white", fg="black", font=("Arial", 12))
user_input_entry.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

# Function to handle user input
def send_message():
    user_input = user_input_entry.get()
    conversation_text.insert(tk.END, "You: " + user_input + "\n")
    user_input_entry.delete(0, tk.END)

    # Preprocess user input
    preprocessed_input = preprocess_text(user_input)

    # Vectorize the preprocessed input
    vectorized_input = vectorizer.transform([preprocessed_input])

    # Check for specific phrases to exit the chat
    if user_input == 'bye':
        response = "Chatbot: Bye! Have a nice day!\n"
        conversation_text.insert(tk.END, response)
    # Check for other specific phrases to respond
    elif user_input == 'thank you':
        response = "Chatbot: You're welcome!\n"
        conversation_text.insert(tk.END, response)
    else:
        # Calculate similarity between user input and dataset questions
        similarities = cosine_similarity(vectorized_input, X_vectorized)
        most_similar_index = similarities.argmax()
        prediction = y[most_similar_index]
        response = f"Chatbot: {prediction}.\n"
        conversation_text.insert(tk.END, response)

# Create a send button
send_button = tk.Button(input_frame, text="Send", command=send_message, bg="#4CAF50", fg="white",
                         font=("Arial", 12, "bold"))
send_button.pack(side=tk.LEFT, padx=10)

# Function to start the chatbot
def start_chatbot():
    conversation_text.insert(tk.END, "Chatbot: Welcome!\n")

# Call the function to start the chatbot
start_chatbot()

# Start the Tkinter event loop
window.mainloop()
