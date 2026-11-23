import pickle

# Load saved model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("=== AI Support Assistant ===")
print("Type 'exit' to stop\n")

while True:
    text = input("You: ")
    if text.lower() == "exit":
        print("Bot: Bye! Have a great day!")
        break

    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]

    if prediction == "angry":
        print("Bot: I understand you are angry. Sorry! Connecting you to senior support...")
    elif prediction == "happy":
        print("Bot: Glad you are happy! Thank you so much!")
    else:
        print("Bot: Got it. Let me check your issue...")
    print(f"[Detected: {prediction}]\n")