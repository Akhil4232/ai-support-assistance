import streamlit as st
import pickle

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

st.set_page_config(page_title="AI Support Assistant", page_icon="🤖")
st.title("🤖 AI Support Assistant")
st.write("Type your message, I will detect if you are angry, happy or neutral")

# Example buttons
col1, col2, col3 = st.columns(3)
if col1.button("I am angry service is bad"):
    st.session_state.text = "I am angry service is bad"
if col2.button("Thank you great service"):
    st.session_state.text = "Thank you great service"
if col3.button("My internet is not working"):
    st.session_state.text = "My internet is not working"

text = st.text_input("Your message:", value=st.session_state.get("text", ""))

if st.button("Check Sentiment"):
    if text:
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]

        if prediction == "angry":
            st.error("😡 You seem Angry. Sorry! Connecting to senior support...")
        elif prediction == "happy":
            st.success("😊 You seem Happy! Thank you so much!")
        else:
            st.info("🤔 Got it. Let me check your issue...")
        st.write(f"**Detected:** {prediction}")
    else:
        st.warning("Please type something first")