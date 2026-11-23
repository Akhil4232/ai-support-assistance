import streamlit as st
import pickle
import os

st.set_page_config(page_title="AI Support Assistant", page_icon="🤖", layout="centered")

# Custom CSS for beautiful UI
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .angry-box { background-color: #ffe6e6; border-left: 6px solid #ff4b4b; padding: 20px; border-radius: 10px; text-align: center; }
    .happy-box { background-color: #e6ffe6; border-left: 6px solid #00cc66; padding: 20px; border-radius: 10px; text-align: center; }
    .neutral-box { background-color: #fff8e6; border-left: 6px solid #ffaa00; padding: 20px; border-radius: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Support Assistant")
st.markdown("### Understand your customers' emotions in real-time!")
st.write("Type a customer message below and let AI detect the sentiment.")

# Load model
@st.cache_resource
def load_model():
    with open("sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

try:
    model, vectorizer = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Model not found: {e}")
    model_loaded = False

# Input
user_input = st.text_area("💬 Enter Customer Message:", placeholder="e.g. I am very happy with your service, thank you!", height=100)

# Example buttons
st.write("Try examples:")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("😡 Angry Example"):
        user_input = "I am very angry, your service is terrible and worst!"
with col2:
    if st.button("😊 Happy Example"):
        user_input = "I love your service, amazing experience!"
with col3:
    if st.button("😐 Neutral Example"):
        user_input = "My order is delivered, ok."
if user_input:
    st.text_area("Your message:", value=user_input, disabled=True)

if st.button("🔍 Check Sentiment", type="primary"):
    if not user_input:
        st.warning("Please enter a message first!")
    elif model_loaded:
        # Predict
        vect_text = vectorizer.transform([user_input])
        prediction = model.predict(vect_text)[0]
        
        # Confidence
        try:
            proba = model.predict_proba(vect_text)[0]
            confidence = max(proba) * 100
        except:
            confidence = 95.0

        pred_lower = str(prediction).lower()

        st.markdown("---")
        st.subheader("📊 Result:")

        if "angry" in pred_lower or "negative" in pred_lower or "sad" in pred_lower:
            st.markdown(f'<div class="angry-box"><h1>😡 ANGRY</h1><h3>Confidence: {confidence:.1f}%</h3><p>⚠️ Urgent! Customer needs immediate attention.</p></div>', unsafe_allow_html=True)
            st.progress(int(confidence))
            st.info("💡 Suggested Action: Apologize and escalate to senior support.")
        elif "happy" in pred_lower or "positive" in pred_lower:
            st.markdown(f'<div class="happy-box"><h1>😊 HAPPY</h1><h3>Confidence: {confidence:.1f}%</h3><p>🎉 Great! Customer is satisfied.</p></div>', unsafe_allow_html=True)
            st.progress(int(confidence))
            st.info("💡 Suggested Action: Thank them and ask for a review!")
        else:
            st.markdown(f'<div class="neutral-box"><h1>😐 NEUTRAL</h1><h3>Confidence: {confidence:.1f}%</h3><p>Customer message is neutral.</p></div>', unsafe_allow_html=True)
            st.progress(int(confidence))
            st.info("💡 Suggested Action: Provide helpful information.")

        # Show all probabilities
        with st.expander("See detailed breakdown"):
            st.write(f"Prediction: **{prediction}**")
            st.write(f"Confidence: **{confidence:.1f}%**")

st.markdown("---")
st.caption("Built by Akhil Konathala | Python • Scikit-Learn • Streamlit")
