import streamlit as st
from rag_functionality import rag_func

# Set initial message
if "messages" not in st.session_state.keys():
    st.session_state['messages'] = [
        {"role": "assistant", "content": "Hi, I am your Uganda Bureau of Statistics AI assistant. How can I help you?"}
    ]

# Display messages
if "messages" in st.session_state.keys():
    for message in st.session_state['messages']:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# User's ability to type prompts
user_prompt = st.chat_input()
if user_prompt is not None:
    # Append user's message to the conversation history
    st.session_state['messages'].append({"role": "user", "content": user_prompt})

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ai_response = rag_func(user_prompt)
            st.write(ai_response)
    
    # Append the AI's response to the conversation history
    new_ai_message = {"role": "assistant", "content": ai_response}
    st.session_state['messages'].append(new_ai_message)