import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any

# Load environment variables
load_dotenv()

class StreamlitCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for streaming responses to Streamlit"""
    
    def __init__(self, container):
        self.container = container
        self.text = ""
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Handle new token from LLM"""
        self.text += token
        self.container.markdown(self.text + "▌")
    
    def on_llm_end(self, response, **kwargs: Any) -> None:
        """Handle end of LLM response"""
        self.container.markdown(self.text)

class NVIDIAChatbot:
    def __init__(self):
        self.api_key = os.getenv("NVIDIA_API_KEY")
        self.model_name = os.getenv("MODEL_NAME")
        self.app_title = os.getenv("APP_TITLE", "LangChain Chatbot")
        
        # Initialize LangChain ChatOpenAI with NVIDIA endpoint
        self.llm = ChatOpenAI(
            model=self.model_name,
            openai_api_key=self.api_key,
            openai_api_base="https://integrate.api.nvidia.com/v1",
            temperature=0.7,
            max_tokens=1024,
            streaming=True
        )
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="history"
        )
        
        # Create conversation prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["history", "input"],
            template="""You are a helpful AI assistant powered by NVIDIA's technology. 
            Be conversational, helpful, and engaging in your responses.

            Chat History:
            {history}

            Human: {input}
            AI Assistant:"""
        )
        
        # Initialize conversation chain
        self.conversation_chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=self.prompt_template,
            verbose=False
        )
    
    def validate_config(self):
        """Validate that required environment variables are set"""
        if not self.api_key:
            st.error("NVIDIA_API_KEY not found in environment variables. Please set it in your .env file.")
            st.stop()
        
        if not self.model_name:
            st.error("MODEL_NAME not found in environment variables. Please set it in your .env file.")
            st.stop()
    
    def update_model(self, new_model_name: str):
        """Update the model and reinitialize the LLM"""
        if new_model_name != self.model_name:
            self.model_name = new_model_name
            
            # Reinitialize LLM with new model
            self.llm = ChatOpenAI(
                model=self.model_name,
                openai_api_key=self.api_key,
                openai_api_base="https://integrate.api.nvidia.com/v1",
                temperature=0.7,
                max_tokens=1024,
                streaming=True
            )
            
            # Reinitialize conversation chain with new LLM
            self.conversation_chain = ConversationChain(
                llm=self.llm,
                memory=self.memory,
                prompt=self.prompt_template,
                verbose=False
            )
    
    def get_streaming_response(self, user_input: str, container):
        """Get streaming response using LangChain"""
        try:
            # Create callback handler for streaming
            callback_handler = StreamlitCallbackHandler(container)
            
            # Get response from conversation chain with streaming
            response = self.conversation_chain.predict(
                input=user_input,
                callbacks=[callback_handler]
            )
            
            return response
            
        except Exception as e:
            st.error(f"Error calling NVIDIA API: {str(e)}")
            return None
    
    def get_conversation_history(self):
        """Get formatted conversation history"""
        messages = []
        if hasattr(self.memory, 'chat_memory') and self.memory.chat_memory.messages:
            for message in self.memory.chat_memory.messages:
                if isinstance(message, HumanMessage):
                    messages.append({"role": "user", "content": message.content})
                elif isinstance(message, AIMessage):
                    messages.append({"role": "assistant", "content": message.content})
        return messages
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()

def initialize_session_state():
    """Initialize Streamlit session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = NVIDIAChatbot()

def main():
    st.set_page_config(
        page_title="NVIDIA Chatbot with LangChain",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Validate configuration
    st.session_state.chatbot.validate_config()
    
    # App header
    st.title(f"{st.session_state.chatbot.app_title} 🦜🔗")
    st.markdown("Powered by NVIDIA NIM API and LangChain")
    
    # Sidebar for model configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Model selection
        current_model = st.text_input(
            "Model Name",
            value=st.session_state.chatbot.model_name,
            help="Enter the NVIDIA model name (e.g., meta/llama3-70b-instruct)"
        )
        
        # Update model if changed
        if current_model != st.session_state.chatbot.model_name:
            st.session_state.chatbot.update_model(current_model)
            st.success(f"Model updated to: {current_model}")
        
        st.markdown("---")
        
        # Memory and conversation controls
        st.subheader("Conversation Controls")
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.chatbot.clear_memory()
            st.session_state.messages = []
            st.rerun()
        
        # Show memory info
        memory_messages = st.session_state.chatbot.get_conversation_history()
        st.info(f"Messages in memory: {len(memory_messages)}")
        
        st.markdown("---")
        
        # Display current model
        st.info(f"Current Model: {st.session_state.chatbot.model_name}")
        
        # LangChain info
        st.markdown("### 🦜🔗 LangChain Features")
        st.markdown("""
        - **Conversation Memory**: Maintains context across messages
        - **Streaming Responses**: Real-time token generation
        - **Prompt Templates**: Structured conversation prompts
        - **Chain Management**: Modular conversation handling
        """)
    
    # Display chat messages from LangChain memory
    memory_messages = st.session_state.chatbot.get_conversation_history()
    for message in memory_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response using LangChain
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Get streaming response from LangChain
            with st.spinner("Thinking..."):
                response = st.session_state.chatbot.get_streaming_response(prompt, message_placeholder)
            
            if not response:
                st.error("Failed to get response from NVIDIA API. Please check your configuration.")


if __name__ == "__main__":
    main()
