from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import streamlit as st

# Load env vars
load_dotenv()

st.title('AutoGPT Wizard')
prompt = st.text_input('Tell me a topic you want to learn its programming language:') 

# Prompt templates
language_template = PromptTemplate(
    input_variables = ['topic'], 
    template='Suggest me a programming language for {topic} and respond in a code block with the language name only'
)

book_recommendation_template = PromptTemplate(
    input_variables = ['programming_language'], 
    template='''Recommend me a book based on this programming language {programming_language}

    The book name should be in a code block and the book name should be the only text in the code block
    '''
)

llm = OpenAI(temperature=0.9, model_name="gpt-3.5-turbo") 
language_chain = LLMChain(llm=llm, prompt=language_template, verbose=True, output_key='programming_language')
book_recommendation_chain = LLMChain(llm=llm, prompt=book_recommendation_template, verbose=True, output_key='book_name')

sequential_chain = SequentialChain(
    chains = [language_chain, book_recommendation_chain], 
    input_variables=['topic'], 
    output_variables=['programming_language', 'book_name'],
    verbose=True)

if prompt: 
    reply = sequential_chain({'topic': prompt})

    with st.expander("Result"):
        st.info(reply)
        
    with st.expander("Programming Language"):
        st.info(reply['programming_language'])

    with st.expander("Recommended Book"):
        st.info(reply['book_name'])
