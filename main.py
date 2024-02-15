import os
import streamlit as st

from credentials import OPENAI_API_KEY
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

# Environmental Variable
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Streamlit Framework
st.set_page_config(page_title="recipe generator")
st.title("Recipe Generation")
ip_dish_name = st.text_input("Find Recipe for: ")

# LLM
llm = OpenAI(temperature = 1) 

# Save the content in memory
ip_dish_memory = ConversationBufferMemory(input_key = "dish_name", memory_key = "dish_name_history")
recipe_memory = ConversationBufferMemory(input_key = "recipe", memory_key = "recipe_history")

# Prompt Template
template1 = PromptTemplate(input_variables = ["dish_name"], 
			template = "Consider yourself as Chef. Try to give clear instruction in 5 points or lesser for {dish_name}")
chain1 = LLMChain(llm = llm, prompt = template1, verbose = True, output_key = "recipe", memory = ip_dish_memory)

template2 = PromptTemplate(input_variables = ["recipe"], 
			template = "Rate the difficulty level of the below recipe. Recipe: \n {recipe} \n Difficulty Level: ")
chain2 = LLMChain(llm = llm, prompt = template2, verbose = True, output_key = "recipe_level", memory = recipe_memory)

template3 = PromptTemplate(input_variables = ["dish_name"], 
			template = "List one place where {dish_name} is famous for")
chain3 = LLMChain(llm = llm, prompt = template3, verbose = True, output_key ="famous_place")

# To exceute the chain sequentially
final_chain = SequentialChain( chains = [chain1, chain2, chain3],
				input_variables = ["dish_name"], output_variables = ["recipe", "recipe_level", "famous_place"], verbose = True)

if ip_dish_name:
	output = final_chain({"dish_name": ip_dish_name})
	st.write(output)

	with st.expander("Storing Dish name and Recipe Generated:"):
		st.info(ip_dish_memory.buffer)

	with st.expander("Storing the Recipe and Predicted difficulty level:"):
		st.info(recipe_memory.buffer)
