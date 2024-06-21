import os
import time
import math
import shopify
import streamlit as st
from dotenv import load_dotenv
from shopify import Session, ShopifyResource
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Float, select, insert, func
from sqlalchemy.exc import SQLAlchemyError
from langchain.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.prompt import SQL_FUNCTIONS_SUFFIX
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.agents import create_openai_tools_agent
from langchain.agents.agent import AgentExecutor

load_dotenv()


# Set OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define the database and table for Shopify products
engine = create_engine("sqlite:///shopify_products.db")
metadata_obj = MetaData()

# Create Shopify products table
table_name = "shopify_products"
shopify_products_table = Table(
    table_name,
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("title", String, nullable=False),
    Column("price", Float, nullable=False),
    Column("colors", String, nullable=True),
    Column("material_type", String, nullable=True),
    Column("option_name", String, nullable=True),
    Column("option_value", String, nullable=True),
    Column("image_paths", String, nullable=True),
)

metadata_obj.create_all(engine)

def get_all_products(store_handle, api_version, token, retries=3):
    api_session = Session(store_handle, api_version, token)
    ShopifyResource.activate_session(api_session)
    shop_url = f"https://{store_handle}.myshopify.com/admin/api/{api_version}"
    ShopifyResource.set_site(shop_url)
    
    total_products = shopify.Product.count()
    num_pages = math.ceil(total_products / 250)
    
    get_next_page = True
    page = 1
    since_id = 0

    while get_next_page and page <= num_pages:
        for _ in range(retries):
            try:
                products = shopify.Product.find(since_id=since_id, limit=250)
                if not products:
                    get_next_page = False
                    break
                for product in products:
                    yield product
                if len(products) < 250:
                    get_next_page = False
                page += 1
                since_id = products[-1].id
                break
            except Exception as e:
                print(f"Error fetching products: {e}")
                time.sleep(2)  # Wait for 2 seconds before retrying
                continue
        else:
            print("Failed to fetch products after several retries.")
            break

# Insert products into the database
def store_products_in_db(products, engine, table):
    with engine.begin() as connection:
        for product in products:
            data = {
                "id": product.id,
                "title": product.title,
                "price": float(product.variants[0].price),
                "colors": ",".join([option.value for option in product.options if option.name == "Color"]),
                "material_type": ",".join([option.value for option in product.options if option.name == "Material Type"]),
                "option_name": ",".join([option.name for option in product.options]),
                "option_value": ",".join([option.value for option in product.options]),
                "image_paths": ",".join([image.src for image in product.images]),
            }
            stmt = insert(table).values(**data)
            try:
                connection.execute(stmt)
            except SQLAlchemyError as e:
                print(f"Error inserting product {product.id}: {e}")

def update_products_in_db(store_handle, api_version, token, engine, table):
    latest_product_id = get_latest_product_id(engine, table)
    products = get_all_products(store_handle, api_version, token)

    new_products = (product for product in products if product.id > latest_product_id)
    store_products_in_db(new_products, engine, table)

# Check if table has data
def get_latest_product_id(engine, table):
    with engine.connect() as connection:
        latest_id = connection.execute(select(func.max(table.c.id))).scalar()
        return latest_id

# Initialize database with latest products if there are new ones
SHOP_HANDLE = 'Lemmon-24april'
API_VERSION = '2024-04'
TOKEN = os.getenv("SHOPIFY_API_KEY")

update_products_in_db(SHOP_HANDLE, API_VERSION, TOKEN, engine, shopify_products_table)

# Use LangChain for querying
db = SQLDatabase.from_uri("sqlite:///shopify_products.db")

llm = ChatOpenAI(model="gpt-4o", temperature=0, verbose=True)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
context = toolkit.get_context()
tools = toolkit.get_tools()

messages = [
    HumanMessagePromptTemplate.from_template("You are a helpful AI assistant expert in querying SQL Database to find answers to user's question about Products and products image, title, price, description and give me the correct answer in Markdown format.Provide the response in the following format:### Product Suggestion- **Title:** <title>- **Price:** $<price>  - **Description:** <description> - **Image:** ![Product Image](<image_url> Input Question:{input}"),
    AIMessage(content=SQL_FUNCTIONS_SUFFIX),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
]

prompt = ChatPromptTemplate.from_messages(messages)
prompt = prompt.partial(**context)

agent = create_openai_tools_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=True,
)

# Streamlit Chatbot Interface
st.title("Shopify Product Chatbot")
st.write("Ask me anything about the products or any other queries!")

query = st.text_input("Enter your query:")

if st.button("Submit"):
    try:
        response = agent_executor.invoke({"input": query})
        st.markdown(response["output"])
    except Exception as e:
        st.error(f"An error occurred: {e}")
