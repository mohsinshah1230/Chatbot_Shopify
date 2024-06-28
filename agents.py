import os
import time
import math
import shopify
import streamlit as st
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

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
# Define the database and tables for Shopify products and orders
engine = create_engine("sqlite:///shopify_data.db")
metadata_obj = MetaData()

# Create Shopify products table
products_table = Table(
    "shopify_products",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("title", String, nullable=False),
    Column("price", Float, nullable=False),
    Column("colors", String, nullable=True),
    Column("size", String, nullable=True),
    Column("material_type", String, nullable=True),
    Column("image_paths", String, nullable=True),
)

# Create Shopify orders table
orders_table = Table(
    "shopify_orders",
    metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("email", String, nullable=False),
    Column("created_at", String, nullable=False),
    Column("total_price", Float, nullable=False),
    Column("line_items", String, nullable=True),
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

def get_all_orders(store_handle, api_version, token, retries=3):
    api_session = Session(store_handle, api_version, token)
    ShopifyResource.activate_session(api_session)
    shop_url = f"https://{store_handle}.myshopify.com/admin/api/{api_version}"
    ShopifyResource.set_site(shop_url)
    
    total_orders = shopify.Order.count()
    num_pages = math.ceil(total_orders / 250)
    
    get_next_page = True
    page = 1
    since_id = 0

    while get_next_page and page <= num_pages:
        for _ in range(retries):
            try:
                orders = shopify.Order.find(since_id=since_id, limit=250)
                if not orders:
                    get_next_page = False
                    break
                for order in orders:
                    yield order
                if len(orders) < 250:
                    get_next_page = False
                page += 1
                since_id = orders[-1].id
                break
            except Exception as e:
                print(f"Error fetching orders: {e}")
                time.sleep(2)  # Wait for 2 seconds before retrying
                continue
        else:
            print("Failed to fetch orders after several retries.")
            break

def store_products_in_db(products, engine, table):
    with engine.begin() as connection:
        for product in products:
            product_colors = set()
            product_sizes = set()

            # Extract colors and sizes from options and variants
            for option in product.options:
                for variant in product.variants:
                    if option.name.lower() in ["color", "colour"]:
                        if option.position == 1 and variant.option1:
                            product_colors.add(variant.option1)
                        elif option.position == 2 and variant.option2:
                            product_colors.add(variant.option2)
                        elif option.position == 3 and variant.option3:
                            product_colors.add(variant.option3)
                    if option.name.lower() == "size":
                        if option.position == 1 and variant.option1:
                            product_sizes.add(variant.option1)
                        elif option.position == 2 and variant.option2:
                            product_sizes.add(variant.option2)
                        elif option.position == 3 and variant.option3:
                            product_sizes.add(variant.option3)

            # Prepare data for insertion
            data = {
                "id": product.id,
                "title": product.title,
                "price": float(product.variants[0].price) if product.variants else None,
                "colors": ", ".join(product_colors),
                "size": ", ".join(product_sizes),
                "material_type": "",  # Add logic for material_type if needed
                "image_paths": ", ".join([image.src for image in product.images]),
            }

            # Check if product already exists in the database
            existing_product = connection.execute(
                table.select().where(table.c.id == product.id)
            ).fetchone()

            if existing_product:
                print(f"Product {product.id} already exists in the database.")
                continue

            # Insert into the database
            stmt = insert(table).values(**data)
            try:
                connection.execute(stmt)
                # Print product details after insertion
                print(f"Saved product: {data}")
            except SQLAlchemyError as e:
                print(f"Error inserting product {product.id}: {e}")

def update_data_in_db(store_handle, api_version, token, engine, products_table, orders_table):
    # Update products
    latest_product_id = get_latest_id(engine, products_table)
    products = get_all_products(store_handle, api_version, token)
    new_products = (product for product in products if latest_product_id is None or product.id > latest_product_id)
    store_products_in_db(new_products, engine, products_table)

    # Update orders
    latest_order_id = get_latest_id(engine, orders_table)
    orders = get_all_orders(store_handle, api_version, token)
    new_orders = (order for order in orders if latest_order_id is None or order.id > latest_order_id)
    store_orders_in_db(new_orders, engine, orders_table)

def store_orders_in_db(orders, engine, table):
    with engine.begin() as connection:
        for order in orders:
            line_items = ", ".join([f"{item.name} (Quantity: {item.quantity})" for item in order.line_items])

            # Prepare data for insertion
            data = {
                "id": order.id,
                "email": order.email,
                "created_at": order.created_at,
                "total_price": float(order.total_price),
                "line_items": line_items,
            }
            
            # Insert into the database
            stmt = insert(table).values(**data)
            try:
                connection.execute(stmt)
            except SQLAlchemyError as e:
                print(f"Error inserting order {order.id}: {e}")

def get_latest_id(engine, table):
    with engine.connect() as connection:
        latest_id = connection.execute(select(func.max(table.c.id))).scalar()
        return latest_id

# Initialize database with latest products and orders if there are new ones
SHOP_HANDLE = 'Lemmon-24april'
API_VERSION = '2024-04'
TOKEN = os.getenv("SHOPIFY_API_KEY")
update_data_in_db(SHOP_HANDLE, API_VERSION, TOKEN, engine, products_table, orders_table)

# Use LangChain for querying
db = SQLDatabase.from_uri("sqlite:///shopify_data.db")
llm = ChatOpenAI(model="gpt-4o", temperature=0, verbose=True)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
context = toolkit.get_context()
tools = toolkit.get_tools()
messages = [
    HumanMessagePromptTemplate.from_template("""
    You are a helpful AI assistant expert in querying SQL Database to find answers to user's questions about Products or Orders. Provide the response in the following format:
    ### Product Suggestion
    - **Product ID:** <product_id>
    - **Title:** <title>
    - **Price:** $<price>
    - **Colors:** <colors>
    - **Material Type:** <material_type>
    - **Size:** <size>
    - **Image:** ![Product Image](<image_url>)

    ### Order Detail
    - **Order ID:** <order_id>
    - **Email:** <email>
    - **Created At:** <created_at>
    - **Total Price:** $<total_price>
    - **Line Items:** <line_items>

    If the user asks "What kind of product do you sell? or what product do you sell?", respond with a brief description and a list of 5 to 6 sample products in the same format.

    Input Question: {input}
    """),
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
st.title("Shopify Product")
st.write("Hello! How may I be of assistance?")

query = st.text_input("Enter your query:")

if st.button("Submit"):
    try:
        if query.lower() in ["hello", "hi", "hey"]:
            st.markdown("Hello! How may I help you?")
        else:
            response = agent_executor.invoke({"input": query})
            st.markdown(response["output"])
    except Exception as e:
        st.error(f"An error occurred: {e}")
