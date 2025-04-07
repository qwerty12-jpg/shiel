import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
import firebase_admin
from firebase_admin import auth, credentials
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
import requests
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import google.generativeai as genai
import json
from langchain_google_genai import GoogleGenerativeAI

with open('config.json', 'r') as f:
    config = json.load(f)

GEMINI_API_KEY = config["GOOGLE_API_KEY"]
MONGO_URI = config["MONGODB_ATLAS_URI"]
SERPER_API_KEY = config["SERPER_API_KEY"]

genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()

# Enable CORS to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://10.0.2.2:8005"],  # Change "*" to the exact address
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model & Tokenizer
MODEL_NAME = "annahaz/xlm-roberta-base-misogyny-sexism-indomain-mix-bal"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Request Model
class URLRequest(BaseModel):
    url: str

def extract_text_from_url(url: str) -> str:
    """Scrape and extract text from a given URL."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)  # Set a timeout

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        content = " ".join([p.get_text() for p in paragraphs])
        return content if content else "No textual content found."
    else:
        raise HTTPException(status_code=400, detail="Failed to fetch URL content.")

def check_misogyny(text: str) -> dict:
    """Check if text is misogynistic using the model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    misogyny_score = probabilities[0][1].item()  # Probability of misogynistic class
    is_misogynistic = misogyny_score > 0.5

    return {"is_misogynistic": is_misogynistic, "score": round(misogyny_score, 4)}

@app.post("/predict/")
def predict_misogyny(request: URLRequest):
    """API endpoint to analyze a URL for misogynistic content."""
    try:
        content = extract_text_from_url(request.url)
        result = check_misogyny(content)
        return {"url": request.url, "misogynistic": result["is_misogynistic"], "score": result["score"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



cred = credentials.Certificate(r"C:\Users\manas\Desktop\Projects\sh_bknd_models\auth\shieldher-31a9f-firebase-adminsdk-tzaxb-8de7e3862a.json")
firebase_admin.initialize_app(cred)


FIREBASE_WEB_API_KEY = "AIzaSyCzfREtLXkFiMLCub1BFjybM0UQU8yGG9I"  

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

@app.post("/register")
async def register_user(request: RegisterRequest):
    """Register a new user with Firebase Authentication."""
    try:
        user = auth.create_user(email=request.email, password=request.password)
        return {"message": "User registered successfully", "uid": user.uid}
    except firebase_admin.auth.EmailAlreadyExistsError:
        raise HTTPException(status_code=400, detail="Email already exists")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/login")
async def login_user(request: LoginRequest):
    """Log in a user with Firebase Authentication."""
    try:
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"
        payload = {
            "email": request.email,
            "password": request.password,
            "returnSecureToken": True
        }
        response = requests.post(url, json=payload)
        result = response.json()

        if "idToken" in result:
            return {
                "message": "Login successful",
                "idToken": result["idToken"],  
                "refreshToken": result["refreshToken"],  
                "uid": result["localId"]  
            }
        else:
            raise HTTPException(status_code=400, detail=result.get("error", {}).get("message", "Login failed"))

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Firebase Authentication API is running"}




# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["shieldHer"]
collection = db["agenticRAG"]

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Fetch all documents from MongoDB
docs = list(collection.find({}, {"title": 1, "link": 1, "content": 1, "embedding": 1}))

if not docs:
    raise ValueError("No documents found in MongoDB! Check your collection.")

if "embedding" not in docs[0]:
    raise ValueError("Documents do not contain embeddings! Ensure they are generated.")

# Create FAISS index
dimension = len(docs[0]["embedding"])
index = faiss.IndexFlatL2(dimension)
doc_map = []

for i, doc in enumerate(docs):
    index.add(np.array([doc["embedding"]], dtype="float32"))
    doc_map.append(doc)

# Define request model
class QueryRequest(BaseModel):
    query: str

# -------------------- Knowledge Retrieval Agent --------------------
def retrieve_docs(query, k=3):
    query_embedding = embed_model.encode([query]).astype("float32")
    _, indices = index.search(query_embedding, k)
    return [doc_map[i] for i in indices[0]]

def fetch_knowledge(query):
    relevant_docs = retrieve_docs(query)
    context = "\n".join([f"Name: {doc['title']}, Link: {doc['link']}, Content: {doc['content']}" for doc in relevant_docs])
    return context

knowledge_tool = Tool(
    name="Women's Safety Knowledge Base",
    func=fetch_knowledge,
    description="Retrieve knowledge related to women's safety, protection, and empowerment."
)

# -------------------- Crime Statistics Agent --------------------
def fetch_crime_statistics(_):
    crime_stats = """
    1. UN Report (2023): 1 in 3 women worldwide experience physical or sexual violence.
    2. India NCRB (2022): Over 400,000 cases of crimes against women reported.
    3. US DOJ: 81% of women have experienced sexual harassment in their lifetime.
    4. UK ONS (2023): 1.6 million women faced domestic abuse last year.
    """
    return crime_stats

crime_statistics_tool = Tool(
    name="Crime Statistics Tool",
    func=fetch_crime_statistics,
    description="Provides statistics about crimes against women globally."
)

# -------------------- Web Search Agent --------------------
def web_search(query):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": SERPER_API_KEY}
    payload = {"q": query, "gl": "us"}

    response = requests.post(url, json=payload, headers=headers)
    data = response.json()

    if "organic" in data:
        results = "\n".join([f"{item['title']}: {item['link']}" for item in data["organic"][:3]])
        return results
    return "No results found."

web_search_tool = Tool(
    name="Web Search Agent",
    func=web_search,
    description="Performs Google searches when queries are out of scope."
)

# -------------------- Gemini API Wrapper --------------------
def gemini_generate(prompt):
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text if response else "Error generating response."

# Initialize the custom Gemini model
llm = GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GEMINI_API_KEY)

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize agent with tools
agent = initialize_agent(
    tools=[knowledge_tool, crime_statistics_tool, web_search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# -------------------- API Route --------------------
@app.post("/chat/")
def chat_with_agent(request: QueryRequest):
    response_content = agent.run(request.query)
    return {"response": response_content}

# -------------------- Run the FastAPI Server --------------------
# Start the server using: uvicorn main:app --reload
