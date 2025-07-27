from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
import os, shutil, uuid

os.environ["GOOGLE_API_KEY"] = "AIzaSyAdiK-m-fRxKo3HsGA-XThGeWDcQPQ8es0"


from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "🚀 Gemini backend is running!"}


# Allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionInput(BaseModel):
    question: str

db_store = {}

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    temp_filename = f"{uuid.uuid4()}.pdf"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    loader = PyPDFLoader(temp_filename)
    pages = loader.load_and_split()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_documents(chunks, embeddings)

    db_id = str(uuid.uuid4())
    db_store[db_id] = db

    os.remove(temp_filename)
    return {"db_id": db_id, "message": "PDF processed and stored."}

@app.post("/ask/")
async def ask_question(db_id: str = Form(...), question: str = Form(...)):
    if db_id not in db_store:
        return {"error": "Invalid DB ID"}

    docs = db_store[db_id].similarity_search(question)
    context = "\n\n".join([doc.page_content for doc in docs[:3]])

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    prompt = f"""Answer the question using only the context below:

    Context:
    {context}

    Question: {question}

    Answer in short and precise format:
    - Answer: [your answer]
    - Reference: [source/line if possible]"""

    response = llm.invoke(prompt)
    return {"answer": response.content}
