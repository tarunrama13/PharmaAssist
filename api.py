# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from fastapi import FastAPI, File,Form
from chatbotllm import answer_question
from starlette.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Blazor WebAssembly app's origin
    allow_credentials=True,  # Adjust as needed for cookies/session data
    allow_methods=["*"],  # Adjust for specific allowed methods (GET, POST, etc.)
    allow_headers=["*"],  # Adjust for specific allowed headers (Content-Type, etc.)
    
)

@app.post("/getResponse/")
async def first_api(query:str=Form(...)):
    print("received request to fast api")
    result=answer_question(query)
    return result
