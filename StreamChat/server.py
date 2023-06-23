import asyncio
import uvicorn

from typing import AsyncIterable, Awaitable
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

load_dotenv()
async def wait_done(fn: Awaitable, event: asyncio.Event):
    try:
        await fn
    except Exception as e:
        print(e)
        event.set()
    finally:
        event.set()

async def call_openai(question: str) -> AsyncIterable[str]:
    callback = AsyncIteratorCallbackHandler()
    model = ChatOpenAI(streaming=True, verbose=True, callbacks=[callback])

    coroutine = wait_done(model.agenerate(messages=[[HumanMessage(content=question)]]), callback.done)
    task = asyncio.create_task(coroutine)

    async for token in callback.aiter():
        yield f"{token}"

    await task


app = FastAPI()

@app.post("/ask")
def ask(body: dict):
    return StreamingResponse(call_openai(body['question']), media_type="text/event-stream")

@app.get("/")
async def homepage():
    return FileResponse('statics/index.html')

if __name__ == "__main__":
    uvicorn.run(host="0.0.0.0", port=8888, app=app)