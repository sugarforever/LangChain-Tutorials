from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.schema import HumanMessage
import asyncio
import sys

load_dotenv()
handler = AsyncIteratorCallbackHandler()
llm = ChatOpenAI(streaming=True, callbacks=[handler], temperature=0) 

async def consumer():
    iterator = handler.aiter()
    async for item in iterator:
        sys.stdout.write(item)
        sys.stdout.flush()

if __name__ == '__main__':
    message = "What is AI?"
    loop = asyncio.get_event_loop()
    loop.create_task(llm.agenerate(messages=[[HumanMessage(content=message)]]))
    loop.create_task(consumer())
    loop.run_forever()
    loop.close()