from langchain.llms import Bedrock
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = Bedrock(
    credentials_profile_name="william",
    model_id="amazon.titan-text-express-v1",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)
conversation = ConversationChain(
    llm=llm, verbose=True, memory=ConversationBufferMemory()
)
response = conversation.predict(input="How to set up a new profile for boto3?")