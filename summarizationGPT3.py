from langchain import PromptTemplate
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain

from youtube_transcript_api import YouTubeTranscriptApi

from dotenv import load_dotenv
import os

load_dotenv()

raw_transcript = YouTubeTranscriptApi.get_transcript("Pp2wbyLoEtM")
transcript = ""
for i in raw_transcript:
    transcript = transcript + i["text"]

text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = text_splitter.split_text(transcript)

docs = [Document(page_content=t) for t in chunks]

# prompt_template = """Write a concise summary of the following transcript of a youtube video:
#
# {text}
#
# """
# prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

llm = OpenAI(temperature=0, openai_api_key=os.getenv("API_KEY"))

chain = load_summarize_chain(llm, chain_type="map_reduce")
summary = chain.run(docs)

print(summary)
