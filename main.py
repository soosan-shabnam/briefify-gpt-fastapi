from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain

from youtube_transcript_api import YouTubeTranscriptApi

from fastapi import FastAPI

from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "backend is up and running"}


@app.get("/v1/api/{video_id}")
async def generate_summary(video_id):
    raw_transcript = YouTubeTranscriptApi.get_transcript(video_id)
    transcript = ""
    for i in raw_transcript:
        transcript = transcript + i["text"]

    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_text(transcript)

    docs = [Document(page_content=t) for t in chunks]

    llm = OpenAI(temperature=0, openai_api_key=os.getenv("API_KEY"))

    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)

    return {"summary": summary}
