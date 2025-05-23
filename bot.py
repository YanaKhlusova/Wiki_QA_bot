import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain import HuggingFacePipeline, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import CharacterTextSplitter

# hf authentication for gated models, may be skipped
# import os
# from huggingface_hub import login
# login(token=os.environ_get("HF_TOKEN"))

# choose the model name and paste it there
MODEL_NAME=("MODEL_NAME")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
    )

model_4bit = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=quantization_config
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
qa_pipeline = pipeline(
        "text-generation",
        model=model_4bit,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=1024,
        do_sample=True,
        top_k=5,
        top_p=0.1,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
        )

llm = HuggingFacePipeline(pipeline=qa_pipeline)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
embeddings = HuggingFaceEmbeddings()

def answer_question(question: str, topic: str):
  """
  receives question and the topik of it from user,
  loads the necessary article from Wikipedia
  and generates an answer
  """
  loader = WikipediaLoader(query=topic, load_max_docs=1)
  documents = loader.load()
  docs = text_splitter.split_documents(documents)
  db = FAISS.from_documents(documents, embeddings)
  retrieval_qa = RetrievalQA.from_llm(
      retriever=db.as_retriever(),
      llm=llm
  )
  return retrieval_qa(question)["result"]

def main():
    topic = input("Укажите тему вопроса: ")
    question = input("Ваш вопрос: ")

    print("\nДумаю...\n")
    answer = answer_question(question, topic)

    print("Ответ:\n", answer)

if __name__ == "__main__":
    main()
