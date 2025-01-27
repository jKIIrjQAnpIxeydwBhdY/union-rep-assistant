from pathlib import Path
import os 

from langchain_openai import ChatOpenAI
from langchain.schema import Document

from union_rep_assistant.utils import generate_contract_text, create_vector_store


PAGE_OFFSET = 0

def main():
  LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY")) # will require setting this env.
  CONTRACT_PATH = Path("src/union_rep_assistant/contract")
  SPLIT_CONTRACT_PATH = CONTRACT_PATH / "split_contract"
  VECTOR_STORE_PATH = Path("src/union_rep_assistant/faiss_index")

  CONTRACT_PATH.mkdir(parents=True, exist_ok=True)
  SPLIT_CONTRACT_PATH.mkdir(parents=True, exist_ok=True)
  VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

  docs = generate_contract_text(
    CONTRACT_PATH / "2024-2028-RAB-Security-Officers-Owners-Agreement-Executed.pdf",
    CONTRACT_PATH / "2024-2028-RAB-Security-Officers-Owners-Agreement-Executed.txt",
    LLM
  )

  for i, doc in enumerate(docs):
    (SPLIT_CONTRACT_PATH / f"page_{i+ PAGE_OFFSET}").write_text(doc, encoding="utf-8")

    docs_with_page_number = [
      Document(page_content=doc, metadata={"page_number": i+ PAGE_OFFSET})  
      for i, doc in enumerate(docs)
  ]
    # i-3 is specific to this contract.  The reason for this is because
      #  this contract has a title and table of contents which shouldn't count as pages.  With -1 
      #  the page number of the document will reflect the page number in the contract. 
  vector_store = create_vector_store(docs_with_page_number,VECTOR_STORE_PATH)

if __name__ == "__main__":
  main()