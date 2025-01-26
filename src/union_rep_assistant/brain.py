import logging
from pathlib import Path

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


logging.basicConfig(level=logging.INFO)
logger = logger = logging.getLogger(__name__)


class Response(BaseModel):
    response: str = Field(description="LLM response")
    source_text: str = Field(
        description="original text from union contract that informated LLM response"
    )
    page_no: int = Field(description="meta data page_no for source_text")


class UnionRep:
    def __init__(
        self, model: str, temperature: int, openai_api_key: str, top_k: int = 3
    ):
        self._llm = ChatOpenAI(
            model=model, temperature=temperature, openai_api_key=openai_api_key
        ).with_structured_output(Response)  # TODO: should I set a max tokens?
        self._embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self._vector_store = FAISS.load_local(
            Path(__file__).parent / "faiss_index",
            self._embeddings,
            allow_dangerous_deserialization=True,
        )
        self._retriever = self._vector_store.as_retriever(search_kwargs={"k": top_k})
        self._prompt = self._prompt = (
            ChatPromptTemplate.from_template("""You are a helpful assistant that answers questions related to the union contract.
            Answer the question based on the following context and prior conversation history:

            Context:
            {context}

            Current Question:
            {question}

            If the question is unrelated to the union contract, respond with:
            - response: "This assistant is designed to answer questions related to the union contract. Please ask a question about the union contract."
            - source_text: "N/A"
            - page_no: "N/A"

            Your response should include:
            - A summary of the answer (`response`)
            - The original text from the context (`source_text`)
            - The page number for the source text (`page_no`) if available.
            """)
        )

        self._rag_chain = self._prompt | self._llm

    def ask(self, query: str) -> str:
        logger.info("question asked: %s", query)
        context = self._retriever.invoke(query)
        logger.info("context provided to question: %s", context)
        response = self._rag_chain.invoke({"context": context, "question": query})

        # Escape special Markdown characters
        def escape_markdown(text: str) -> str:
            return text.replace("$", "\\$").replace("*", "\\*").replace("_", "\\_")

        formatted_response = (
            f"Here’s what I found:\n\n"
            f"{escape_markdown(response.response)}\n\n"
            f"**Source:** {escape_markdown(response.source_text)}\n\n"
            f"**Contract Page Number:** {response.page_no}"
        )

        return formatted_response
