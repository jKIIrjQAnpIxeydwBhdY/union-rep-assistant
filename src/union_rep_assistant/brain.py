import logging
from pathlib import Path
from typing import Dict, List, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Response(BaseModel):
    response: str = Field(description="LLM response")
    source_text: str = Field(
        description="original text from union contract that informed LLM response"
    )
    page_no: int = Field(description="meta data page_no for source_text")

    
# Define the state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]  # Automatically accumulates messages
    context: List[str]  # Add context to the state

class UnionRep:
    def __init__(
        self, model: str, temperature: int, openai_api_key: str, top_k: int = 3
    ):
        # Initialize LLM and embeddings
        self._llm = ChatOpenAI(
            model=model, temperature=temperature, openai_api_key=openai_api_key
        ).with_structured_output(Response)
        self._embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self._vector_store = FAISS.load_local(
            Path(__file__).parent / "faiss_index",
            self._embeddings,
            allow_dangerous_deserialization=True,
        )
        self._retriever = self._vector_store.as_retriever(search_kwargs={"k": top_k})

        # Define the prompt
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a union contract assistant. Use the following context and conversation history to answer questions:

            Context:
            {context}

            Conversation History:
            {history}

            Current Question:
            {question}

            If the question is unrelated to the union contract and conversation history , politely decline to answer."""),
            MessagesPlaceholder(variable_name="messages"),
        ])

        # Define the RAG chain
        self._rag_chain = self._prompt | self._llm

        # Initialize LangGraph workflow
        self._workflow = StateGraph(AgentState)

        # Add nodes
        self._workflow.add_node("retrieve", self.retrieve)
        self._workflow.add_node("generate", self.generate)
        self._workflow.add_node("update_memory", self.update_memory)

        # Define the flow
        self._workflow.set_entry_point("retrieve")
        self._workflow.add_edge("retrieve", "generate")
        self._workflow.add_edge("generate", "update_memory")
        self._workflow.add_edge("update_memory", END)

        # Compile the workflow with memory
        self._app = self._workflow.compile(
            checkpointer=MemorySaver(),
            interrupt_before=["update_memory"]
        )

    def retrieve(self, state: AgentState) -> Dict:
        """Retrieve relevant context for the question."""
        query = state["messages"][-1].content  # Get the latest user message
        context = self._retriever.invoke(query)
        return {"context": [doc.page_content for doc in context]}  # Add context to the state

    def generate(self, state: AgentState) -> Dict:
        """Generate a response using the RAG chain."""
        query = state["messages"][-1].content
        context = state["context"]
        history = state["messages"][:-1]  # Exclude the latest user message

        # Format history for the prompt
        history_str = "\n".join(f"{msg.type}: {msg.content}" for msg in history)

        # Invoke the RAG chain
        response = self._rag_chain.invoke({
            "context": "\n".join(context),  # Join context documents into a single string
            "history": history_str,
            "question": query,
            "messages": [state["messages"][-1]]  # Pass the latest user message
        })

        # Add the assistant's response to the state
        return {"messages": [AIMessage(content=response.response)]}

    def update_memory(self, state: AgentState) -> Dict:
        """Update memory with the latest interaction."""
        return state  # Memory is automatically handled by the checkpointer

    def ask(self, query: str, thread_id: str = "default") -> str:
        """Ask the chatbot a question and return the response."""
        logger.info("Question asked: %s", query)

        # Create initial state with the user's message
        initial_state = {"messages": [HumanMessage(content=query)], "context": []}

        # Execute the workflow
        config = {"configurable": {"thread_id": thread_id}}  # Use thread_id for conversation history
        for event in self._app.stream(initial_state, config):
            for key, value in event.items():
                if key == "generate":
                    response = value["messages"][-1].content

        # Format the response
        return f"Here’s what I found:\n\n{response}"

        
        # formatted_response = (
        #     f"Here’s what I found:\n\n"
        #     f"{escape_markdown(response.response)}\n\n"
        #     f"**Source:** {escape_markdown(response.source_text)}\n\n"
        #     f"**Contract Page Number:** {response.page_no}"
        # )

        # return formatted_response
    
