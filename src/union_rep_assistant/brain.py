from langchain_core.prompts import ChatPromptTemplate


class UnionRep:
    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        self.prompt = ChatPromptTemplate.from_template("""Answer the question based only on the following context:
            {context}
            Question: {question}
            """)
        self.rag_chain = self.prompt | self.llm

    def ask(self, query: str) -> str:
        context = self.retriever.invoke(query)
        return self.rag_chain.invoke({"context": context, "question": query}).content
