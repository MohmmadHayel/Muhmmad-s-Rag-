from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(
    api_key="api_key",
    base_url="https://openrouter.ai/api/v1",
    model="openrouter/free",
    temperature=0.8
)


class VectorDBBuilder:
    def __init__(self, file_path, chunk_size=200, chunk_overlap=20,
                 model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device}
        )
        self.vectorDB = None

    def load_document(self):
        loader = TextLoader(self.file_path)
        return loader.load()

    def chunk_document(self, document):
        splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return splitter.split_documents(document)

    def build(self):
        document = self.load_document()
        chunks = self.chunk_document(document)
        self.vectorDB = FAISS.from_documents(chunks, self.embeddings)
        return self.vectorDB

    def save(self, path="faiss_index"):
        if self.vectorDB is None:
            raise ValueError("No vectorDB to save. Call build() first.")
        self.vectorDB.save_local(path)

    def load(self, path="faiss_index"):
        self.vectorDB = FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )
        return self.vectorDB

    def as_retriever(self, k=3):
        if self.vectorDB is None:
            raise ValueError("No vectorDB loaded. Call build() or load() first.")
        return self.vectorDB.as_retriever(search_kwargs={"k": k})


class RAGQueryEngine:
    def __init__(self, vectorDB, llm, k=4):
        self.vectorDB = vectorDB
        self.llm = llm
        self.k = k
        self.prompt_template = PromptTemplate.from_template(
            """you are an assistant answering questions about Mohammad Al-Hmoud's
professional profile (education, skills, projects, and career goal),
Answer the next Question using provided context,
If you don't know the answer, just say you don't know.
answer should be within 200 words or lower only

## context :
{context}

## Question :
{Question}"""
        )

    def retrieve(self, query):
        similar_docs = self.vectorDB.similarity_search_with_score(query, k=self.k)
        return [doc.page_content for doc, score in similar_docs]

    def build_prompt(self, query):
        context = self.retrieve(query)
        return self.prompt_template.format(context="\n".join(context), Question=query)
    def answer(self, query):
        prompt = self.build_prompt(query)
        response = self.llm.invoke(prompt).content
        return response

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    builder = VectorDBBuilder(
        file_path=os.path.join(script_dir, "aboutme.txt")
    )
    builder.build()
    builder.save(os.path.join(script_dir, "faiss_index"))

    rag = RAGQueryEngine(builder.vectorDB,llm, k=4)
    query = "what degree did Mohammad Al-Hmoud study?"
    prompt = rag.build_prompt(query)
    print(prompt)