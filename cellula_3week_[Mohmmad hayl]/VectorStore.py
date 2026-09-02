# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import TokenTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# # Loading
# loader = TextLoader(r"c:\Users\User\Documents\Cellula\cellula_3week_[Mohmmad hayl]\aboutme.txt")
# document = loader.load()

# # Chunking
# splitter = TokenTextSplitter(chunk_size=200, chunk_overlap=20)
# chunks = splitter.split_documents(document)

# # Embedding
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2",
#     model_kwargs={"device": "cpu"}
# )
# vectors = embeddings.embed_documents([i.page_content for i in chunks])

# # Vector store
# vectorDB = FAISS.from_documents(chunks, embeddings)

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os


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


if __name__ == "__main__":
    builder = VectorDBBuilder(
        file_path=r"c:\Users\User\Documents\Cellula\cellula_3week_[Mohmmad hayl]\aboutme.txt"
    )
    builder.build()
    builder.save(r"c:\Users\User\Documents\Cellula\cellula_3week_[Mohmmad hayl]\faiss_index")# persists to disk so you don't rebuild every run
