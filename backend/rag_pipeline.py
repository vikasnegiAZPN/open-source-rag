import os
import pandas as pd
from pathlib import Path
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.schema import Document
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(
        self,
        model_name: str = "mistral",
        embedding_model: str = "all-MiniLM-L6-v2",
        chroma_db_dir: str = "./chroma_db",
        kb_dir: str = "./knowledge_base",
        temperature: float = 0.7
    ):
        """Initialize RAG Pipeline"""
        
        self.model_name = model_name
        self.embedding_model_name = embedding_model
        self.chroma_db_dir = chroma_db_dir
        self.kb_dir = kb_dir
        self.temperature = temperature
        
        Path(self.kb_dir).mkdir(exist_ok=True, parents=True)
        Path(self.chroma_db_dir).mkdir(exist_ok=True, parents=True)
        
        logger.info("🚀 Initializing RAG Pipeline...")
        
        logger.info(f"📦 Loading embeddings: {embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        logger.info(f"🤖 Connecting to Ollama: {model_name}")
        self.llm = Ollama(
            model=model_name,
            base_url="http://localhost:11434",
            temperature=temperature
        )
        
        self.vectorstore = None
        self.qa_chain = None
        
        logger.info("✅ RAG Pipeline ready!")
    
    def ingest_excel(self, excel_path: str, file_name: str = None) -> bool:
        """Load Excel and create embeddings"""
        try:
            if file_name is None:
                file_name = Path(excel_path).stem
            
            logger.info(f"📄 Loading: {excel_path}")
            df = pd.read_excel(excel_path)
            logger.info(f"✅ Loaded {len(df)} rows")
            
            local_path = os.path.join(self.kb_dir, f"{file_name}.csv")
            df.to_csv(local_path, index=False)
            
            documents = self._convert_df_to_documents(df, file_name)
            logger.info(f"📚 Created {len(documents)} documents")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""]
            )
            split_docs = text_splitter.split_documents(documents)
            logger.info(f"✂️ Split into {len(split_docs)} chunks")
            
            logger.info("🔄 Creating embeddings...")
            start_time = time.time()
            
            collection_name = file_name.replace(" ", "_").lower()
            
            self.vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory=self.chroma_db_dir,
                collection_name=collection_name
            )
            self.vectorstore.persist()
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Done in {elapsed:.2f}s!")
            
            self._create_qa_chain()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            raise
    
    def _convert_df_to_documents(self, df: pd.DataFrame, source: str) -> List[Document]:
        """Convert DataFrame to Documents"""
        documents = []
        
        for idx, row in df.iterrows():
            content = "\n".join(
                [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
            )
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": source,
                    "row": idx,
                    "title": str(row.iloc[0]) if len(row) > 0 else f"Row {idx}"
                }
            )
            documents.append(doc)
        
        return documents
    
    def _create_qa_chain(self):
        """Create QA chain"""
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized!")
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            return_source_documents=True,
            verbose=False
        )
        logger.info("✅ QA chain created")
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """Query knowledge base"""
        if self.qa_chain is None:
            return {
                "error": "Knowledge base not loaded!",
                "status": "error"
            }
        
        try:
            start_time = time.time()
            logger.info(f"❓ Query: {question}")
            
            self.qa_chain.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k}
            )
            
            result = self.qa_chain({"query": question})
            
            sources = []
            for doc in result.get("source_documents", []):
                sources.append({
                    "source": doc.metadata.get("source"),
                    "row": doc.metadata.get("row"),
                    "content": doc.page_content[:300],
                    "similarity_score": None
                })
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            return {
                "answer": result["result"],
                "sources": sources,
                "status": "success",
                "elapsed_ms": elapsed_ms
            }
        
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return {"error": str(e), "status": "error"}
    
    def load_existing_kb(self, collection_name: str = None) -> bool:
        """Load existing KB"""
        try:
            logger.info(f"📂 Loading: {collection_name}")
            
            self.vectorstore = Chroma(
                persist_directory=self.chroma_db_dir,
                embedding_function=self.embeddings,
                collection_name=collection_name or "default"
            )
            self._create_qa_chain()
            
            logger.info("✅ Loaded!")
            return True
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return False
