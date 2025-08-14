import openai
import chromadb
from dotenv import load_dotenv
import os
import time

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class EmbeddingManager:
    def __init__(self):
        # настройка chromadb
        self.client = chromadb.PersistentClient(path="data/chroma_db")
        self.collection = self.client.get_or_create_collection("documents")
        
    def create_embeddings(self, texts):
        """Создает эмбеддинги через OpenAI"""
        embeddings = []
        
        # обрабатываем по батчам чтобы не превысить лимиты
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = openai.embeddings.create(
                    input=batch,
                    model="text-embedding-ada-002"
                )
                
                for item in response.data:
                    embeddings.append(item.embedding)
                    
                time.sleep(0.1)  # небольшая пауза
                print(f"Обработано {len(embeddings)}/{len(texts)}")
                
            except Exception as e:
                print(f"Ошибка создания эмбеддингов: {e}")
                return []
                
        return embeddings
    
    def add_to_db(self, texts, embeddings, metadatas):
        """Добавляет в векторную БД"""
        ids = [f"doc_{i}" for i in range(len(texts))]
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Добавлено в БД: {len(texts)} документов")
    
    def search(self, query, n_results=5):
        """Ищет похожие документы"""
        # создаем эмбеддинг для запроса
        response = openai.embeddings.create(
            input=[query],
            model="text-embedding-ada-002"
        )
        query_embedding = response.data[0].embedding
        
        # ищем в БД
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        return results
