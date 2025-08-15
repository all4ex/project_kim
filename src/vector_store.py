import faiss
import numpy as np
import json
import os
import openai
from typing import List, Dict

class FAISSVectorStore:
    def __init__(self, store_path="data/vectors"):
        self.store_path = store_path
        self.dimension = 1536  # размер OpenAI embeddings
        self.index = None
        self.documents = []
        self.index_file = os.path.join(store_path, "faiss.index")
        self.docs_file = os.path.join(store_path, "documents.json")
        
        os.makedirs(store_path, exist_ok=True)
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Загружает существующий индекс или создает новый"""
        if os.path.exists(self.index_file) and os.path.exists(self.docs_file):
            print("Загружаю существующий индекс...")
            self.index = faiss.read_index(self.index_file)
            
            with open(self.docs_file, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
                
            print(f"Загружено: {len(self.documents)} документов")
        else:
            print("Создаю новый индекс...")
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product для cosine similarity
            self.documents = []
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Создает embeddings через OpenAI API"""
        embeddings = []
        batch_size = 20
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = openai.embeddings.create(
                    input=batch,
                    model="text-embedding-ada-002"
                )
                
                for item in response.data:
                    embeddings.append(item.embedding)
                    
                print(f"Создано embeddings: {len(embeddings)}/{len(texts)}")
                
            except Exception as e:
                print(f"Ошибка создания embeddings: {e}")
                raise
                
        return embeddings
    
    def add_documents(self, docs: List[Dict]):
        """Добавляет документы в индекс"""
        if not docs:
            return
            
        print("Создаю embeddings для документов...")
        texts = [doc['text'] for doc in docs]
        embeddings = self.create_embeddings(texts)
        
        # нормализуем для cosine similarity
        embeddings_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)
        
        # добавляем в индекс
        self.index.add(embeddings_array)
        self.documents.extend(docs)
        
        # сохраняем
        self._save_index()
        print(f"Добавлено документов: {len(docs)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Ищет похожие документы"""
        if self.index.ntotal == 0:
            return []
            
        # создаем embedding для запроса
        query_embedding = self.create_embeddings([query])[0]
        query_vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)
        
        # поиск
        scores, indices = self.index.search(query_vector, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['similarity_score'] = float(score)
                results.append(doc)
        
        return results
    
    def _save_index(self):
        """Сохраняет индекс и документы"""
        faiss.write_index(self.index, self.index_file)
        
        with open(self.docs_file, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
    
    def clear(self):
        """Очищает индекс"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        self._save_index()
