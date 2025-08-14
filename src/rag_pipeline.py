import openai
from src.embeddings import EmbeddingManager
from src.data_loader import DocLoader
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class RAGPipeline:
    def __init__(self):
        self.loader = DocLoader()
        self.embedding_manager = EmbeddingManager()
        self.chat_history = []
    
    def prepare_documents(self):
        """Подготавливает документы - загружает и индексирует"""
        print("Загружаем документы...")
        docs = self.loader.load_all_docs()
        
        if not docs:
            print("Документы не найдены!")
            return False
        
        # режем на куски и создаем эмбеддинги
        all_chunks = []
        all_metadatas = []
        
        for doc in docs:
            chunks = self.loader.split_text(doc['text'])
            all_chunks.extend(chunks)
            
            for i, chunk in enumerate(chunks):
                all_metadatas.append({
                    'filename': doc['filename'],
                    'chunk_id': i,
                    'chunk_count': len(chunks)
                })
        
        print(f"Создаем эмбеддинги для {len(all_chunks)} кусков...")
        embeddings = self.embedding_manager.create_embeddings(all_chunks)
        
        if embeddings:
            self.embedding_manager.add_to_db(all_chunks, embeddings, all_metadatas)
            print("Документы проиндексированы!")
            return True
        return False
    
    def ask(self, question):
        """Отвечает на вопрос"""
        print(f"\nВопрос: {question}")
        
        # ищем релевантные документы
        search_results = self.embedding_manager.search(question)
        
        if not search_results['documents'][0]:
            return "Не найдено релевантной информации"
        
        # собираем контекст
        context = "\n---\n".join(search_results['documents'][0])
        sources = [meta['filename'] for meta in search_results['metadatas']]
        
        # создаем промпт
        prompt = f"""
Ответь на вопрос на основе предоставленного контекста. 
Если информации нет в контексте, скажи об этом.

Контекст:
{context}

Вопрос: {question}

Ответ:
"""
        
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Ты помощник который отвечает только на основе предоставленного контекста."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            
            # выводим результат
            print(f"Ответ: {answer}")
            print(f"Источники: {set(sources)}")
            
            return {
                'answer': answer,
                'sources': list(set(sources)),
                'context_used': len(search_results['documents'][0])
            }
            
        except Exception as e:
            print(f"Ошибка генерации ответа: {e}")
            return "Ошибка при генерации ответа"
