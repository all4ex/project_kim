import openai
from src.document_processor import DocProcessor
from src.vector_store import FAISSVectorStore
from typing import Dict, List

class RAGSystem:
    def __init__(self, openai_api_key: str):
        openai.api_key = openai_api_key
        
        self.doc_processor = DocProcessor()
        self.vector_store = FAISSVectorStore()
        self.chat_history = {}  # история чатов по user_id
    
    def reload_documents(self):
        """Перезагружает документы"""
        try:
            # очищаем старый индекс
            self.vector_store.clear()
            
            # загружаем документы
            docs = self.doc_processor.load_all_documents()
            
            if not docs:
                return False, "Документы не найдены в папке data/documents/"
            
            # добавляем в векторное хранилище
            self.vector_store.add_documents(docs)
            
            return True, f"Загружено {len(docs)} частей документов"
            
        except Exception as e:
            return False, f"Ошибка загрузки: {str(e)}"
    
    def ask_question(self, user_id: int, question: str) -> Dict:
        """Отвечает на вопрос пользователя"""
        try:
            # поиск релевантных документов
            relevant_docs = self.vector_store.search(question, top_k=3)
            
            if not relevant_docs:
                return {
                    'answer': 'К сожалению, я не нашел релевантной информации в документах.',
                    'sources': [],
                    'success': False
                }
            
            # собираем контекст
            context = self._build_context(relevant_docs)
            sources = self._get_sources(relevant_docs)
            
            # получаем историю чата
            history = self.chat_history.get(user_id, [])
            
            # генерируем ответ
            answer = self._generate_answer(question, context, history)
            
            # обновляем историю
            self._update_chat_history(user_id, question, answer)
            
            return {
                'answer': answer,
                'sources': sources,
                'found_docs': len(relevant_docs),
                'success': True
            }
            
        except Exception as e:
            return {
                'answer': f'Произошла ошибка: {str(e)}',
                'sources': [],
                'success': False
            }
    
    def _build_context(self, docs: List[Dict]) -> str:
        """Собирает контекст из найденных документов"""
        context_parts = []
        
        for i, doc in enumerate(docs):
            score = doc.get('similarity_score', 0)
            source = doc['source']
            text = doc['text']
            
            context_parts.append(f"Документ {i+1} ({source}, релевантность: {score:.2f}):\n{text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _get_sources(self, docs: List[Dict]) -> List[str]:
        """Извлекает уникальные источники"""
        return list(set([doc['source'] for doc in docs]))
    
    def _generate_answer(self, question: str, context: str, history: List[Dict]) -> str:
        """Генерирует ответ через OpenAI"""
        
        messages = [
            {
                "role": "system", 
                "content": """Ты помощник, который отвечает на вопросы на основе предоставленных документов.

ПРАВИЛА:
- Отвечай только на основе предоставленного контекста
- Если информации нет в контексте, честно скажи об этом  
- Будь кратким и точным
- Всегда указывай на какие источники ссылаешься
- Отвечай на русском языке"""
            }
        ]
        
        # добавляем последние сообщения истории
        if history:
            messages.extend(history[-4:])  # последние 2 пары вопрос-ответ
        
        # добавляем текущий вопрос с контекстом
        user_content = f"""КОНТЕКСТ:
{context}

ВОПРОС: {question}"""

        messages.append({"role": "user", "content": user_content})
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.1,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def _update_chat_history(self, user_id: int, question: str, answer: str):
        """Обновляет историю чата"""
        if user_id not in self.chat_history:
            self.chat_history[user_id] = []
        
        self.chat_history[user_id].extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ])
        
        # ограничиваем историю
        if len(self.chat_history[user_id]) > 10:
            self.chat_history[user_id] = self.chat_history[user_id][-10:]
    
    def get_stats(self) -> Dict:
        """Возвращает статистику системы"""
        total_docs = len(self.vector_store.documents)
        sources = list(set([doc['source'] for doc in self.vector_store.documents]))
        
        return {
            'total_chunks': total_docs,
            'total_sources': len(sources),
            'sources': sources
        }
