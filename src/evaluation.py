import json
from src.rag_pipeline import RAGPipeline

class SimpleEvaluator:
    def __init__(self):
        self.rag = RAGPipeline()
        
    def create_test_questions(self):
        """Создает тестовые вопросы (замените на свои)"""
        test_qa = [
            {
                "question": "О чем этот документ?",
                "expected_answer": "Описание содержания документа"
            },
            # добавьте больше вопросов
        ]
        
        with open('data/test_qa.json', 'w', encoding='utf-8') as f:
            json.dump(test_qa, f, ensure_ascii=False, indent=2)
    
    def run_evaluation(self):
        """Запускает оценку"""
        # подготавливаем документы
        self.rag.prepare_documents()
        
        # загружаем тестовые вопросы
        try:
            with open('data/test_qa.json', 'r', encoding='utf-8') as f:
                test_qa = json.load(f)
        except:
            print("Создайте файл data/test_qa.json с тестовыми вопросами")
            return
        
        results = []
        for item in test_qa:
            question = item['question']
            result = self.rag.ask(question)
            
            results.append({
                'question': question,
                'answer': result['answer'] if isinstance(result, dict) else result,
                'sources': result.get('sources', []) if isinstance(result, dict) else []
            })
        
        # сохраняем результаты
        with open('data/evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Оценка завершена. Результаты в data/evaluation_results.json")

if __name__ == "__main__":
    evaluator = SimpleEvaluator()
    evaluator.run_evaluation()
