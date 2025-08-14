from src.rag_pipeline import RAGPipeline

def main():
    print("RAG Чат-бот")
    print("=" * 50)
    
    rag = RAGPipeline()
    
    # подготавливаем документы (делается один раз)
    print("Подготовка документов...")
    if not rag.prepare_documents():
        print("Не удалось подготовить документы")
        return
    
    print("\nГотов к работе! Задавайте вопросы (или 'exit' для выхода)")
    
    while True:
        question = input("\n> ").strip()
        
        if question.lower() in ['exit', 'quit', 'выход']:
            break
            
        if question:
            result = rag.ask(question)

if __name__ == "__main__":
    main()
