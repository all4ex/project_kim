from telegram import Update, BotCommand
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from src.rag_system import RAGSystem
import os
from dotenv import load_dotenv

load_dotenv()

# инициализируем RAG систему
rag = RAGSystem(os.getenv("OPENAI_API_KEY"))

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start"""
    welcome_text = """🤖 Привет! Я RAG-бот для поиска по документам.

📋 Команды:
/start - это сообщение  
/reload - перезагрузить документы из папки
/stats - статистика системы
/clear - очистить историю чата

💡 Просто напиши любой вопрос, и я найду ответ в загруженных документах!"""

    await update.message.reply_text(welcome_text)

async def reload_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /reload"""
    loading_msg = await update.message.reply_text("🔄 Перезагружаю документы...")
    
    success, message = rag.reload_documents()
    
    if success:
        await loading_msg.edit_text(f"✅ {message}")
    else:
        await loading_msg.edit_text(f"❌ {message}")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /stats"""
    stats = rag.get_stats()
    
    stats_text = f"""📊 Статистика системы:

📄 Всего частей документов: {stats['total_chunks']}
📁 Всего файлов: {stats['total_sources']}

📚 Загруженные файлы:"""
    
    if stats['sources']:
        for source in stats['sources']:
            stats_text += f"\n• {source}"
    else:
        stats_text += "\n(документы не загружены)"
    
    await update.message.reply_text(stats_text)

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /clear"""
    user_id = update.effective_user.id
    if user_id in rag.chat_history:
        del rag.chat_history[user_id]
    
    await update.message.reply_text("🗑️ История чата очищена")

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка вопросов пользователя"""
    question = update.message.text
    user_id = update.effective_user.id
    
    # проверяем есть ли документы
    if len(rag.vector_store.documents) == 0:
        await update.message.reply_text(
            "⚠️ Документы не загружены. Используй команду /reload для загрузки документов из папки."
        )
        return
    
    # показываем что обрабатываем
    thinking_msg = await update.message.reply_text("🤔 Ищу ответ в документах...")
    
    # получаем ответ от RAG системы
    result = rag.ask_question(user_id, question)
    
    if result['success']:
        response = f"🤖 {result['answer']}"
        
        if result['sources']:
            sources_text = ", ".join(result['sources'])
            response += f"\n\n📚 Источники: {sources_text}"
            response += f"\n📊 Найдено документов: {result['found_docs']}"
        
        await thinking_msg.edit_text(response)
    else:
        await thinking_msg.edit_text(f"❌ {result['answer']}")

def create_bot():
    """Создает и настраивает бота"""
    bot_token = os.getenv("BOT_TOKEN")
    
    if not bot_token:
        raise ValueError("BOT_TOKEN не найден в .env файле")
    
    app = Application.builder().token(bot_token).build()
    
    # регистрируем обработчики команд
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("reload", reload_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("clear", clear_command))
    
    # обработчик текстовых сообщений
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))
    
    return app
