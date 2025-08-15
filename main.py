import asyncio
import os
from dotenv import load_dotenv
from src.bot import create_bot
from telegram import BotCommand

load_dotenv()

async def main():
    """Запуск бота"""
    print("🚀 Запускаю Telegram RAG-бота...")
    
    # проверяем переменные окружения
    if not os.getenv("BOT_TOKEN"):
        print("❌ BOT_TOKEN не найден в .env файле")
        return
        
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY не найден в .env файле")
        return
    
    # создаем папки если их нет
    os.makedirs("data/documents", exist_ok=True)
    os.makedirs("data/vectors", exist_ok=True)
    
    # создаем и запускаем бота
    app = create_bot()
    
    # устанавливаем меню команд
    commands = [
        BotCommand("start", "Начать работу с ботом"),
        BotCommand("reload", "Перезагрузить документы"),
        BotCommand("stats", "Показать статистику"),
        BotCommand("clear", "Очистить историю чата")
    ]
    
    await app.bot.set_my_commands(commands)
    
    print("✅ Бот запущен и готов к работе!")
    print("📁 Добавь документы в папку data/documents/ и используй команду /reload")
    
    # запускаем polling
    await app.run_polling()

if __name__ == "__main__":
    asyncio.run(main())
