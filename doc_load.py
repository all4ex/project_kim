import PyPDF2
from docx import Document
from telegram import File

class DocumentLoader:
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt']
    
    async def process_telegram_document(self, bot, file_id, file_name):
        """Скачивает и обрабатывает документ из Telegram"""
        # Скачиваем файл
        file = await bot.get_file(file_id)
        file_path = f"data/documents/{file_name}"
        await file.download_to_drive(file_path)
        
        # Извлекаем текст в зависимости от формата
        if file_name.endswith('.pdf'):
            return self.extract_from_pdf(file_path)
        elif file_name.endswith('.docx'):
            return self.extract_from_docx(file_path)
        elif file_name.endswith('.txt'):
            return self.extract_from_txt(file_path)
    
    def extract_from_pdf(self, file_path):
        """Извлекает текст из PDF"""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
