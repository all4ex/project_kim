import os
import PyPDF2
from docx import Document
import re

class DocLoader:
    def __init__(self, docs_folder="data/documents"):
        self.docs_folder = docs_folder
        
    def load_all_docs(self):
        """Загружает все документы из папки"""
        docs = []
        
        for filename in os.listdir(self.docs_folder):
            filepath = os.path.join(self.docs_folder, filename)
            
            if filename.endswith('.pdf'):
                text = self._extract_pdf(filepath)
            elif filename.endswith('.docx'):
                text = self._extract_docx(filepath)
            elif filename.endswith('.txt'):
                text = self._extract_txt(filepath)
            else:
                continue
                
            if text:
                docs.append({
                    'filename': filename,
                    'text': text
                })
                print(f"Загружен: {filename}")
                
        return docs
    
    def _extract_pdf(self, filepath):
        """Достает текст из PDF"""
        try:
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
        except:
            print(f"Ошибка чтения PDF: {filepath}")
            return ""
    
    def _extract_docx(self, filepath):
        """Достает текст из DOCX"""
        try:
            doc = Document(filepath)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        except:
            print(f"Ошибка чтения DOCX: {filepath}")
            return ""
    
    def _extract_txt(self, filepath):
        """Достает текст из TXT"""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read()
        except:
            print(f"Ошибка чтения TXT: {filepath}")
            return ""
    
    def split_text(self, text, chunk_size=1000, overlap=200):
        """Режет текст на куски"""
        # убираем лишнее
        text = re.sub(r'\s+', ' ', text).strip()
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
                
        return chunks
