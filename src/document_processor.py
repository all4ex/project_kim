import os
import PyPDF2
from docx import Document
import re
import json

class DocProcessor:
    def __init__(self, docs_path="data/documents"):
        self.docs_path = docs_path
        
    def load_all_documents(self):
        """Загружает все документы из папки"""
        docs = []
        
        if not os.path.exists(self.docs_path):
            os.makedirs(self.docs_path)
            return docs
            
        for filename in os.listdir(self.docs_path):
            if filename.startswith('.'):
                continue
                
            filepath = os.path.join(self.docs_path, filename)
            text = self._extract_text(filepath, filename)
            
            if text:
                chunks = self._split_into_chunks(text)
                for i, chunk in enumerate(chunks):
                    docs.append({
                        'id': f"{filename}_chunk_{i}",
                        'text': chunk,
                        'source': filename,
                        'chunk_num': i,
                        'total_chunks': len(chunks)
                    })
                print(f"Обработан: {filename} -> {len(chunks)} частей")
                
        return docs
    
    def _extract_text(self, filepath, filename):
        """Извлекает текст из файла"""
        try:
            if filename.lower().endswith('.pdf'):
                return self._read_pdf(filepath)
            elif filename.lower().endswith('.docx'):
                return self._read_docx(filepath)
            elif filename.lower().endswith('.txt'):
                return self._read_txt(filepath)
        except Exception as e:
            print(f"Ошибка чтения {filename}: {e}")
        return ""
    
    def _read_pdf(self, filepath):
        with open(filepath, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _read_docx(self, filepath):
        doc = Document(filepath)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    
    def _read_txt(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _split_into_chunks(self, text, chunk_size=1000, overlap=200):
        """Разбивает текст на части с перекрытием"""
        # очищаем текст
        text = re.sub(r'\s+', ' ', text.strip())
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
                
        return chunks
