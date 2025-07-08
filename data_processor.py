# data_processor.py

import re
import logging
from pathlib import Path
from typing import List

import nltk
from langchain_community.document_loaders import PyMuPDFLoader

from config import APP_CONFIG, DocumentMetadata, EnrichedDocument

class DocumentProcessor:
    """
    Tek bir PDF dosyasını, halüsinasyonu en aza indirme hedefiyle işler.
    Yapısal gürültüyü temizler ve metni anlamsal bütünlüğünü koruyarak işler.
    """
    def __init__(self):
        self._config = APP_CONFIG
        self._logger = logging.getLogger(self.__class__.__name__)
        self._ensure_nltk_data()

    def _ensure_nltk_data(self):
        """NLTK'nın cümle ayırıcı modelinin kurulu olduğundan emin olur."""
        try:
            # Doğrudan modeli bulmaya çalış.
            nltk.data.find('tokenizers/punkt')
            self._logger.info("NLTK 'punkt' paketi zaten yüklü.")
        except LookupError:
            # Eğer bulunamazsa, standart 'LookupError' hatası verir.
            # Bu hatayı yakalayıp indirme işlemini başlat.
            self._logger.info("Gerekli NLTK 'punkt' veri paketi bulunamadı, indiriliyor...")
            try:
                nltk.download('punkt', quiet=True)
                self._logger.info("NLTK 'punkt' paketi başarıyla indirildi.")
            except Exception as e:
                # İnternet bağlantısı yoksa veya başka bir sorun olursa diye
                # indirme işlemini de bir hata yakalama bloğuna alalım.
                self._logger.critical(f"NLTK 'punkt' paketi indirilemedi! Cümle ayırma çalışmayabilir. Hata: {e}")
                # Bu kritik bir hata olduğu için uygulamayı durdurabiliriz.
                raise SystemError("NLTK veri paketi indirilemedi, uygulama devam edemiyor.") from e

    def process_pdf(self, pdf_path: Path, metadata: DocumentMetadata) -> List[EnrichedDocument]:
        """
        Bir PDF dosyasını ve ona ait meta veriyi alıp, her sayfası için
        temizlenmiş ve zenginleştirilmiş 'EnrichedDocument' nesneleri oluşturur.
        """
        self._logger.info(f"İçerik çıkarılıyor ve analiz ediliyor: '{pdf_path.name}'")
        try:
            pages = PyMuPDFLoader(str(pdf_path)).load()
            if not pages:
                self._logger.warning(f"'{pdf_path.name}' içinden hiç sayfa okunamadı.")
                return []

            all_enriched_docs: List[EnrichedDocument] = []
            for page_doc in pages:
                page_num = page_doc.metadata.get('page', 0) + 1
                
                # Katman 1: Agresif Gürültü Temizliği
                clean_content = self._clean_text(page_doc.page_content)
                if not clean_content:
                    continue

                # Katman 2: Cümle Bazında Anlamsal Paragraflar Oluşturma
                logical_paragraphs = self._create_semantic_paragraphs(clean_content)

                # Katman 3: Zenginleştirilmiş Dokümanları Oluşturma
                from dataclasses import asdict
                base_metadata = asdict(metadata)
                base_metadata['page'] = page_num
                
                for paragraph in logical_paragraphs:
                    # Her anlamlı paragraf için ayrı bir doküman nesnesi oluşturuyoruz.
                    all_enriched_docs.append(
                        EnrichedDocument(
                            page_content=paragraph,
                            metadata=base_metadata.copy() # Her biri için meta verinin kopyasını oluştur
                        )
                    )
            
            self._logger.info(f"'{pdf_path.name}' için {len(all_enriched_docs)} anlamlı metin bloğu oluşturuldu.")
            return all_enriched_docs

        except Exception as e:
            self._logger.error(f"'{pdf_path.name}' işlenirken kritik hata: {e}", exc_info=True)
            return []

    def _clean_text(self, text: str) -> str:
        """
        Verilen bir metin bloğunu yapısal gürültüden temizler.
        """
        if not text:
            return ""
        
        # Satır sonu tirelemesini kaldır ve kelimeleri birleştir (örn: "hi- kaye" -> "hikaye")
        text = re.sub(r'-\n\s*', '', text, re.MULTILINE)
        
        # Sadece sayfa numarası veya anlamsız karakterlerden oluşan satırları temizle
        lines = text.split('\n')
        cleaned_lines = [line for line in lines if not re.fullmatch(r'[\s\-\d\.]+', line.strip())]
        text = " ".join(cleaned_lines)
        
        # Çoklu boşluk karakterlerini tek bir boşluğa indirge
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _create_semantic_paragraphs(self, text: str) -> List[str]:
        """
        Bir metin bloğunu cümlelere ayırır ve anlamsal olarak tutarlı,
        yaklaşık olarak CHUNK_SIZE boyutunda paragraflar oluşturur.
        """
        try:
            sentences = nltk.sent_tokenize(text, language='turkish')
        except Exception as e:
            self._logger.warning(f"NLTK cümle ayırmada hata, metin bütün olarak alınıyor. Hata: {e}")
            # Hata durumunda, metni tek bir paragraf olarak döndür.
            return [text] if text else []
        
        paragraphs: List[str] = []
        current_paragraph = ""
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # Eğer bir sonraki cümleyi eklemek chunk boyutunu aşacaksa, mevcut paragrafı kaydet.
            if len(current_paragraph) + len(sentence) + 1 > self._config.CHUNK_SIZE:
                if current_paragraph:
                    paragraphs.append(current_paragraph.strip())
                current_paragraph = sentence
            else:
                # Cümleleri boşlukla birleştirerek paragrafa ekle.
                current_paragraph += " " + sentence
        
        # Döngü bittikten sonra kalan son paragrafı da listeye ekle.
        if current_paragraph:
            paragraphs.append(current_paragraph.strip())
            
        return paragraphs