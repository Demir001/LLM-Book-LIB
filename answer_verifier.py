# answer_verifier.py

import re
import logging
from typing import List, Set, Tuple

from langchain.docstore.document import Document

from config import RAGResult
from utils import ConsoleManager

class AnswerVerifier:
    """
    LLM tarafından üretilen cevabın kaynaklarını, sağlanan orijinal
    dokümanlarla karşılaştırarak doğrular ve sonucu kullanıcıya sunar.
    """
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def verify_and_present(self, final_answer: str, source_docs: List[Document], console: ConsoleManager):
        """
        Üretilen tam cevabı ve kaynaklarını alır, doğrular ve sonucu
        terminale biçimlendirilmiş bir şekilde yazdırır.
        
        Args:
            final_answer: LLM'den akıtılarak oluşturulan tam metin cevap.
            source_docs: RAGCore tarafından LLM'e kanıt olarak sunulan dokümanlar.
            console: Çıktıları biçimlendirmek için ConsoleManager nesnesi.
        """
        self._logger.info("Cevap doğrulama ve sunum işlemi başlatıldı...")
        
        # 1. Adım: Cevaptaki kaynak etiketlerini regex ile bul.
        # Format: [Kaynak: Kitap Başlığı - Yazar, Sayfa: X]
        pattern = r"\[Kaynak: (.*?)\s*-\s*(.*?),\s*Sayfa:\s*(\d+)\]"
        answer_references_raw = re.findall(pattern, final_answer)

        if not answer_references_raw:
            self._logger.warning("Cevapta hiçbir kaynak referansı bulunamadı.")
            self._present_unverified_answer(final_answer, source_docs, console)
            return

        # 2. Adım: Hem cevaptaki referansları hem de orijinal kaynakları
        # karşılaştırma için standart bir formata (tuple) getir.
        answer_references_set: Set[Tuple[str, str, str]] = {
            (title.strip(), author.strip(), page.strip()) 
            for title, author, page in answer_references_raw
        }
        
        source_docs_map = {
            (
                doc.metadata.get("title", "Bilinmiyor"),
                doc.metadata.get("author", "Bilinmiyor"),
                str(doc.metadata.get("page", 0))
            ): doc
            for doc in source_docs
        }

        # 3. Adım: Kümeleri karşılaştırarak doğrulanmış ve sahte kaynakları bul.
        verified_refs = answer_references_set.intersection(source_docs_map.keys())
        hallucinated_refs = answer_references_set.difference(source_docs_map.keys())

        verified_sources = [source_docs_map[ref] for ref in verified_refs]
        
        # 4. Adım: Sonuçları kullanıcıya sun.
        console.print_info("\n💡 YANIT:", "HEADER")
        print(console.color(final_answer, "BOLD"))
        
        if verified_sources:
            console.print_info("\n✅ DOĞRULANMIŞ KAYNAKLAR:", "GREEN")
            # Kaynakları sayfa numarasına göre sıralayarak sunmak daha okunaklı olur.
            verified_sources.sort(key=lambda d: (d.metadata.get('title'), d.metadata.get('page')))
            for doc in verified_sources:
                source_str = f"{doc.metadata.get('title')} - {doc.metadata.get('author')}"
                print(f"  - {source_str} (Sayfa: {doc.metadata.get('page')})")
        
        if hallucinated_refs:
            self._logger.warning(f"Halüsinasyon Tespiti! Şu referanslar kanıtlar arasında bulunamadı: {hallucinated_refs}")
            console.print_info("\n❌ DİKKAT: Aşağıdaki referanslar uydurulmuş olabilir!", "RED")
            for title, author, page in hallucinated_refs:
                print(f"  - [Uydurma Kaynak: {title} - {author}, Sayfa: {page}]")

    def _present_unverified_answer(self, answer: str, source_docs: List[Document], console: ConsoleManager):
        """Cevapta hiç referans bulunamadığı durum için özel sunum yapar."""
        console.print_info("\n💡 YANIT:", "HEADER")
        print(console.color(answer, "BOLD"))
        console.print_info("\n⚠️ Bu cevabın kaynakları otomatik olarak doğrulanamadı.", "YELLOW")
        
        if source_docs:
            console.print_info("Cevabı oluşturmak için kullanılan olası kaynaklar:", "CYAN")
            for doc in source_docs:
                source_str = f"{doc.metadata.get('title')} - {doc.metadata.get('author')}"
                print(f"  - {source_str} (Sayfa: {doc.metadata.get('page')})")