# reranker.py

import logging
from typing import List

from langchain.docstore.document import Document
from sentence_transformers import CrossEncoder

from config import APP_CONFIG
from model_registry import ModelRegistry

class ReRanker:
    """
    Cross-Encoder modelini kullanarak dokümanları yeniden sıralar.
    Bu sınıf, arama sonuçlarının hassasiyetini artırmak için kritik bir rol oynar.
    """
    def __init__(self, models: ModelRegistry):
        self._config = APP_CONFIG
        self._logger = logging.getLogger(self.__class__.__name__)
        self._cross_encoder = models.get_cross_encoder_model()

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Verilen bir sorgu ve doküman listesi için, dokümanları alaka düzeyine
        göre yeniden sıralar ve en iyi olanları döndürür.

        Args:
            query: Kullanıcının orijinal sorgusu.
            documents: Veritabanından gelen aday doküman listesi (retriever sonuçları).

        Returns:
            Yeniden sıralanmış ve en alakalı dokümanların listesi.
        """
        if not documents or not query:
            return []

        self._logger.info(f"Yeniden sıralama başlıyor. Aday doküman sayısı: {len(documents)}")
        
        # 1. Adım: Cross-Encoder için (sorgu, doküman_içeriği) çiftleri oluştur.
        sentence_pairs = [(query, doc.page_content) for doc in documents]
        
        # 2. Adım: Skorları hesapla.
        # Bu, modelin tüm çiftleri GPU üzerinde toplu olarak verimli bir şekilde
        # işlemesini sağlar.
        scores = self._cross_encoder.predict(sentence_pairs, show_progress_bar=False)
        
        # 3. Adım: Dokümanları ve hesaplanan skorlarını birleştir.
        scored_docs = list(zip(scores, documents))
        
        # 4. Adım: Dokümanları skorlarına göre büyükten küçüğe doğru sırala.
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # 5. Adım: Yapılandırmada belirtilen en iyi N dokümanı seç.
        top_n = self._config.RERANKER_TOP_N
        reranked_docs = [doc for score, doc in scored_docs[:top_n]]
        
        self._logger.info(f"Yeniden sıralama tamamlandı. En iyi {len(reranked_docs)} doküman seçildi.")
        
        # Hata ayıklama için en iyi dokümanların skorlarını ve bilgilerini logla
        if self._logger.isEnabledFor(logging.DEBUG):
            for i, (score, doc) in enumerate(scored_docs[:top_n]):
                self._logger.debug(
                    f"Sıra: {i+1}, Skor: {score:.4f}, Kaynak: {doc.metadata.get('source_file')}, Sayfa: {doc.metadata.get('page')}"
                )

        return reranked_docs