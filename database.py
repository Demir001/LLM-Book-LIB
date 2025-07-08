# database.py (YENİ HALİ - CHROMA İÇİN)

import logging
from typing import List, Dict, Any

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

from config import APP_CONFIG, DatabaseError, EnrichedDocument

class VectorDatabase:
    """
    Lokal ChromaDB vektör veritabanı ile tüm etkileşimleri yönetir.
    Veri ekleme, silme ve arama için arayüz sağlar.
    """
    def __init__(self, embedding_model: HuggingFaceEmbeddings):
        self._config = APP_CONFIG
        self._logger = logging.getLogger(self.__class__.__name__)
        self._embedding_model = embedding_model
        
        try:
            self._logger.info(f"ChromaDB veritabanı başlatılıyor. Veri yolu: {self._config.CHROMA_PERSIST_DIR}")
            # Veritabanını kalıcı olarak diskte saklamak için bir istemci oluştur
            self._client = chromadb.PersistentClient(path=str(self._config.CHROMA_PERSIST_DIR))
            
            # LangChain entegrasyonu için Chroma nesnesini oluştur
            self._db_store = Chroma(
                client=self._client,
                collection_name=self._config.CHROMA_COLLECTION_NAME,
                embedding_function=self._embedding_model
            )
            self._logger.info("ChromaDB başarıyla başlatıldı.")
            
        except Exception as e:
            raise DatabaseError(f"ChromaDB başlatılamadı. Hata: {e}", exc_info=True)

    def add_documents(self, enriched_docs: List[EnrichedDocument]):
        """
        İşlenmiş ve zenginleştirilmiş dokümanları alır ve ChromaDB'ye ekler.
        """
        if not enriched_docs:
            return

        source_file = enriched_docs[0].metadata.get('source_file', 'Bilinmeyen dosya')
        self._logger.info(f"'{source_file}' için {len(enriched_docs)} metin bloğu veritabanına ekleniyor...")

        texts = [doc.page_content for doc in enriched_docs]
        metadatas = [doc.metadata for doc in enriched_docs]
        
        try:
            # Chroma, metinleri ve meta verileri ayrı listeler olarak alır.
            self._db_store.add_texts(texts=texts, metadatas=metadatas)
            self._logger.info(f"'{source_file}' için ekleme işlemi başarıyla tamamlandı.")
        except Exception as e:
            raise DatabaseError(f"'{source_file}' belgeleri ChromaDB'ye eklenirken hata: {e}")

    def get_retriever(self, search_kwargs: Dict[str, Any]) -> Any:
        """Arama parametrelerine göre bir retriever nesnesi döndürür."""
        # ChromaDB'de hibrit arama bu şekilde doğrudan desteklenmez, sadece vektör araması yapar.
        # Ancak Re-ranker kullandığımız için bu bir sorun teşkil etmez.
        return self._db_store.as_retriever(search_kwargs=search_kwargs)

    def delete_collection(self):
        """Veritabanındaki koleksiyonu (tüm veriyi) tamamen siler."""
        collection_name = self._config.CHROMA_COLLECTION_NAME
        self._logger.warning(f"'{collection_name}' ChromaDB koleksiyonu siliniyor...")
        try:
            # Koleksiyonu silmek için istemciyi kullan
            self._client.delete_collection(name=collection_name)
            self._logger.info("Koleksiyon başarıyla silindi.")
        except Exception as e:
             # Eğer koleksiyon zaten yoksa hata verebilir, bunu görmezden gelebiliriz.
             self._logger.warning(f"ChromaDB koleksiyonu silinirken hata oluştu (belki zaten yoktu): {e}")

    def close(self):
        """ChromaDB için özel bir kapatma işlemi gerekmez."""
        self._logger.info("ChromaDB oturumu sonlandırıldı.")