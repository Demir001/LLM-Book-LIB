# rag_core.py

import logging
from typing import List, Generator, Union

from langchain.docstore.document import Document

from config import APP_CONFIG
from model_registry import ModelRegistry
from database import VectorDatabase
from reranker import ReRanker
from utils import ConsoleManager

class RAGCore:
    """
    Retrieval-Augmented Generation mantığının çekirdeği. Bu sınıf,
    sorgulama, belge alma, yeniden sıralama ve cevap üretme adımlarını yönetir
    ve yanıtı bir akış (stream) olarak sunar.
    """
    def __init__(self, models: ModelRegistry, db: VectorDatabase, reranker: ReRanker):
        self._config = APP_CONFIG
        self._logger = logging.getLogger(self.__class__.__name__)
        self._console = ConsoleManager()
        
        # Bağımlılıkları enjekte et (Dependency Injection)
        self._models = models
        self._database = db
        self._reranker = reranker
        
        # Sık kullanılacak modelleri başlangıçta al
        self._llm = self._models.get_llm()

    def _format_context(self, documents: List[Document]) -> str:
        """Verilen doküman listesini, LLM'in anlayacağı bir bağlam metnine dönüştürür."""
        context_parts = []
        for doc in documents:
            metadata = doc.metadata
            source_str = f"{metadata.get('title', 'Bilinmiyor')} - {metadata.get('author', 'Bilinmiyor')}"
            page = metadata.get('page', 0)
            context_parts.append(f"[Kaynak: {source_str}, Sayfa: {page}]\n{doc.page_content}")
        
        return "\n\n---\n\n".join(context_parts)

    def stream_query_response(self, user_query: str) -> Generator[Union[List[Document], str], None, None]:
        """
        Kullanıcı sorgusunu baştan sona işler ve cevabı bir 'generator' olarak,
        parça parça (token token) akıtır.
        
        Akışın Protokolü:
        1. İlk yield: Kaynak dokümanların bir listesi `List[Document]`.
           Eğer hiç doküman bulunamazsa, bu liste boş olur.
        2. Sonraki yield'ler: Cevabın metin token'ları (str).
        """
        # Adım 1: Geniş Arama (Retrieve)
        self._console.print_info("1/3: Genişletilmiş arama ile aday dokümanlar bulunuyor...", "BLUE")
        search_kwargs = {'k': self._config.RETRIEVER_K}
        retriever = self._database.get_retriever(search_kwargs)
        
        # Sorgu genişletme LLM çağrısı gerektirdiği için performansı düşürebilir.
        # Hibrit arama ve Re-ranker zaten çok güçlü olduğu için şimdilik sadece
        # orijinal sorguyu kullanmak daha verimlidir.
        retrieved_docs = retriever.invoke(user_query)
        
        # Gelen belgeleri tekilleştir
        unique_docs_map = {
            (doc.page_content, doc.metadata.get('source_file'), doc.metadata.get('page')): doc 
            for doc in retrieved_docs
        }
        unique_docs = list(unique_docs_map.values())
        self._logger.info(f"Veritabanından {len(unique_docs)} adet tekil doküman adayı bulundu.")
        
        if not unique_docs:
            yield []  # Kaynak doküman olarak boş liste gönder
            return

        # Adım 2: Akıllı Sıralama (Re-Rank)
        self._console.print_info(f"2/3: {len(unique_docs)} aday doküman hassas bir şekilde yeniden sıralanıyor...", "BLUE")
        reranked_docs = self._reranker.rerank(user_query, unique_docs)
        
        if not reranked_docs:
            yield []  # Sıralama sonrası alakalı doküman kalmazsa boş liste gönder
            return
        
        # --- Akış Başlangıcı ---
        # İlk olarak, kaynak dokümanları akışın ilk verisi olarak gönder.
        # Arayüz bu bilgiyi alıp daha sonra doğrulama için kullanabilir.
        yield reranked_docs

        # Adım 3: Cevap Üretme (Generate)
        self._console.print_info(f"3/3: En alakalı {len(reranked_docs)} doküman ile cevap üretiliyor...", "BLUE")
        final_context = self._format_context(reranked_docs)
        
        full_prompt = self._config.GENERATION_PROMPT.format(context=final_context, question=user_query)
        
        # stream=True ile LLM'i çağırarak akışı başlat
        stream = self._llm(
            prompt=full_prompt,
            max_tokens=self._config.LLM_MAX_TOKENS,
            temperature=self._config.LLM_TEMPERATURE,
            repeat_penalty=self._config.LLM_REPEAT_PENALTY,
            stop=["<|eot_id|>", "<|end_of_text|>"],
            stream=True
        )
        
        # Gelen her bir token'ı anında dışarıya (TerminalApp'e) yield et
        for chunk in stream:
            token = chunk["choices"][0].get("text", "")
            if token:
                yield token