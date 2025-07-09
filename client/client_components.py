# client_components.py

import logging
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from sentence_transformers.cross_encoder import CrossEncoder
from shared_utils import FinalResponse
# Projemizin kendi modüllerini import ediyoruz
from config import AppConfig
from shared_utils import ConsoleFormatter

class ModelRegistry:
    """
    İstemci tarafında çalışacak olan ve GPU gerektiren tüm modelleri
    yükler ve yönetir.
    """
    def __init__(self, config: AppConfig):
        self._config = config
        self._logger = logging.getLogger(self.__class__.__name__)
        self._formatter = ConsoleFormatter()
        self._models = {}

    def load_all_models(self):
        """Tüm gerekli modelleri sırayla yükler."""
        self._formatter.print_banner("İSTEMCİ MODELLERİ YÜKLENİYOR", "cyan")
        
        print(self._formatter.color("-> Re-ranker modeli yükleniyor...", "cyan"))
        self.get_reranker_model()
        print(self._formatter.color("   Re-ranker başarıyla yüklendi.", "green"))

        print(self._formatter.color("-> Büyük Dil Modeli (LLM) yükleniyor... (Bu işlem biraz sürebilir)", "cyan"))
        self.get_llm()
        print(self._formatter.color("   LLM başarıyla yüklendi.", "green"))
        
        self._formatter.print_banner("Tüm Modeller Hazır", "green")


    def get_reranker_model(self) -> CrossEncoder:
        """CrossEncoder re-ranker modelini yükler veya önbellekten alır."""
        if "reranker" not in self._models:
            try:
                self._models["reranker"] = CrossEncoder(
                    self._config.RERANKER_MODEL, 
                    device=self._config.DEVICE,
                    max_length=512
                )
            except Exception as e:
                self._logger.critical(f"Re-ranker modeli yüklenemedi: {e}", exc_info=True)
                raise
        return self._models["reranker"]

    def get_llm(self) -> Llama:
        """Llama.cpp (GGUF) modelini yükler veya önbellekten alır."""
        if "llm" not in self._models:
            try:
                # Modeli HuggingFace Hub'dan indir (eğer yoksa)
                self._logger.info(f"LLM indiriliyor/doğrulanıyor: {self._config.LLAMA_REPO}/{self._config.LLAMA_FILE}")
                model_path = hf_hub_download(
                    repo_id=self._config.LLAMA_REPO,
                    filename=self._config.LLAMA_FILE,
                    cache_dir=str(self._config.MODELS_DIR)
                )
                self._logger.info(f"LLM '{model_path}' yolundan yükleniyor.")
                
                self._models["llm"] = Llama(
                    model_path=str(model_path),
                    n_gpu_layers=self._config.LLM_N_GPU_LAYERS,
                    n_ctx=self._config.LLM_N_CTX,
                    n_batch=512,
                    f16_kv=True,
                    verbose=False,
                )
            except Exception as e:
                self._logger.critical(f"LLM yüklenemedi: {e}", exc_info=True)
                raise
        return self._models["llm"]
# client_components.py (Mevcut kodun altına ekleyin)

from typing import List

# Gerekli importları dosyanın en üstüne ekleyelim (eğer zaten yoksa)
from shared_utils import DocumentInfo, QueryResult
# ... (ModelRegistry sınıfı burada)

class RAGCore:
    """
    İstemci tarafındaki RAG sürecini yönetir.
    (Re-ranking -> Context Formatting -> Generation)
    """
    def __init__(self, config: AppConfig, models: ModelRegistry):
        self._config = config
        self._logger = logging.getLogger(self.__class__.__name__)
        self._formatter = ConsoleFormatter()
        self._models = models

    def _rerank_documents(self, query: str, docs: List[DocumentInfo]) -> List[DocumentInfo]:
        """
        Sunucudan gelen belgeleri Cross-Encoder ile yeniden sıralar.
        """
        if not docs:
            return []
        
        self._logger.info(f"Yeniden sıralama için {len(docs)} belge alındı. Cross-encoder çalıştırılıyor...")
        
        # Cross-encoder modelinin beklediği format: [sorgu, metin]
        pairs = [[query, doc.page_content] for doc in docs]
        
        reranker = self._models.get_reranker_model()
        scores = reranker.predict(pairs, show_progress_bar=False)
        
        # Belgeleri skorlarıyla birlikte eşleştirip en yüksek skorludan aza doğru sırala
        doc_scores = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        
        # Yapılandırmada belirtilen en iyi N belgeyi seç
        reranked_docs = [doc for doc, score in doc_scores[:self._config.RERANKER_N]]
        
        self._logger.info(f"Yeniden sıralama tamamlandı. En iyi {len(reranked_docs)} belge seçildi.")
        
        # Debug için skorları gösterelim
        for doc, score in doc_scores[:self._config.RERANKER_N]:
             self._logger.debug(f"  - Skor: {score:.4f} | Kaynak: {doc.metadata.get('book_title')}, Sayfa: {doc.metadata.get('page')}")
        
        return reranked_docs

    def _format_context(self, docs: List[DocumentInfo]) -> str:
        """
        LLM'e gönderilecek bağlam (context) metnini formatlar.
        """
        if not docs:
            return "Kullanıcının sorusunu yanıtlamak için hiçbir kaynak bulunamadı."

        context_parts = []
        for doc in docs:
            # Metadata'dan zengin bir kaynak referansı oluştur
            title = doc.metadata.get('book_title', 'Bilinmeyen Kitap')
            author = doc.metadata.get('author', 'Bilinmeyen Yazar')
            page = doc.metadata.get('page', 'Bilinmeyen Sayfa')
            source_ref = f"[Kaynak: {title} ({author}), Sayfa: {page}]"
            
            # İçeriği ve referansı birleştir
            context_parts.append(f"{source_ref}\n{doc.page_content}")
            
        return "\n\n---\n\n".join(context_parts)

    def generate_answer(self, query: str, retrieved_docs: List[DocumentInfo]) -> (str, List[DocumentInfo]):
        """
        RAG sürecinin tamamını çalıştırır ve nihai cevabı üretir.
        """
        # Adım 1: Belgeleri yeniden sırala
        self._formatter.print_banner("AŞAMA 2: BELGELERİ YENİDEN SIRALAMA", "cyan")
        reranked_docs = self._rerank_documents(query, retrieved_docs)

        # Adım 2: LLM için bağlamı hazırla
        self._formatter.print_banner("AŞAMA 3: BAĞLAM OLUŞTURMA VE CEVAP ÜRETME", "cyan")
        context = self._format_context(reranked_docs)

        # Adım 3: Gelişmiş prompt'u doldur
        full_prompt = self._config.GENERATION_PROMPT.format(
            context=context, 
            question=query
        )
        
        self._logger.debug(f"--- LLM'e GÖNDERİLEN TAM PROMPT ---\n{full_prompt}\n---------------------------------")
        print(self.formatter.color("LLM cevap üretiyor...", "blue"))

        # Adım 4: LLM'i çalıştır
        llm = self._models.get_llm()
        response = llm(
            prompt=full_prompt,
            max_tokens=self._config.LLM_MAX_TOKENS,
            temperature=self._config.LLM_TEMPERATURE,
            stop=["<|eot_id|>", "<|end_of_text|>"] # Modelin gereksiz yere devam etmesini engelle
        )

        raw_answer = response['choices'][0]['text'].strip()
        self._logger.info(f"LLM'den ham cevap alındı: {raw_answer}")
        
        # Cevap ve bu cevabın dayandığı kaynakları (reranked_docs) döndür
        return raw_answer, reranked_docs    
class AnswerVerifier:
    """
    LLM tarafından üretilen cevabın, kullanılan kaynaklar tarafından
    desteklenip desteklenmediğini doğrular.
    """
    def __init__(self, config: AppConfig, models: ModelRegistry):
        self._config = config
        self._logger = logging.getLogger(self.__class__.__name__)
        self._formatter = ConsoleFormatter()
        self._models = models

    def verify(self, raw_answer: str, source_docs: List[DocumentInfo]) -> FinalResponse:
        """
        Cevabın her bir kaynak belge tarafından desteklenip desteklenmediğini kontrol eder.
        """
        self._formatter.print_banner("AŞAMA 4: CEVAP DOĞRULAMA", "yellow")
        
        if "Sağlanan belgelerde bu bilgi mevcut değil." in raw_answer:
            self._logger.warning("LLM, bilginin kaynaklarda olmadığını belirtti. Doğrulama atlanıyor.")
            return FinalResponse(
                final_answer=raw_answer,
                is_fully_verified=True, # Bu durumda cevap "doğru" kabul edilir.
                raw_llm_answer=raw_answer
            )

        llm = self._models.get_llm()
        verified_sources = []
        unverified_sources = []

        # Cevabı, kaynak gösterme etiketlerinden temizleyerek "iddia" haline getir.
        # Bu, doğrulama prompt'unun daha temiz çalışmasını sağlar.
        claim = re.sub(r'\[Kaynak:.*?\]', '', raw_answer).strip()

        if not claim:
             return FinalResponse(
                final_answer="Model boş bir cevap üretti.",
                is_fully_verified=False,
                raw_llm_answer=raw_answer
            )


        print(self.formatter.color(f"Doğrulanacak İddia: \"{claim[:100]}...\"", "yellow"))

        for doc in source_docs:
            source_chunk = doc.page_content
            
            # Doğrulama için özel prompt'u doldur
            verification_prompt = self._config.VERIFICATION_PROMPT.format(
                source_chunk=source_chunk,
                generated_answer=claim # Temizlenmiş iddiayı kullan
            )
            
            self._logger.debug(f"Doğrulama Promptu (Kaynak: {doc.metadata['book_title']} - S.{doc.metadata['page']}):\n{verification_prompt}")

            # LLM'e EVET/HAYIR sorusunu sor
            response = llm(
                prompt=verification_prompt,
                max_tokens=10, # Sadece EVET/HAYIR için
                temperature=0.0, # Maksimum kesinlik
                stop=["<|eot_id|>"]
            )
            
            verification_result = response['choices'][0]['text'].strip().upper()
            self._logger.info(f"Doğrulama sonucu: {verification_result} (Kaynak: {doc.metadata['book_title']} S.{doc.metadata['page']})")

            if "EVET" in verification_result:
                verified_sources.append(doc.metadata)
            else:
                unverified_sources.append(doc.metadata)
        
        # Sonucu oluştur
        is_fully_verified = len(verified_sources) > 0 and len(unverified_sources) == 0
        
        # Son cevaba, sadece doğrulanmış kaynakların etiketlerini ekleyerek daha güvenilir hale getirebiliriz.
        # Şimdilik ham cevabı koruyalım ama bu bir sonraki iyileştirme olabilir.
        final_answer = raw_answer

        return FinalResponse(
            final_answer=final_answer,
            verified_sources=verified_sources,
            unverified_sources=unverified_sources,
            is_fully_verified=is_fully_verified,
            raw_llm_answer=raw_answer
        )    