# config.py (v1.1 - Gelişmiş Sorgu Testlerine Göre İyileştirildi)

from pathlib import Path
import torch

class AppConfig:
    """
    Uygulamanın tüm yapılandırma ayarlarını barındıran merkezi sınıf.
    Model adları, dosya yolları, RAG parametreleri ve prompt şablonları burada tanımlanır.
    """
    # --------------------------------------------------------------------------
    # TEMEL DOSYA YOLLARI
    # --------------------------------------------------------------------------
    BASE_DIR: Path = Path(__file__).resolve().parent
    SERVER_DATA_DIR: Path = BASE_DIR / "server_data"
    SOURCE_PDF_DIR: Path = BASE_DIR / "kitaplar"
    PROCESSED_TXT_DIR: Path = SERVER_DATA_DIR / "islenmis_metinler"
    VECTOR_DB_DIR: Path = SERVER_DATA_DIR / "vector_db"
    STATE_FILE: Path = SERVER_DATA_DIR / "system_state.json"
    MODELS_DIR: Path = BASE_DIR / "models"

    # --------------------------------------------------------------------------
    # SUNUCU AYARLARI
    # --------------------------------------------------------------------------
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000

    # --------------------------------------------------------------------------
    # MODEL VE CİHAZ AYARLARI (İstemci tarafı için)
    # --------------------------------------------------------------------------
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    LLAMA_REPO: str = "QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF"
    LLAMA_FILE: str = "Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf"
    LLM_N_CTX: int = 8192
    LLM_N_GPU_LAYERS: int = -1
    LLM_TEMPERATURE: float = 0.05
    LLM_MAX_TOKENS: int = 2048

    # --------------------------------------------------------------------------
    # RAG PARAMETRELERİ
    # --------------------------------------------------------------------------
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    RETRIEVER_K: int = 25
    RERANKER_N: int = 7 # Karşılaştırmalı sorular için daha fazla bağlam sağlamak adına biraz artırıldı.

    # --------------------------------------------------------------------------
    # *** GELİŞTİRİLMİŞ PROMPT ŞABLONLARI (v2) ***
    # Sentez ve karşılaştırma yeteneklerini artırmak için eklenen yeni kurallar.
    # --------------------------------------------------------------------------
    GENERATION_PROMPT: str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
<TALIMATLAR>
Sen, verilen kaynak metinler konusunda uzman bir akademik asistansın. Görevin, kullanıcıdan gelen SORU'yu aşağıdaki kurallara harfiyen uyarak yanıtlamaktır.

**KESİN KURALLAR:**
1.  **YALNIZCA KAYNAKLARI KULLAN:** Cevabını SADECE ve YALNIZCA aşağıda <KAYNAKLAR> etiketi içinde verilen metinlere dayandır. Kendi ön bilgini veya internetten öğrendiklerini KESİNLİKLE kullanma.
2.  **KAYNAK GÖSTERME ZORUNLULUĞU:** Cevabına eklediğin HER BİR cümle veya iddia için, o bilginin alındığı metin parçasının başındaki referansı (örn: `[Kaynak: dosya.pdf, Sayfa: 123]`) cümlenin sonuna ekle. Bu kural istisnasızdır.
3.  **BİLGİ YOKSA, "YOK" DE:** Eğer <KAYNAKLAR> bölümündeki metinler SORU'yu yanıtlamak için yeterli veya ilgili bilgi içermiyorsa, başka hiçbir açıklama yapmadan SADECE ve TAM OLARAK şu cevabı ver: "Sağlanan belgelerde bu bilgi mevcut değil."
4.  **SENTEZ ve KARŞILAŞTIRMA:** Eğer soru karşılaştırma veya özetleme gerektiriyorsa, farklı kaynaklardaki bilgileri birleştirerek tutarlı bir cevap oluştur, ancak her zaman 2. kurala uyarak kaynak göstermeye devam et.
5.  **YORUMSUZ OL:** Cevabına kendi yorumunu, çıkarımını veya görüşünü asla ekleme. Sadece kaynaklarda ne yazıyorsa onu aktar.
</TALIMATLAR><|eot_id|><|start_header_id|>user<|end_header_id|>

<KAYNAKLAR>
{context}
</KAYNAKLAR>

<SORU>
{question}
</SORU><|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    VERIFICATION_PROMPT: str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
<GÖREV>
Aşağıdaki 'İDDİA' metninin, verilen 'KAYNAK METİN' tarafından %100 oranında, hiçbir ek yorum veya bilgi katmadan desteklenip desteklenmediğini analiz et.
Yanıtın SADECE 'EVET' veya 'HAYIR' olmalıdır. Başka hiçbir açıklama yapma.
</GÖREV><|eot_id|><|start_header_id|>user<|end_header_id|>

<KAYNAK_METİN>
{source_chunk}
</KAYNAK_METİN>

<İDDİA>
{generated_answer}
</İDDİA><|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""