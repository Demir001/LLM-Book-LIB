# config.py

import os
from pathlib import Path
import torch
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Dict, Any
# Proje kök dizininden .env dosyasını yükle
load_dotenv()

@dataclass(frozen=True)
class AppConfig:
    """
    Tüm uygulama yapılandırmalarını ve ayarlarını merkezi olarak yönetir.
    'frozen=True' bu nesnenin oluşturulduktan sonra değiştirilemez olmasını sağlar.
    """
    
    # --- Proje Yapısı ---
    APP_NAME: str = "Bilgi İmparatorluğu v13 (Modüler Çekirdek)"
    BASE_DIR: Path = Path(__file__).resolve().parent
    SOURCE_DIR: Path = BASE_DIR / "kutuphane"
    PROCESSED_DIR: Path = BASE_DIR / "islenmis_veriler"
    MODELS_DIR: Path = BASE_DIR / "modeller"
    STATE_FILE: Path = PROCESSED_DIR / "sistem_durumu.json"
    METADATA_DIR: Path = SOURCE_DIR / "metaveriler"
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Model Ayarları ---
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CROSS_ENCODER_MODEL: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    LLAMA_REPO: str = "QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF"
    LLAMA_FILE: str = "Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf"

    # --- Weaviate Ayarları (Self-Hosted Docker için .env'den) ---
    WEAVIATE_HOST: str = os.getenv("WEAVIATE_HOST", "localhost")
    WEAVIATE_PORT: int = int(os.getenv("WEAVIATE_PORT", "8080"))
    WEAVIATE_GRPC_PORT: int = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
    WEAVIATE_URL: str = f"http://{WEAVIATE_HOST}:{WEAVIATE_PORT}"
    WEAVIATE_INDEX_NAME: str = "BilgiImparatorluguV13"

    CHROMA_PERSIST_DIR: Path = BASE_DIR / "chroma_db"
    CHROMA_COLLECTION_NAME: str = "bilgi_imparatorlugu_v13"

    # --- RAG ve LLM Parametreleri ---
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    RETRIEVER_K: int = 20
    RERANKER_TOP_N: int = 5
    LLM_N_CTX: int = 8192
    LLM_N_GPU_LAYERS: int = -1
    LLM_MAX_TOKENS: int = 2048
    LLM_TEMPERATURE: float = 0.15
    LLM_REPEAT_PENALTY: float = 1.1

    # --- Prompt Şablonu ---
    GENERATION_PROMPT: str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
## GÖREV TANIMI
Sen, birincil kaynağa dayalı bir araştırma uzmanısın. Tek görevin, sağlanan KANITLAR bölümündeki metinleri kullanarak, kullanıcının SORU'suna doğrudan ve kanıta dayalı bir cevap oluşturmaktır.

## KESİN KURALLAR
1.  **KANIT DIŞINA ÇIKMA:** Cevabındaki her kelime, doğrudan KANITLAR bölümünden türetilmelidir. Kendi bilgini, yorumunu veya çıkarımını ASLA ekleme.
2.  **KAYNAK GÖSTERME ZORUNLULUĞU:** Cevabındaki her anlamlı cümle veya bilgi bloğunun sonuna, o bilginin alındığı kaynağı `[Kaynak: Kitap Başlığı - Yazar, Sayfa: X]` formatında ekle.
3.  **YETERSİZ KANIT DURUMU:** Eğer KANITLAR, soruyu cevaplamak için yeterli değilse veya konuyla ilgili değilse, başka hiçbir açıklama yapmadan SADECE şu cümleyi yaz: "Sağlanan kanıtlarda bu soruya yanıt verecek spesifik bir bilgi bulunmamaktadır."
<|eot_id|><|start_header_id|>user<|end_header_id|>
KANITLAR:
{context}

SORU: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

# --- Özel Hata Sınıfları ---
# Uygulama genelinde daha anlaşılır hata yönetimi için.
class SystemError(Exception):
    def __init__(self, message, *args):
        super().__init__(message, *args)
        self.message = message

class ConfigError(SystemError): pass
class ProcessingError(SystemError): pass
class DatabaseError(SystemError): pass
class ModelError(SystemError): pass
# --- Temel Veri Sınıfları (Dataclasses) ---
# Uygulamanın farklı modülleri arasında veri taşımak için standart yapılar.
@dataclass(frozen=True)
class DocumentMetadata:
    source_file: str
    title: str
    author: str

@dataclass
class EnrichedDocument:
    page_content: str
    metadata: Dict[str, Any]

@dataclass
class RAGResult:
    final_answer: str
    source_documents: list

# Yapılandırma nesnesini projenin diğer kısımlarından kolayca
# import edilebilmesi için bir örnek oluşturuyoruz.
APP_CONFIG = AppConfig()