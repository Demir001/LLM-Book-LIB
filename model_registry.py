# model_registry.py

import logging
from typing import Dict, Any

from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

from config import APP_CONFIG, ModelError
from utils import ConsoleManager

class ModelRegistry:
    """
    Tüm yapay zeka modellerini (Embedding, Cross-Encoder, LLM) yönetir.
    Modelleri sadece bir kez yükleyerek kaynak kullanımını ve başlatma
    süresini optimize eder.
    """
    def __init__(self):
        self._config = APP_CONFIG
        self._logger = logging.getLogger(self.__class__.__name__)
        self._models: Dict[str, Any] = {}
        self._console = ConsoleManager()

    def load_all_models_proactively(self):
        """
        Uygulama için gerekli tüm modelleri önceden yükler. Bu, ilk sorgu
        sırasında yaşanabilecek "ısınma" gecikmesini önler.
        """
        self._console.print_header("Yapay Zeka Modelleri Yükleniyor")
        self.get_embedding_model()
        self.get_cross_encoder_model()
        self.get_llm()
        self._console.print_info("Tüm modeller başarıyla yüklendi ve kullanıma hazır.", "GREEN")

    def get_embedding_model(self) -> HuggingFaceEmbeddings:
        """Embedding modelini yükler veya önbellekten döndürür."""
        return self._load_model(
            'embedding',
            lambda: HuggingFaceEmbeddings(
                model_name=self._config.EMBEDDING_MODEL,
                model_kwargs={'device': self._config.DEVICE},
                encode_kwargs={'normalize_embeddings': True}
            )
        )

    def get_cross_encoder_model(self) -> CrossEncoder:
        """Cross-Encoder (Re-Ranker) modelini yükler veya önbellekten döndürür."""
        return self._load_model(
            'cross_encoder',
            lambda: CrossEncoder(
                self._config.CROSS_ENCODER_MODEL,
                max_length=512,
                device=self._config.DEVICE
            )
        )

    def get_llm(self) -> Llama:
        """Büyük Dil Modelini (LLM) yükler veya önbellekten döndürür."""
        return self._load_model('llm', self._load_llama_model)

    def _load_llama_model(self) -> Llama:
        """LLM'i Hugging Face Hub'dan indirir ve Llama.cpp ile yükler."""
        self._logger.info(f"LLM dosyası indiriliyor: {self._config.LLAMA_FILE}")
        model_path = hf_hub_download(
            repo_id=self._config.LLAMA_REPO,
            filename=self._config.LLAMA_FILE,
            cache_dir=str(self._config.MODELS_DIR),
            resume_download=True,
        )
        self._logger.info("LLM modeli GPU'ya yükleniyor. Bu işlem biraz sürebilir...")
        
        llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=self._config.LLM_N_GPU_LAYERS,
            n_ctx=self._config.LLM_N_CTX,
            n_batch=512,
            f16_kv=True,
            verbose=True
        )
        
        # Yükleme sonrası GPU kullanımını programatik olarak doğrula
        # DÜZELTME: Artık llm.n_gpu_layers() metodu kullanılıyor.
        if llm.n_gpu_layers() > 0:
            self._console.print_info("LLM, GPU üzerinde başarıyla çalışıyor.", "GREEN")
        else:
            self._console.print_info(
                "DİKKAT: LLM, CPU üzerinde çalışıyor. Bu, performansı ciddi şekilde düşürecektir.", "RED"
            )
        return llm

    def _load_model(self, model_key: str, loader_func) -> Any:
        """
        Modelleri yüklemek için genel bir yardımcı fonksiyon.
        Model zaten yüklenmişse, onu döndürür. Aksi takdirde, verilen
        fonksiyonu kullanarak yükler ve önbelleğe alır.
        """
        if model_key not in self._models:
            self._logger.info(f"Yükleniyor: {model_key.replace('_', ' ').title()} Modeli ({getattr(self._config, f'{model_key.upper()}_MODEL', self._config.LLAMA_FILE)})")
            try:
                self._models[model_key] = loader_func()
            except Exception as e:
                raise ModelError(f"'{model_key}' modeli yüklenemedi: {e}", exc_info=True)
        return self._models[model_key]