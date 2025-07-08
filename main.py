# main.py
import datetime
import os
import shutil
import logging
from pathlib import Path
from dataclasses import asdict
import json
from config import APP_CONFIG, RAGResult, ConfigError, ModelError, DatabaseError
from utils import ConsoleManager
from metadata_manager import MetadataManager
from data_processor import DocumentProcessor
from model_registry import ModelRegistry
from database import VectorDatabase
from reranker import ReRanker
from rag_core import RAGCore
from answer_verifier import AnswerVerifier

class StateManager:
    """İşlenmiş PDF dosyalarının durumunu (işlenip işlenmediğini) takip eder."""
    def __init__(self):
        self._file = APP_CONFIG.STATE_FILE
        self._state = self._load()
        self._logger = logging.getLogger(self.__class__.__name__)

    def _load(self) -> dict:
        if self._file.exists():
            with open(self._file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save(self):
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._file, 'w', encoding='utf-8') as f:
            json.dump(self._state, f, indent=4)

    def is_pdf_processed(self, pdf_path: Path) -> bool:
        return str(pdf_path.resolve()) in self._state

    def mark_pdf_as_processed(self, pdf_path: Path):
        self._state[str(pdf_path.resolve())] = datetime.now().isoformat()
        self._save()

    def reset(self):
        self._state = {}
        if self._file.exists():
            os.remove(self._file)
            self._logger.info("Durum dosyası sıfırlandı.")


class MainApplication:
    """Uygulamanın ana döngüsünü, komutları ve kullanıcı arayüzünü yönetir."""
    def __init__(self):
        self._config = APP_CONFIG
        self._console = ConsoleManager()
        self._setup_logging()
        
        # Bileşenleri başlat
        self.state_manager = StateManager()
        self.metadata_manager = MetadataManager()
        self.processor = DocumentProcessor()
        self.models = ModelRegistry()
        self.database = VectorDatabase(self.models.get_embedding_model())
        self.reranker = ReRanker(self.models)
        self.rag_core = RAGCore(self.models, self.database, self.reranker)
        self.verifier = AnswerVerifier()

    def _setup_logging(self):
        """Uygulama için loglama ayarlarını yapılandırır."""
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO").upper(),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                # İleride bir log dosyasına da yazdırılabilir
                # logging.FileHandler("app.log") 
            ]
        )

    def _sync_library(self):
        """Kütüphaneyi tarar, eksik meta verileri ister ve yeni dosyaları işler."""
        self._console.print_header("Kütüphane Senkronizasyonu")
        
        # 1. Adım: Eksik meta verileri interaktif olarak tamamla
        self.metadata_manager.ensure_metadata_for_all_pdfs()

        # 2. Adım: İşlenmemiş PDF'leri bul
        pdfs_to_process = [
            p for p in self._config.SOURCE_DIR.glob("*.pdf") 
            if not self.state_manager.is_pdf_processed(p)
        ]

        if not pdfs_to_process:
            self._console.print_info("Kütüphane güncel. İşlenecek yeni PDF yok.", "GREEN")
            return
            
        self._console.print_info(f"{len(pdfs_to_process)} yeni/işlenmemiş PDF işlenecek.", "YELLOW")
        self.models.load_all_models_proactively() # Ağır modelleri şimdi yükle

        # 3. Adım: Yeni PDF'leri işle ve veritabanına ekle
        for pdf_path in pdfs_to_process:
            self._console.print_info(f"--- {pdf_path.name} işleniyor ---", "CYAN")
            try:
                metadata = self.metadata_manager.get_metadata_for_pdf(pdf_path)
                if not metadata:
                    logging.warning(f"{pdf_path.name} için meta veri okunamadı, atlanıyor.")
                    continue

                enriched_docs = self.processor.process_pdf(pdf_path, metadata)
                if not enriched_docs:
                    logging.warning(f"{pdf_path.name} içeriği işlenemedi, atlanıyor.")
                    continue

                self.database.add_documents(enriched_docs)
                self.state_manager.mark_pdf_as_processed(pdf_path)
                self._console.print_info(f"--- {pdf_path.name} başarıyla işlendi ---", "GREEN")

            except Exception as e:
                logging.critical(f"'{pdf_path.name}' işlenirken kritik hata: {e}", exc_info=True)
        
        self._console.print_info("Senkronizasyon tamamlandı.", "GREEN")

    def _handle_query(self, user_query: str):
        """Tek bir kullanıcı sorgusunu baştan sona işler ve sonucu akıtarak yazdırır."""
        self._console.print_header("Sorgu İşleniyor")
        final_answer, source_docs = "", []
        try:
            response_stream = self.rag_core.stream_query_response(user_query)
            source_docs = next(response_stream)
            
            if not source_docs:
                self._console.print_info("Bu soruya yanıt verecek ilgili doküman bulunamadı.", "YELLOW")
                return
            
            # Cevap akışını başlat
            self._console.print_info("\n💡 YANIT:", "HEADER")
            for token in response_stream:
                print(self._console.color(token, "BOLD"), end="", flush=True)
                final_answer += token
            print() # Cevap bittikten sonra yeni satıra geç

            # Akış bittikten sonra doğrula ve kaynakları sun
            self.verifier.verify_and_present(final_answer, source_docs, self._console)

        except StopIteration:
            self._console.print_info("Bu soruya yanıt verecek spesifik bir bilgi bulunmamaktadır.", "YELLOW")
        except Exception as e:
            logging.error(f"Sorgu sırasında hata: {e}", exc_info=True)
            self._console.print_info("\nCevap üretilirken bir hata oluştu.", "RED")

    def _cmd_rebuild(self):
        """Tüm veritabanını, durumu ve işlenmiş dosyaları siler."""
        confirm = self._console.get_user_input("DİKKAT! Tüm veritabanı ve meta veriler dahil her şey silinecek. Emin misiniz? (evet/hayır): ", "RED")
        if confirm.lower() == 'evet':
            self._console.print_info("Sistem sıfırlanıyor...", "YELLOW")
            self.database.delete_collection()
            self.state_manager.reset()
            if self._config.METADATA_DIR.exists(): shutil.rmtree(self._config.METADATA_DIR)
            self._console.print_info("Sistem sıfırlandı. Uygulamayı yeniden başlatın veya ':reload' komutunu kullanın.", "GREEN")
        else:
            self._console.print_info("İşlem iptal edildi.", "GREEN")

    def run(self):
        """Uygulamanın ana giriş noktası ve komut döngüsü."""
        self._console.print_header(self._config.APP_NAME)
        # Gerekli klasörlerin var olduğundan emin ol
        self._config.SOURCE_DIR.mkdir(parents=True, exist_ok=True)
        self._config.METADATA_DIR.mkdir(parents=True, exist_ok=True)
        
        try:
            self._sync_library()
            
            while True:
                user_input = self._console.get_user_input("\n❓> ")
                if not user_input: continue
                
                cmd = user_input.lower().strip()
                if cmd in [':q', ':quit', ':exit']: break
                elif cmd == ':rebuild': self._cmd_rebuild()
                elif cmd == ':reload': self._sync_library()
                else: self._handle_query(user_input)

        except (ConfigError, ModelError, DatabaseError, SystemError) as e:
            logging.critical(f"Kurtarılamayan sistem hatası: {e}", exc_info=True)
        except KeyboardInterrupt:
            logging.warning("Kullanıcı tarafından uygulama sonlandırıldı.")
        finally:
            if self.database: self.database.close()
            self._console.print_info("\nOturum sonlandırıldı.", "YELLOW")
if __name__ == "__main__":
    """
    Bu blok, yalnızca `python main.py` komutuyla dosya doğrudan
    çalıştırıldığında devreye girer.
    """
    try:
        # 1. Ana uygulama nesnesini başlat. Bu, __init__ içinde tüm
        #    diğer bileşenleri (modeller, veritabanı vb.) kuracaktır.
        app = MainApplication()
        
        # 2. Uygulamanın ana döngüsünü çalıştır.
        app.run()

    except Exception as e:
        # Uygulama başlatma sırasında oluşabilecek en temel hataları bile
        # yakalamak için son bir güvenlik ağı.
        # Örn: Kütüphane import hatası, yapılandırma hatası vb.
        logging.basicConfig(level="INFO", format='%(asctime)s - %(levelname)s - %(message)s')
        logging.critical(f"Uygulama başlatılırken kurtarılamayan kritik bir hata oluştu: {e}", exc_info=True)