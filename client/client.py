# client.py (Nihai Sürüm)

import json
import logging
import sys
from urllib import request, error

# Projemizin kendi modüllerini import ediyoruz
from config import AppConfig
from shared_utils import ConsoleFormatter, QueryResult, DocumentInfo, FinalResponse
from client_components import ModelRegistry, RAGCore, AnswerVerifier

# Python'un standart kütüphanelerinden
import re
import os

class ServerAPIClient:
    """
    Uzak sunucu ile iletişimi yöneten sınıf.
    (Adım 6'da oluşturduğumuz sınıf, burada tekrar yer alıyor)
    """
    def __init__(self, server_url: str, formatter: ConsoleFormatter):
        if not server_url.endswith('/'):
            server_url += '/'
        self.base_url = server_url
        self.formatter = formatter
        self._logger = logging.getLogger(self.__class__.__name__)

    def _make_request(self, endpoint: str, data: dict = None, method: str = 'GET', timeout=60):
        url = self.base_url + endpoint
        headers = {'Content-Type': 'application/json'}
        req_data = json.dumps(data).encode('utf-8') if data else None
        
        req = request.Request(url, data=req_data, headers=headers, method=method)
        
        try:
            with request.urlopen(req, timeout=timeout) as response:
                if 200 <= response.status < 300:
                    return json.loads(response.read().decode('utf-8'))
                else:
                    error_content = response.read().decode('utf-8')
                    self._logger.error(f"Sunucudan hata kodu {response.status}: {error_content}")
                    print(self.formatter.color(f"Sunucu hatası: {response.status} - {error_content}", "red", "bold"))
                    return None
        except error.HTTPError as e:
            error_content = e.read().decode('utf-8')
            self._logger.error(f"HTTP Hatası: {e.code} - {error_content}", exc_info=True)
            print(self.formatter.color(f"Sunucu ile iletişimde HTTP hatası: {e.code}", "red", "bold"))
            return None
        except error.URLError as e:
            self._logger.error(f"URL Hatası: {e.reason}", exc_info=True)
            print(self.formatter.color(f"Sunucuya bağlanılamadı: {e.reason}. Sunucunun çalıştığından ve URL'nin doğru olduğundan emin olun.", "red", "bold"))
            return None

    def get_status(self):
        print(self.formatter.color("Sunucu durumu kontrol ediliyor...", "blue"))
        return self._make_request('status', method='GET')

    def trigger_sync(self):
        print(self.formatter.color("Sunucuya senkronizasyon komutu gönderiliyor... (Bu işlem yeni dosya varsa uzun sürebilir)", "blue"))
        # Sync işlemi uzun sürebilir, timeout'u artıralım
        return self._make_request('sync', method='POST', timeout=600)

    def retrieve_documents(self, query: str) -> QueryResult | None:
        self._logger.info(f"'{query}' sorgusu için sunucudan belgeler isteniyor...")
        print(self.formatter.color("AŞAMA 1: SUNUCUDAN İLGİLİ BELGELER ALINIYOR", "cyan"))
        payload = {"query": query}
        response_data = self._make_request('retrieve', data=payload, method='POST')
        
        if response_data and 'retrieved_docs' in response_data:
            docs = [DocumentInfo(**doc) for doc in response_data['retrieved_docs']]
            print(self.formatter.color(f"-> Sunucudan {len(docs)} adet potansiyel belge alındı.", "green"))
            return QueryResult(retrieved_docs=docs, query=response_data['query'], message=response_data['message'])
        return None

class TerminalApplication:
    """
    Tüm istemci uygulamasını yöneten ana sınıf.
    """
    def __init__(self, config: AppConfig, server_url: str, log_level: str = 'INFO'):
        self._config = config
        self._is_debug_mode = log_level.upper() == 'DEBUG'

        # Logging ayarları
        logging.basicConfig(level=log_level.upper(), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self._logger = logging.getLogger("CLIENT_APP")

        # Bileşenleri başlat
        self.formatter = ConsoleFormatter()
        self.server_client = ServerAPIClient(server_url, self.formatter)
        
        # GPU Ağırlıklı bileşenler (başarılı sunucu bağlantısından sonra yüklenecek)
        self.models = None
        self.rag_core = None
        self.verifier = None

    def start(self):
        """Uygulamayı başlatır ve ana döngüyü çalıştırır."""
        self.formatter.print_banner("KİŞİSEL BİLGİ ASİSTANI İSTEMCİSİ", "header")
        
        # 1. Sunucu bağlantısını kontrol et
        if not self.server_client.get_status():
            self.formatter.print_banner("SUNUCU BAĞLANTISI BAŞARISIZ! UYGULAMA KAPATILIYOR.", "red")
            sys.exit(1)
        
        # 2. Modelleri yükle
        self.models = ModelRegistry(self._config)
        self.models.load_all_models()
        
        # 3. Diğer bileşenleri hazırla
        self.rag_core = RAGCore(self._config, self.models)
        self.verifier = AnswerVerifier(self._config, self.models)

        self.formatter.print_banner("Sistem Sorgu İçin Hazır", "green")
        print(self.formatter.color("Yardım için ':help' yazın, çıkmak için ':quit'.", "yellow"))
        
        # 4. Ana sorgu döngüsünü çalıştır
        self._run_main_loop()

    def _run_main_loop(self):
        """Kullanıcıdan girdi alan ve işleyen ana döngü."""
        while True:
            try:
                prompt_color = "yellow" if self._is_debug_mode else "cyan"
                user_input = input(self.formatter.color("❓> ", "bold", prompt_color)).strip()
                if not user_input:
                    continue
                
                if user_input.startswith(':'):
                    if self._handle_command(user_input):
                        break # :quit komutu True döndürürse döngüyü kır
                else:
                    self._handle_query(user_input)

            except (KeyboardInterrupt, EOFError):
                print("\nÇıkış yapılıyor...")
                break
    
    def _handle_command(self, cmd: str) -> bool:
        """Özel komutları işler."""
        command = cmd.lower().strip()
        if command == ':quit':
            return True
        elif command == ':help':
            self._cmd_help()
        elif command == ':clear':
            os.system('cls' if os.name == 'nt' else 'clear')
        elif command == ':sync':
            self.server_client.trigger_sync()
        elif command == ':stats':
            status_data = self.server_client.get_status()
            if status_data:
                self._cmd_stats(status_data)
        else:
            print(self.formatter.color(f"Bilinmeyen komut: '{cmd}'. Yardım için ':help' yazın.", "red"))
        return False

    def _handle_query(self, query: str):
        """Bir kullanıcı sorgusunu baştan sona işler."""
        # Adım 1: Sunucudan ilgili belgeleri al
        query_result = self.server_client.retrieve_documents(query)
        if not query_result or not query_result.retrieved_docs:
            print(self.formatter.color("Sunucudan ilgili belge alınamadı veya arama sonucu boş.", "yellow"))
            return

        # Adım 2 & 3: RAG çekirdeği ile cevap üret
        raw_answer, source_docs = self.rag_core.generate_answer(query, query_result.retrieved_docs)

        # Adım 4: Cevabı doğrula
        verified_result = self.verifier.verify(raw_answer, source_docs)
        
        # Adım 5: Sonucu kullanıcıya göster
        self._display_final_result(verified_result)

    def _display_final_result(self, result: FinalResponse):
        """Nihai sonucu formatlı bir şekilde ekrana basar."""
        self.formatter.print_banner("SONUÇ", "header")
        
        # Doğrulama durumuna göre başlık rengini ayarla
        if result.is_fully_verified:
            print(self.formatter.color("💡 YANIT (Tamamen Doğrulandı)", "bold", "green"))
        else:
            print(self.formatter.color("💡 YANIT (Kısmen veya Doğrulanamadı)", "bold", "yellow"))
        
        print(result.final_answer)

        print("\n" + self.formatter.color("✅ DOĞRULANMIŞ KAYNAKLAR:", "bold", "green"))
        if result.verified_sources:
            for src in result.verified_sources:
                print(f"  - Kitap: {src['book_title']} ({src['author']}), Sayfa: {src['page']}")
        else:
            print("  - Bu cevabı destekleyen doğrulanmış kaynak bulunamadı.")

        if result.unverified_sources:
            print("\n" + self.formatter.color("❌ DOĞRULANAMAYAN KAYNAKLAR:", "bold", "red"))
            for src in result.unverified_sources:
                print(f"  - Kitap: {src['book_title']} ({src['author']}), Sayfa: {src['page']}")

        print("-" * 80)

    def _cmd_help(self):
        self.formatter.print_banner("YARDIM MENÜSÜ", "yellow")
        print(f"{self.formatter.color(':help', 'bold', 'cyan')}      - Bu yardım menüsünü gösterir.")
        print(f"{self.formatter.color(':quit', 'bold', 'cyan')}      - Uygulamadan çıkar.")
        print(f"{self.formatter.color(':clear', 'bold', 'cyan')}     - Terminal ekranını temizler.")
        print(f"{self.formatter.color(':sync', 'bold', 'cyan')}      - Sunucudaki /kitaplar klasörünü taramayı ve yeni dosyaları işlemeyi tetikler.")
        print(f"{self.formatter.color(':stats', 'bold', 'cyan')}     - Sunucunun mevcut durumunu (işlenmiş dosya sayısı, vb.) gösterir.")

    def _cmd_stats(self, stats: dict):
        self.formatter.print_banner("SUNUCU İSTATİSTİKLERİ", "yellow")
        print(f"  - Durum: {self.formatter.color(stats.get('status', 'Bilinmiyor').upper(), 'green')}")
        print(f"  - İşlenmiş Dosya Sayısı: {self.formatter.color(str(stats.get('processed_files_count')), 'cyan')}")
        print(f"  - Veritabanındaki Metin Parçası: {self.formatter.color(str(stats.get('vector_db_chunk_count')), 'cyan')}")
        processed_files = stats.get('processed_files_list', []) # Anahtarı sunucuya göre ayarla
        if processed_files:
            print("  - İşlenmiş Dosyalar:")
            for f in processed_files:
                print(f"    - {os.path.basename(f)}")


def main():
    """Uygulamanın giriş noktası."""
    cfg = AppConfig()
    
    # ngrok'tan aldığınız URL'yi buraya yapıştırın veya bir çevre değişkeninden alın
    NGROK_URL = os.environ.get("NGROK_URL", "http://localhost:8000") # Önce çevre değişkenini dene
    
    if "localhost" in NGROK_URL:
        # Manuel giriş iste
        url_input = input(f"Lütfen ngrok URL'nizi girin (veya localhost için Enter'a basın: http://localhost:8000): ").strip()
        if url_input:
            NGROK_URL = url_input

    # DEBUG log seviyesi için argüman kontrolü
    log_level = 'DEBUG' if '--debug' in sys.argv else 'INFO'
    
    app = TerminalApplication(cfg, NGROK_URL, log_level)
    app.start()

if __name__ == '__main__':
    main()