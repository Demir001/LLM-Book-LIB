# server.py

import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import cgi

# Projemizin kendi modüllerini import ediyoruz
from config import AppConfig
from shared_utils import ConsoleFormatter, QueryResult, DocumentInfo
from server_components import DocumentConverter, StateManager, VectorDatabase

# Gerekli kütüphaneleri import et (bu sunucuda LLM olmayacak, sadece embedding modeli)
from langchain_huggingface import HuggingFaceEmbeddings

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Aynı anda birden fazla isteği işleyebilen sunucu."""
    pass

class RequestHandler(BaseHTTPRequestHandler):
    """
    Gelen HTTP isteklerini işleyen sınıf.
    Endpoint'leri (URL yolları) burada tanımlayacağız.
    """
    # Sınıf seviyesinde bileşenleri tanımla, böylece tüm isteklerde aynı nesneler kullanılır
    config = AppConfig()
    formatter = ConsoleFormatter()

    # Embedding modelini bir kez yükle ve bellekte tut
    formatter.print_banner("SERVER BAŞLATILIYOR", "blue")
    print(formatter.color("Embedding modeli yükleniyor...", "cyan"))
    embedding_model = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'} # Sunucu CPU üzerinde çalışacak şekilde ayarlandı
    )
    print(formatter.color("Embedding modeli yüklendi.", "green"))

    # Diğer sunucu bileşenlerini başlat
    state_manager = StateManager(config.STATE_FILE)
    converter = DocumentConverter(config)
    db = VectorDatabase(config, embedding_model)
    retriever = db.get_retriever()

    def _send_response(self, status_code, data):
        """İstemciye standart bir JSON cevabı gönderir."""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, default=vars).encode('utf-8'))

    def do_POST(self):
        """Sadece POST isteklerini kabul ediyoruz."""
        try:
            if self.path == '/sync':
                self.handle_sync()
            elif self.path == '/retrieve':
                self.handle_retrieve()
            else:
                self._send_response(404, {"error": "Endpoint not found"})
        except Exception as e:
            logging.critical("POST isteği işlenirken kritik hata!", exc_info=True)
            self._send_response(500, {"error": f"Internal Server Error: {e}"})
    
    def do_GET(self):
        """GET isteklerini de tanımlayalım (örn: durum kontrolü için)"""
        try:
            if self.path == '/status':
                self.handle_status()
            else:
                self._send_response(404, {"error": "Endpoint not found"})
        except Exception as e:
            logging.critical("GET isteği işlenirken kritik hata!", exc_info=True)
            self._send_response(500, {"error": f"Internal Server Error: {e}"})

    def handle_sync(self):
        """
        /kitaplar klasörünü tarar, yeni PDF'leri işler ve veritabanına ekler.
        Bu versiyonda, meta veri girişi için interaktif bir yapı kurulur.
        """
        self.formatter.print_banner("Kütüphane Senkronizasyonu Başlatıldı", "yellow")
        
        all_pdfs = list(self.config.SOURCE_PDF_DIR.glob("*.pdf"))
        new_files_processed = 0

        for pdf_path in all_pdfs:
            if not self.state_manager.is_file_processed(pdf_path):
                self.formatter.color(f"-> Yeni dosya bulundu: {pdf_path.name}", "cyan")
                
                # --- KULLANICI ETKİLEŞİMİ BURADA ---
                # Sunucunun çalıştığı terminalde kullanıcıdan bilgi istenir.
                print(f"Lütfen '{pdf_path.name}' için bilgileri girin:")
                book_title = input("   - Kitap Adı: ").strip()
                author = input("   - Yazar Adı: ").strip()

                if not book_title or not author:
                    print(self.formatter.color("Kitap adı ve yazar boş bırakılamaz. Bu dosya atlanıyor.", "red"))
                    continue

                # Adım 1: PDF'i işle ve Document listesi al
                documents = self.converter.convert_pdf_to_documents(pdf_path, book_title, author)
                if not documents:
                    print(self.formatter.color(f"'{pdf_path.name}' işlenemedi. Hata için logları kontrol edin.", "red"))
                    continue
                
                # Adım 2: Belgeleri veritabanına ekle
                self.db.add_documents(documents)
                
                # Adım 3: Dosyayı işlenmiş olarak işaretle
                self.state_manager.mark_file_as_processed(pdf_path, book_title, author)
                new_files_processed += 1

        if new_files_processed == 0:
            message = "Kütüphane güncel. Yeni dosya bulunamadı."
        else:
            message = f"{new_files_processed} yeni dosya başarıyla işlendi ve veritabanına eklendi."
        
        print(self.formatter.color(message, "green"))
        self._send_response(200, {"status": "success", "message": message})

    def handle_retrieve(self):
        """
        İstemciden gelen bir sorgu için ilgili belgeleri veritabanından çeker.
        """
        ctype, pdict = cgi.parse_header(self.headers.get('content-type'))
        if ctype != 'application/json':
            self._send_response(400, {'error': 'Content-Type must be application/json'})
            return
        
        content_length = int(self.headers.get('content-length'))
        post_data = json.loads(self.rfile.read(content_length))
        query = post_data.get('query')

        if not query:
            self._send_response(400, {'error': 'Query parameter is missing'})
            return

        logging.info(f"Gelen sorgu: '{query}'. Belgeler alınıyor...")
        
        # Retriever'ı kullanarak belgeleri al
        retrieved_docs = self.retriever.invoke(query)
        
        # Langchain Document nesnelerini, paylaşılan DocumentInfo'ya dönüştür
        doc_infos = [DocumentInfo(page_content=doc.page_content, metadata=doc.metadata) for doc in retrieved_docs]

        # QueryResult nesnesi oluştur ve gönder
        result = QueryResult(
            retrieved_docs=doc_infos, 
            query=query,
            message=f"{len(doc_infos)} adet belge başarıyla alındı."
        )
        self._send_response(200, result.__dict__)

    def handle_status(self):
        """Sunucunun mevcut durumunu döndürür."""
        status_data = {
            "status": "online",
            "processed_files_count": len(self.state_manager.get_all_processed_files()),
            "processed_files": self.state_manager.get_all_processed_files(),
            "vector_db_chunk_count": self.db.count()
        }
        self._send_response(200, status_data)

def run_server():
    """Sunucuyu başlatır ve çalıştırır."""
    config = AppConfig()
    
    # Gerekli klasörlerin var olduğundan emin ol
    config.SOURCE_PDF_DIR.mkdir(parents=True, exist_ok=True)
    config.SERVER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_TXT_DIR.mkdir(parents=True, exist_ok=True) # Artık kullanılmıyor ama uyumluluk için kalabilir

    # Logging ayarları
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    server_address = (config.SERVER_HOST, config.SERVER_PORT)
    # httpd = HTTPServer(server_address, RequestHandler) # Tek thread
    httpd = ThreadingHTTPServer(server_address, RequestHandler) # Çoklu thread
    
    formatter = ConsoleFormatter()
    formatter.print_banner(f"SUNUCU {config.SERVER_HOST}:{config.SERVER_PORT} ÜZERİNDE ÇALIŞIYOR", "green")
    print(formatter.color("İstemci bağlantısı bekleniyor... Kapatmak için CTRL+C", "yellow"))
    print(formatter.color("Yeni kitap eklediyseniz, istemciden /sync isteği gönderin.", "yellow"))
    
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()