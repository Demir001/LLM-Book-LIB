# server_components.py

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever # EnsembleRetriever genellikle ana pakette kalır.
import shutil
from datetime import datetime

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Gelişmiş Metin Temizleme Stratejileri ---

def advanced_text_cleanup(text: str) -> str:
    """
    PDF'ten çıkarılan ham metni temizlemek ve yapılandırmak için gelişmiş fonksiyon.
    Bu fonksiyon, akademik metinlerde sıkça rastlanan sorunları hedefler.
    """
    # 1. Adım: Satır sonu tirelerini birleştirme (örn: "kelime-\n" -> "kelime")
    # Bu, en yaygın bozulma nedenlerinden biridir.
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

    # 2. Adım: Tek harf veya anlamsız kısa satırları birleştirme
    # "b u n u n \n g i b i" durumlarını "bunun gibi" haline getirmeye çalışır.
    # Önce birden fazla boşluğu tek boşluğa indirgeyelim.
    text = re.sub(r'\s+', ' ', text)
    # Satır sonlarını geçici bir belirteçle değiştirelim.
    text = text.replace('\n', ' <br> ')
    # Şimdi metni kelimelere ayıralım.
    words = text.split()
    
    reconstructed_text = []
    for i, word in enumerate(words):
        # Eğer kelime çok kısaysa (örn: tek harf) ve bir sonraki kelime de kısaysa,
        # bunları birleştirmeyi deneyebiliriz. Ancak bu riskli olabilir.
        # Daha güvenli bir yaklaşım, anlamsız satır sonlarını temizlemektir.
        if word == '<br>':
            # Önceki ve sonraki kelimelere bakarak bu satır sonunun
            # bir paragraf sonu mu yoksa anlamsız bir kesme mi olduğuna karar ver.
            # Şimdilik, tüm <br>'leri tek boşlukla değiştirelim ve sonra paragrafları ayıralım.
            reconstructed_text.append(' ')
        else:
            reconstructed_text.append(word)
    
    text = " ".join(reconstructed_text)
    
    # 3. Adım: Birden fazla boşluğu tekrar tek boşluğa indirgeme
    text = re.sub(r'\s+', ' ', text).strip()

    # 4. Adım: Cümleler arasında mantıklı boşluklar bırakma.
    # Nokta, soru işareti, ünlemden sonra (eğer devamında büyük harf varsa) yeni satır ekle.
    # Bu, LLM'in metni daha iyi anlamasına yardımcı olur.
    text = re.sub(r'([.!?])\s+(?=[A-ZÇĞİÖŞÜ])', r'\1\n\n', text)

    # 5. Adım (İsteğe bağlı): Sayfa başlıkları/altlıkları gibi tekrar eden gürültüleri temizleme
    # Bu, kitaba özel olduğu için genel bir kural yazmak zordur.
    # Şimdilik bu adımı atlıyoruz, ancak gerekirse buraya eklenebilir.

    return text


class DocumentConverter:
    """
    PDF dosyalarını okur, akıllı temizlik uygular ve yapılandırılmış Document nesnelerine dönüştürür.
    Bu sınıf artık doğrudan TXT dosyası oluşturmak yerine, işlenmiş Document listesi döndürür.
    """
    def __init__(self, config):
        self._config: 'AppConfig' = config
        self._logger = logging.getLogger(self.__class__.__name__)

    def convert_pdf_to_documents(self, pdf_path: Path, book_title: str, author: str) -> Optional[List[Document]]:
        """
        Tek bir PDF dosyasını işler ve temizlenmiş Document listesi döndürür.
        Artık meta verileri (kitap adı, yazar) doğrudan alıyor.
        """
        self._logger.info(f"'{pdf_path.name}' işleniyor... (Başlık: {book_title}, Yazar: {author})")
        
        try:
            # PyPDFLoader'ı sayfa bazında veri çekmek için kullanıyoruz.
            loader = PyPDFLoader(str(pdf_path))
            raw_pages = loader.load() # load_and_split yerine sadece load kullanıyoruz.

            processed_docs = []
            for i, page_doc in enumerate(raw_pages):
                page_num = i + 1
                raw_content = page_doc.page_content

                # Gelişmiş temizlik fonksiyonunu burada çağırıyoruz!
                clean_content = advanced_text_cleanup(raw_content)

                if not clean_content or len(clean_content) < 50: # Çok kısa veya boş sayfaları atla
                    self._logger.debug(f"Sayfa {page_num} içerik yetersiz olduğu için atlandı.")
                    continue

                # Her sayfayı tek bir Document nesnesi olarak, zengin meta veriyle sakla.
                # Metin bölme (chunking) işlemini veritabanına ekleme aşamasında yapacağız.
                metadata = {
                    "source_file": pdf_path.name,
                    "book_title": book_title,
                    "author": author,
                    "page": page_num
                }
                processed_docs.append(Document(page_content=clean_content, metadata=metadata))
            
            if not processed_docs:
                self._logger.error(f"'{pdf_path.name}' dosyasından okunabilir içerik çıkarılamadı.")
                return None

            self._logger.info(f"'{pdf_path.name}' başarıyla işlendi. {len(processed_docs)} sayfa çıkarıldı.")
            return processed_docs

        except Exception as e:
            self._logger.critical(f"'{pdf_path.name}' işlenirken KRİTİK HATA oluştu. PDF bozuk olabilir. Hata: {e}", exc_info=True)
            return None


class CustomTextLoader:
    """
    BU SINIF ARTIK DOĞRUDAN KULLANILMAYACAK.
    PDF'leri doğrudan işlediğimiz için, [SAYFA: X] formatlı TXT dosyalarına ihtiyacımız kalmadı.
    Ancak eski sistemle uyumluluk veya ilerideki olası kullanımlar için burada bırakılabilir.
    """
    pass # Şimdilik boş bırakıyoruz.


class StateManager:
    """
    Sistemin durumunu (işlenen dosyalar, meta veriler vb.) yönetir.
    Bu, artık sadece dosya yollarını değil, her dosya için zengin meta verileri de saklar.
    """
    def __init__(self, state_file: Path):
        self._state_file = state_file
        self._logger = logging.getLogger(self.__class__.__name__)
        # state_file formatı: {"dosya_yolu": {"status": "processed", "metadata": {...}}}
        self._state: Dict[str, Any] = self._load()
    def _load(self) -> Dict[str, Any]:
        if not self._state_file.exists():
            return {}
        try:
            with open(self._state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self._logger.error(f"Durum dosyası ('{self._state_file}') okunamadı veya bozuk. Yeni bir dosya oluşturulacak. Hata: {e}")
            # Bozuk dosyayı yedekle
            shutil.move(str(self._state_file), str(self._state_file) + ".bak")
            return {}

    def _save(self):
        try:
            with open(self._state_file, 'w', encoding='utf-8') as f:
                json.dump(self._state, f, indent=4, ensure_ascii=False)
        except IOError as e:
            self._logger.error(f"Durum dosyası kaydedilemedi! Hata: {e}")

    def get_all_processed_files(self) -> List[str]:
        """İşlenmiş tüm dosyaların yollarını döndürür."""
        return list(self._state.keys())

    def is_file_processed(self, pdf_path: Path) -> bool:
        """Belirli bir dosyanın işlenip işlenmediğini kontrol eder."""
        return str(pdf_path.resolve()) in self._state

    def mark_file_as_processed(self, pdf_path: Path, book_title: str, author: str):
        """Bir dosyayı, meta verileriyle birlikte işlenmiş olarak işaretler."""
        file_key = str(pdf_path.resolve())
        self._state[file_key] = {
            "status": "processed",
            "processed_at": datetime.now().isoformat(),
            "metadata": {
                "book_title": book_title,
                "author": author,
                "original_filename": pdf_path.name
            }
        }
        self._save()
        self._logger.info(f"'{pdf_path.name}' dosyası işlenmiş olarak işaretlendi.")

    def reset(self):
        """Tüm durumu sıfırlar."""
        self._state = {}
        if self._state_file.exists():
            os.remove(self._state_file)
        self._logger.warning("Durum yönetimi (State) sıfırlandı.")


# server_components.py (içinde)

# ... (StateManager ve DocumentConverter sınıfları burada) ...

class VectorDatabase:
    """
    Vektör veritabanı işlemlerini yönetir.
    Hibrit Arama (Vektör + Anahtar Kelime) için optimize edilmiştir.
    """
    def __init__(self, config, embedding_model: HuggingFaceEmbeddings):
        self._config: 'AppConfig' = config
        self._logger = logging.getLogger(self.__class__.__name__)
        self._embedding_model = embedding_model
        
        # Anahtar kelime arama için kullanılacak metin bölücü
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._config.CHUNK_SIZE,
            chunk_overlap=self._config.CHUNK_OVERLAP
        )
        
        # Veritabanı ve retriever'ları başlat
        self._db: Chroma = None
        self._bm25_retriever: BM25Retriever = None
        self._all_docs_in_memory: List[Document] = [] # BM25 için tüm belgeleri hafızada tutar

        self._initialize()

    def _initialize(self):
        """
        Veritabanını başlatır, mevcut tüm belgeleri hafızaya yükler ve
        BM25 retriever'ı ilk kez kurar.
        """
        self._logger.info(f"Vektör veritabanı '{self._config.VECTOR_DB_DIR}' konumunda başlatılıyor...")
        client_settings = chromadb.Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory=str(self._config.VECTOR_DB_DIR)
        )
        self._db = Chroma(
            embedding_function=self._embedding_model,
            client_settings=client_settings
        )
        self._logger.info(f"ChromaDB başarıyla yüklendi. Koleksiyondaki belge sayısı: {self.count()}")
        
        # BM25 için veritabanındaki tüm belgeleri hafızaya yükle
        self._logger.info("Mevcut belgeler BM25 Retriever için hafızaya yükleniyor...")
        db_content = self._db.get(include=["metadatas", "documents"])
        
        # Eğer veritabanı boş değilse
        if db_content and db_content['documents']:
            self._all_docs_in_memory = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(db_content['documents'], db_content['metadatas'])
            ]
            self._rebuild_bm25_retriever()
        else:
            self._logger.warning("Veritabanı boş. BM25 Retriever kurulmadı.")

    def _rebuild_bm25_retriever(self):
        """
        Hafızadaki `_all_docs_in_memory` listesini kullanarak BM25 retriever'ı
        sıfırdan kurar. Sadece gerektiğinde çağrılmalıdır.
        """
        if not self._all_docs_in_memory:
            self._logger.warning("Hafızada belge yok, BM25 kurulamıyor.")
            self._bm25_retriever = None
            return

        self._logger.info(f"BM25 Retriever, {len(self._all_docs_in_memory)} belge ile yeniden kuruluyor...")
        self._bm25_retriever = BM25Retriever.from_documents(self._all_docs_in_memory)
        self._bm25_retriever.k = self._config.RETRIEVER_K
        self._logger.info("BM25 Retriever başarıyla kuruldu/güncellendi.")

    def add_documents(self, documents: List[Document]):
        """
        Yeni belgeleri alır, parçalara (chunks) ayırır, ChromaDB'ye ekler,
        hafızadaki listeyi günceller ve BM25'i yeniden kurar.
        """
        if not documents:
            self._logger.warning("Eklenecek belge bulunamadı.")
            return

        source_file = documents[0].metadata.get("source_file", "Bilinmeyen Dosya")
        self._logger.info(f"'{source_file}' için metin parçalama (chunking) işlemi başlıyor...")
        
        chunks = self._text_splitter.split_documents(documents)
        
        if not chunks:
            self._logger.warning(f"'{source_file}' dosyasından chunk oluşturulamadı.")
            return

        # 1. Yeni chunk'ları ChromaDB'ye ekle
        self._logger.info(f"'{source_file}' içeriğinden {len(chunks)} metin parçası ChromaDB'ye ekleniyor...")
        self._db.add_documents(chunks)
        
        # 2. Yeni chunk'ları hafızadaki listeye ekle
        self._all_docs_in_memory.extend(chunks)
        
        # 3. BM25 Retriever'ı güncellenmiş tam liste ile yeniden kur
        self._rebuild_bm25_retriever()

    def get_retriever(self) -> Any:
        """
        Her çağrıldığında anlık olarak güncel bir hibrit (Ensemble) retriever oluşturur.
        """
        self._logger.info("Hibrit retriever isteniyor...")
        # Vektör arama için retriever (her zaman günceldir)
        chroma_retriever = self._db.as_retriever(search_kwargs={"k": self._config.RETRIEVER_K})
        
        # Eğer BM25 hazır değilse, sadece Chroma'yı döndür
        if not self._bm25_retriever:
            self._logger.warning("BM25 Retriever mevcut değil. Yalnızca vektör arama kullanılacak.")
            return chroma_retriever

        # İki retriever'ı birleştirerek hibrit retriever oluştur
        ensemble_retriever = EnsembleRetriever(
            retrievers=[self._bm25_retriever, chroma_retriever],
            weights=[0.5, 0.5]  # Anahtar kelime ve vektöre eşit ağırlık ver. Ayarlanabilir.
        )
        self._logger.info("Hibrit (Ensemble) Retriever başarıyla oluşturuldu.")
        return ensemble_retriever
    
    def count(self) -> int:
        """Veritabanındaki toplam chunk sayısını döndürür."""
        try:
            return self._db._collection.count()
        except Exception:
            return 0

    def rebuild(self):
        """Veritabanını ve hafızadaki durumu tamamen siler ve yeniden oluşturur."""
        self._logger.warning("Vektör veritabanı ve hafızadaki belgeler siliniyor...")
        if self._config.VECTOR_DB_DIR.exists():
            shutil.rmtree(self._config.VECTOR_DB_DIR)
        
        # Hafızayı temizle
        self._all_docs_in_memory = []
        self._bm25_retriever = None

        # Her şeyi sıfırdan başlat
        self._initialize()
        self._logger.info("Vektör veritabanı ve hibrit arama durumu başarıyla sıfırlandı.")
