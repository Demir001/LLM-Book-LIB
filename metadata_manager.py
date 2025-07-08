# metadata_manager.py

import json
import logging
from pathlib import Path
from dataclasses import asdict

from config import APP_CONFIG, DocumentMetadata
from utils import ConsoleManager

class MetadataManager:
    """
    Kitapların meta verilerini (.json dosyaları aracılığıyla) yönetir.
    Eksik meta verileri için kullanıcıdan girdi alır ve dosyaları oluşturur.
    """
    def __init__(self):
        self._config = APP_CONFIG
        self._logger = logging.getLogger(self.__class__.__name__)
        self._metadata_dir = self._config.METADATA_DIR
        self._console = ConsoleManager()

    def ensure_metadata_for_all_pdfs(self) -> None:
        """
        Kütüphane klasöründeki tüm PDF'leri tarar ve her biri için
        bir meta veri (.json) dosyasının var olduğundan emin olur.
        Eksik olanlar için kullanıcıdan bilgi ister.
        """
        self._logger.info("Meta veri dosyaları kontrol ediliyor...")
        self._metadata_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_files = list(self._config.SOURCE_DIR.glob("*.pdf"))
        if not pdf_files:
            self._logger.warning(f"'{self._config.SOURCE_DIR}' klasöründe PDF dosyası bulunamadı.")
            return

        for pdf_path in pdf_files:
            metadata_path = self._metadata_dir / f"{pdf_path.stem}.json"
            if not metadata_path.exists():
                self._logger.warning(f"'{pdf_path.name}' için meta veri eksik. Kullanıcıdan bilgi isteniyor.")
                self._create_metadata_interactively(pdf_path, metadata_path)

    def _create_metadata_interactively(self, pdf_path: Path, metadata_path: Path):
        """Kullanıcı ile etkileşime girerek bir kitap için meta veri oluşturur."""
        self._console.print_header(f"YENİ KİTAP TESPİT EDİLDİ: {pdf_path.name}")
        self._console.print_info("Lütfen bu kitap için aşağıdaki bilgileri girin:", "YELLOW")

        default_title = pdf_path.stem.replace('_', ' ').replace('-', ' ').title()
        
        title_prompt = f"  Kitap Başlığı [Varsayılan: {default_title}]: "
        title_input = self._console.get_user_input(title_prompt, "GREEN")
        title = title_input.strip() or default_title

        author = ""
        while not author:
            author_input = self._console.get_user_input("  Yazar Adı: ", "GREEN")
            author = author_input.strip()
            if not author:
                self._console.print_info("Yazar adı boş bırakılamaz. Lütfen tekrar girin.", "RED")

        metadata = DocumentMetadata(
            source_file=pdf_path.name,
            title=title,
            author=author
        )
        
        self._save_metadata(metadata, metadata_path)
        self._console.print_info(f"Meta veri '{metadata_path.name}' olarak başarıyla kaydedildi.", "GREEN")

    def get_metadata_for_pdf(self, pdf_path: Path) -> DocumentMetadata | None:
        """Belirtilen bir PDF dosyasına ait meta veriyi .json dosyasından okur."""
        metadata_path = self._metadata_dir / f"{pdf_path.stem}.json"
        if not metadata_path.exists():
            self._logger.error(f"'{metadata_path}' bulunamadı. Lütfen önce senkronizasyon yapın.")
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return DocumentMetadata(**data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self._logger.error(f"'{metadata_path}' dosyası bozuk veya geçersiz. Hata: {e}")
            return None

    def _save_metadata(self, metadata: DocumentMetadata, path: Path):
        """DocumentMetadata nesnesini bir .json dosyasına yazar."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(metadata), f, indent=4, ensure_ascii=False)