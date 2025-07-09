# shared_utils.py

from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class DocumentInfo:
    """
    Bir metin parçasının (chunk) tüm bilgilerini taşıyan standart veri sınıfı.
    Bu yapı, sunucu ve istemci arasında JSON olarak gönderilecektir.
    """
    page_content: str
    metadata: Dict[str, Any]

@dataclass
class QueryResult:
    """
    Sunucudan istemciye döndürülecek ham arama sonuçlarını temsil eder.
    """
    retrieved_docs: List[DocumentInfo]
    query: str
    message: str = "" # Örneğin "Arama başarılı" veya "Hata oluştu"

@dataclass
class FinalResponse:
    """
    İstemcinin son kullanıcıya göstereceği nihai cevabı temsil eder.
    """
    final_answer: str
    verified_sources: List[Dict[str, Any]] = field(default_factory=list)
    unverified_sources: List[Dict[str, Any]] = field(default_factory=list)
    is_fully_verified: bool = False
    raw_llm_answer: str = "" # Debug için LLM'in ham cevabı

class ConsoleFormatter:
    """
    Terminalde renkli ve biçimli çıktılar oluşturmak için yardımcı sınıf.
    """
    # ANSI escape kodları
    COLORS = {
        'HEADER': '\033[95m',
        'BLUE': '\033[94m',
        'CYAN': '\033[96m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'RED': '\033[91m',
        'BOLD': '\033[1m',
        'UNDERLINE': '\033[4m',
        'END': '\033[0m',
    }

    @classmethod
    def color(cls, text: str, *styles: str) -> str:
        """
        Metne bir veya daha fazla stil (renk, kalın vb.) uygular.
        Örnek: color("Merhaba", "red", "bold")
        """
        style_codes = "".join(cls.COLORS.get(s.upper(), "") for s in styles)
        return f"{style_codes}{text}{cls.COLORS['END']}"

    @classmethod
    def print_banner(cls, text: str, style: str = 'header'):
        """
        Belirgin bir başlık (banner) yazdırır.
        """
        color_code = cls.COLORS.get(style.upper(), cls.COLORS['HEADER'])
        line = color_code + '=' * 80 + cls.COLORS['END']
        centered_text = cls.color(text.center(80), 'bold', style)
        print(f"\n{line}\n{centered_text}\n{line}")