# utils.py

import os
from typing import Dict

class ConsoleManager:
    """Terminal çıktılarını ve girdilerini yönetmek için yardımcı sınıf."""
    COLORS: Dict[str, str] = {
        'HEADER': '\033[95m', 'BLUE': '\033[94m', 'CYAN': '\033[96m',
        'GREEN': '\033[92m', 'YELLOW': '\033[93m', 'RED': '\033[91m',
        'BOLD': '\033[1m', 'END': '\033[0m'
    }
    
    @classmethod
    def color(cls, text: str, color: str) -> str:
        """Verilen metni belirtilen renge boyar."""
        return f"{cls.COLORS.get(color.upper(), '')}{text}{cls.COLORS['END']}"

    @staticmethod
    def get_user_input(prompt: str, color: str = "CYAN") -> str:
        """Kullanıcıya renkli bir prompt gösterir ve girdi alır."""
        # Windows'ta renk kodlarının düzgün çalışması için os.system('') çağrısı
        if os.name == 'nt':
            os.system('')
        return input(ConsoleManager.color(prompt, color))

    @staticmethod
    def print_info(message: str, color: str = "BLUE"):
        """Ekrana bilgilendirici, renkli bir mesaj basar."""
        print(ConsoleManager.color(message, color))

    @staticmethod
    def print_header(text: str):
        """Uygulama için standart bir başlık oluşturur."""
        banner = f"\n{ConsoleManager.color('='*70, 'BLUE')}\n"
        banner += f"{ConsoleManager.color(text.center(70), 'BOLD')}\n"
        banner += f"{ConsoleManager.color('='*70, 'BLUE')}"
        print(banner)