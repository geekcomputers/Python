"""Currency Converter Application

A PyQt5-based currency converter with embedded currency data (no external files required).
Fetches real-time exchange rates and provides a user-friendly interface.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLineEdit, QPushButton, 
                             QComboBox, QLCDNumber, QWidget)
from PyQt5 import uic
import httpx


# Embedded currency data (originally from country.txt)
CURRENCIES = [
    "Australia Dollar-AUD",
    "Great Britain Pound-GBP",
    "Euro-EUR",
    "Japan Yen-JPY",
    "Switzerland Franc-CHF",
    "USA Dollar-USD",
    "Afghanistan Afghani-AFN",
    "Albania Lek-ALL",
    "Algeria Dinar-DZD",
    "Angola Kwanza-AOA",
    "Argentina Peso-ARS",
    "Armenia Dram-AMD",
    "Aruba Florin-AWG",
    "Austria Schilling-ATS (EURO)",
    "Belgium Franc-BEF (EURO)",
    "Azerbaijan New Manat-AZN",
    "Bahamas Dollar-BSD",
    "Bahrain Dinar-BHD",
    "Bangladesh Taka-BDT",
    "Barbados Dollar-BBD",
    "Belarus Ruble-BYR",
    "Belize Dollar-BZD",
    "Bermuda Dollar-BMD",
    "Bhutan Ngultrum-BTN",
    "Bolivia Boliviano-BOB",
    "Bosnia Mark-BAM",
    "Botswana Pula-BWP",
    "Brazil Real-BRL",
    "Brunei Dollar-BND",
    "Bulgaria Lev-BGN",
    "Burundi Franc-BIF",
    "CFA Franc BCEAO-XOF",
    "CFA Franc BEAC-XAF",
    "CFP Franc-XPF",
    "Cambodia Riel-KHR",
    "Canada Dollar-CAD",
    "Cape Verde Escudo-CVE",
    "Cayman Islands Dollar-KYD",
    "Chili Peso-CLP",
    "China Yuan/Renminbi-CNY",
    "Colombia Peso-COP",
    "Comoros Franc-KMF",
    "Congo Franc-CDF",
    "Costa Rica Colon-CRC",
    "Croatia Kuna-HRK",
    "Cuba Convertible Peso-CUC",
    "Cuba Peso-CUP",
    "Cyprus Pound-CYP (EURO)",
    "Czech Koruna-CZK",
    "Denmark Krone-DKK",
    "Djibouti Franc-DJF",
    "Dominican Republich Peso-DOP",
    "East Caribbean Dollar-XCD",
    "Egypt Pound-EGP",
    "El Salvador Colon-SVC",
    "Estonia Kroon-EEK (EURO)",
    "Ethiopia Birr-ETB",
    "Falkland Islands Pound-FKP",
    "Finland Markka-FIM (EURO)",
    "Fiji Dollar-FJD",
    "Gambia Dalasi-GMD",
    "Georgia Lari-GEL",
    "Germany Mark-DMK (EURO)",
    "Ghana New Cedi-GHS",
    "Gibraltar Pound-GIP",
    "Greece Drachma-GRD (EURO)",
    "Guatemala Quetzal-GTQ",
    "Guinea Franc-GNF",
    "Guyana Dollar-GYD",
    "Haiti Gourde-HTG",
    "Honduras Lempira-HNL",
    "Hong Kong Dollar-HKD",
    "Hungary Forint-HUF",
    "Iceland Krona-ISK",
    "India Rupee-INR",
    "Indonesia Rupiah-IDR",
    "Iran Rial-IRR",
    "Iraq Dinar-IQD",
    "Ireland Pound-IED (EURO)",
    "Israel New Shekel-ILS",
    "Italy Lira-ITL (EURO)",
    "Jamaica Dollar-JMD",
    "Jordan Dinar-JOD",
    "Kazakhstan Tenge-KZT",
    "Kenya Shilling-KES",
    "Kuwait Dinar-KWD",
    "Kyrgyzstan Som-KGS",
    "Laos Kip-LAK",
    "Latvia Lats-LVL (EURO)",
    "Lebanon Pound-LBP",
    "Lesotho Loti-LSL",
    "Liberia Dollar-LRD",
    "Libya Dinar-LYD",
    "Lithuania Litas-LTL (EURO)",
    "Luxembourg Franc-LUF (EURO)",
    "Macau Pataca-MOP",
    "Macedonia Denar-MKD",
    "Malagasy Ariary-MGA",
    "Malawi Kwacha-MWK",
    "Malaysia Ringgit-MYR",
    "Maldives Rufiyaa-MVR",
    "Malta Lira-MTL (EURO)",
    "Mauritania Ouguiya-MRO",
    "Mauritius Rupee-MUR",
    "Mexico Peso-MXN",
    "Moldova Leu-MDL",
    "Mongolia Tugrik-MNT",
    "Morocco Dirham-MAD",
    "Mozambique New Metical-MZN",
    "Myanmar Kyat-MMK",
    "NL Antilles Guilder-ANG",
    "Namibia Dollar-NAD",
    "Nepal Rupee-NPR",
    "Netherlands Guilder-NLG (EURO)",
    "New Zealand Dollar-NZD",
    "Nicaragua Cordoba Oro-NIO",
    "Nigeria Naira-NGN",
    "North Korea Won-KPW",
    "Norway Kroner-NOK",
    "Oman Rial-OMR",
    "Pakistan Rupee-PKR",
    "Panama Balboa-PAB",
    "Papua New Guinea Kina-PGK",
    "Paraguay Guarani-PYG",
    "Peru Nuevo Sol-PEN",
    "Philippines Peso-PHP",
    "Poland Zloty-PLN",
    "Portugal Escudo-PTE (EURO)",
    "Qatar Rial-QAR",
    "Romania New Lei-RON",
    "Russia Rouble-RUB",
    "Rwanda Franc-RWF",
    "Samoa Tala-WST",
    "Sao Tome/Principe Dobra-STD",
    "Saudi Arabia Riyal-SAR",
    "Serbia Dinar-RSD",
    "Seychelles Rupee-SCR",
    "Sierra Leone Leone-SLL",
    "Singapore Dollar-SGD",
    "Slovakia Koruna-SKK (EURO)",
    "Slovenia Tolar-SIT (EURO)",
    "Solomon Islands Dollar-SBD",
    "Somali Shilling-SOS",
    "South Africa Rand-ZAR",
    "South Korea Won-KRW",
    "Spain Peseta-ESP (EURO)",
    "Sri Lanka Rupee-LKR",
    "St Helena Pound-SHP",
    "Sudan Pound-SDG",
    "Suriname Dollar-SRD",
    "Swaziland Lilangeni-SZL",
    "Sweden Krona-SEK",
    "Syria Pound-SYP",
    "Taiwan Dollar-TWD",
    "Tanzania Shilling-TZS",
    "Thailand Baht-THB",
    "Tonga Pa'anga-TOP",
    "Trinidad/Tobago Dollar-TTD",
    "Tunisia Dinar-TND",
    "Turkish New Lira-TRY",
    "Turkmenistan Manat-TMM",
    "Uganda Shilling-UGX",
    "Ukraine Hryvnia-UAH",
    "Uruguay Peso-UYU",
    "United Arab Emirates Dirham-AED",
    "Vanuatu Vatu-VUV",
    "Venezuela Bolivar-VEB",
    "Vietnam Dong-VND",
    "Yemen Rial-YER",
    "Zambia Kwacha-ZMK",
    "Zimbabwe Dollar-ZWD"
]


class CurrencyConverter(QMainWindow):
    """Main application window for currency conversion"""
    
    def __init__(self) -> None:
        super().__init__()
        self.base_dir: Path = Path(__file__).parent  # Directory of current script
        self.ui: QWidget = self.load_ui()
        self.setup_ui_elements()
        self.load_currency_data()  # Load from embedded list
        
        # Type hints for UI elements (assigned via UI file)
        self.lineEdit: QLineEdit
        self.pushButton: QPushButton
        self.dropDown1: QComboBox
        self.dropDown2: QComboBox
        self.lcdpanel: QLCDNumber


    def load_ui(self) -> QWidget:
        """Load UI file with proper path handling"""
        ui_path: Path = self.base_dir / "gui.ui"
        
        try:
            return uic.loadUi(str(ui_path), self)
        except FileNotFoundError:
            print(f"Error: UI file not found at {ui_path}")
            print(f"Please ensure 'gui.ui' exists in: {self.base_dir}")
            sys.exit(1)


    def setup_ui_elements(self) -> None:
        """Initialize UI components and connections"""
        self.lineEdit.setValidator(QDoubleValidator())
        self.pushButton.clicked.connect(self.convert_currency)
        self.setWindowTitle("Dynamic Currency Converter")


    def load_currency_data(self) -> None:
        """Load currency list from embedded data (no external file)"""
        # Remove duplicates while preserving order
        seen: set = set()
        unique_currencies: List[str] = []
        for currency in CURRENCIES:
            if currency not in seen:
                seen.add(currency)
                unique_currencies.append(currency)
        
        # Populate dropdowns
        self.dropDown1.addItem("Select Currency")
        self.dropDown2.addItem("Select Currency")
        for currency in unique_currencies:
            self.dropDown1.addItem(currency)
            self.dropDown2.addItem(currency)


    def get_exchange_rate(self, from_currency: str, to_currency: str) -> Optional[float]:
        """Fetch exchange rate from API"""
        try:
            # Extract currency codes (format: "Name-CODE")
            from_code: str = from_currency.split("-")[-1].strip()
            to_code: str = to_currency.split("-")[-1].strip()
            
            # Clean up codes with extra info (e.g., "(EURO)")
            from_code = from_code.split()[0]
            to_code = to_code.split()[0]
            
            # API request
            api_key: str = "b43a653672c4a94c4c26"  # Replace with your API key if needed
            url: str = f"https://free.currconv.com/api/v7/convert?q={from_code}_{to_code}&compact=ultra&apiKey={api_key}"
            
            response: httpx.Response = httpx.get(url, timeout=10)
            response.raise_for_status()
            data: Dict[str, float] = response.json()
            
            rate_key: str = f"{from_code}_{to_code}"
            return float(data[rate_key]) if rate_key in data else None
            
        except Exception as e:
            print(f"Error fetching exchange rate: {str(e)}")
            return None


    def convert_currency(self) -> None:
        """Handle currency conversion when button is clicked"""
        amount_text: str = self.lineEdit.text()
        from_currency: str = self.dropDown1.currentText()
        to_currency: str = self.dropDown2.currentText()
        
        # Validate inputs
        if not amount_text or from_currency == "Select Currency" or to_currency == "Select Currency":
            self.lcdpanel.display(0)
            return
            
        try:
            amount: float = float(amount_text)
            rate: Optional[float] = self.get_exchange_rate(from_currency, to_currency)
            
            if rate is not None:
                result: float = amount * rate
                self.lcdpanel.display(round(result, 2))
            else:
                self.lcdpanel.display(0)
                
        except ValueError:
            self.lcdpanel.display(0)


if __name__ == "__main__":
    # High-DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app: QApplication = QApplication(sys.argv)
    window: CurrencyConverter = CurrencyConverter()
    window.show()
    sys.exit(app.exec_())