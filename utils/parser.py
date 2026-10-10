import re
from pypdf import PdfReader
import docx

def extract_text_from_file(uploaded_file) -> str:
    """Extracts raw text from uploaded PDF, DOCX, or TXT file objects."""
    file_type = uploaded_file.name.split('.')[-1].lower()
    text = ""
    
    if file_type == 'pdf':
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
                
    elif file_type == 'docx':
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
        
    elif file_type == 'txt':
        text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        
    return text

def parse_tangedco_bill(raw_text: str) -> dict:
    """Bulletproof regex parser supporting multiple TANGEDCO bill layouts."""
    
    # 1. Total Units Patterns (Multiple naming variations)
    unit_patterns = [
        r"Total\s*Units\s*Consumed\s*[:\|=-]*\s*(\d+)",
        r"Consumed\s*Units\s*[:\|=-]*\s*(\d+)",
        r"Billed\s*Units\s*[:\|=-]*\s*(\d+)",
        r"Total\s*Consumption\s*[:\|=-]*\s*(\d+)",
        r"Units\s*[:\|=-]*\s*(\d+)"
    ]
    units = None
    for p in unit_patterns:
        match = re.search(p, raw_text, re.IGNORECASE)
        if match:
            units = int(match.group(1))
            break

    # 2. Sanctioned Load Patterns
    load_patterns = [
        r"Sanctioned\s*Load\s*[:\|=-]*\s*([\d\.]+)\s*(?:kW|HP)?",
        r"Connected\s*Load\s*[:\|=-]*\s*([\d\.]+)\s*(?:kW|HP)?",
        r"Load\s*[:\|=-]*\s*([\d\.]+)\s*kW"
    ]
    sanctioned = None
    for p in load_patterns:
        match = re.search(p, raw_text, re.IGNORECASE)
        if match:
            sanctioned = float(match.group(1))
            break

    # 3. Bill Amount Patterns
    amt_patterns = [
        r"NET\s*AMOUNT\s*PAYABLE\s*(?:\(Rs\.\))?\s*[:\|=-]*\s*(?:Rs\.?)?\s*([\d,]+\.?\d*)",
        r"Total\s*Amount\s*[:\|=-]*\s*(?:Rs\.?)?\s*([\d,]+\.?\d*)",
        r"Bill\s*Amount\s*[:\|=-]*\s*(?:Rs\.?)?\s*([\d,]+\.?\d*)"
    ]
    net_amt = None
    for p in amt_patterns:
        match = re.search(p, raw_text, re.IGNORECASE)
        if match:
            net_amt = float(match.group(1).replace(",", ""))
            break

    # Tariff Category Check
    tariff_match = re.search(r"Tariff\s*(?:Category)?\s*[:\|=-]*\s*([^\n\r]+)", raw_text, re.IGNORECASE)
    tariff = tariff_match.group(1).strip() if tariff_match else "LT-1A (DOMESTIC)"

    daily_kwh = round(units / 60.0, 2) if units else None

    return {
        "tariff": tariff,
        "is_domestic": "1A" in tariff.upper(),
        "units_consumed": units,
        "daily_kwh": daily_kwh,
        "sanctioned_load_kw": sanctioned or 3.0,
        "bill_amount": net_amt,
        "raw_text_length": len(raw_text)
    }