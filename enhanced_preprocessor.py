"""
Enhanced ESG Document Preprocessor
===================================
Improves text recognition, table extraction, and numeric value detection
from BRSR (Business Responsibility and Sustainability Report) documents.
"""

import os
import re
from typing import Dict, List, Tuple, Optional
import unicodedata


# ==========================================
# PART 1: Character Recognition & Normalization
# ==========================================

def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters for better text recognition.
    Converts fancy quotes, dashes, etc. to ASCII equivalents.
    """
    # Normalize to NFKC (compatibility composition)
    text = unicodedata.normalize("NFKC", text)
    
    # Common Unicode replacements
    replacements = {
        # Quotes
        '\u2018': "'", '\u2019': "'",  # Single curly quotes
        '\u201c': '"', '\u201d': '"',  # Double curly quotes
        '\u201a': "'", '\u201b': "'",  # Low quotes
        '\u201e': '"', '\u201f': '"',  
        '\u00ab': '"', '\u00bb': '"',  # Guillemets
        # Dashes
        '\u2010': '-', '\u2011': '-',  # Hyphen variants
        '\u2012': '-', '\u2013': '-',  # Figure dash, en dash
        '\u2014': '-', '\u2015': '-',  # Em dash, horizontal bar
        '\u2212': '-',  # Minus sign
        # Spaces
        '\u00a0': ' ', '\u2002': ' ',  # Non-breaking, en space
        '\u2003': ' ', '\u2009': ' ',  # Em space, thin space
        '\u200a': ' ', '\u200b': '',   # Hair space, zero-width
        '\u202f': ' ', '\u205f': ' ',  # Narrow, medium math space
        # Numbers
        '\u00b9': '1', '\u00b2': '2', '\u00b3': '3',  # Superscripts
        '\u2070': '0', '\u2074': '4', '\u2075': '5',
        '\u2076': '6', '\u2077': '7', '\u2078': '8', '\u2079': '9',
        # Fractions
        '\u00bd': '0.5', '\u00bc': '0.25', '\u00be': '0.75',
        # Special chars
        '\u2022': '-',  # Bullet point
        '\u2023': '>',  # Triangular bullet
        '\u25aa': '-', '\u25cf': '-',  # Square, circle bullets
        '\u00d7': 'x',  # Multiplication
        '\u00f7': '/',  # Division
        '\u00b0': ' degrees ',  # Degree
        '\u2103': ' Celsius ',  # Celsius
        '\u20b9': 'INR ',  # Rupee symbol
        '\u20ac': 'EUR ',  # Euro
        '\u0024': 'USD ',  # Dollar (leave as-is usually)
        # Indian number system markers
        '₹': 'INR ',
        '`': 'INR ',  # Commonly used as rupee in Indian docs
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text


def fix_ocr_errors(text: str) -> str:
    """
    Fix common OCR/PDF extraction errors.
    """
    # Common OCR misreads
    ocr_fixes = [
        # Letter/number confusion
        (r'\bl\b(?=\d)', '1'),  # "l" before number -> 1
        (r'(?<=\d)l(?=\d)', '1'),  # digit-l-digit -> digit-1-digit
        (r'(?<=\d)O(?=\d)', '0'),  # digit-O-digit -> digit-0-digit
        (r'(?<=\d)o(?=\d)', '0'),  # digit-o-digit -> digit-0-digit
        (r'(?<=\d)I(?=\d)', '1'),  # digit-I-digit -> digit-1-digit
        (r'(?<=\d)S(?=\d)', '5'),  # digit-S-digit -> digit-5-digit
        (r'(?<=\d)Z(?=\d)', '2'),  # digit-Z-digit -> digit-2-digit
        (r'(?<=\d)B(?=\d)', '8'),  # digit-B-digit -> digit-8-digit
        
        # Number formatting issues
        (r'(\d),(\d{3})', r'\1\2'),  # Remove thousands separator confusion
        (r'(\d+)\s*\.\s*(\d+)', r'\1.\2'),  # Fix spaced decimals
        
        # Percentage issues
        (r'(\d+)\s*°/o', r'\1%'),  # °/o -> %
        (r'(\d+)\s*0/0', r'\1%'),  # 0/0 -> %
        (r'(\d+)\s*%\s*%', r'\1%'),  # Duplicate %
        
        # Unit issues
        (r'tC02e', 'tCO2e'),  # Common OCR error for CO2
        (r'tc02e', 'tCO2e'),
        (r'TCO2e', 'tCO2e'),
        (r'Mt\s*CO2', 'MtCO2'),
        (r'k[Ww]h', 'kWh'),
        (r'M[Ww]h', 'MWh'),
        (r'G[Jj]', 'GJ'),
        (r'k[Ll]', 'kL'),
        (r'[Mm]egalitre', 'ML'),
        (r'[Kk]ilolitre', 'kL'),
    ]
    
    for pattern, replacement in ocr_fixes:
        text = re.sub(pattern, replacement, text)
    
    return text


# ==========================================
# PART 2: Table Detection & Extraction
# ==========================================

def extract_tables(text: str) -> Tuple[str, List[Dict]]:
    """
    Detect and parse tabular data from text.
    Returns (cleaned_text, list_of_parsed_tables).
    """
    tables = []
    
    # Pattern 1: Lines with multiple number columns (typical BRSR tables)
    # Example: "Scope 1    1234.5    1200.3    1100.2"
    table_row_pattern = re.compile(
        r'^(.{5,50}?)\s+'  # Label/description (5-50 chars)
        r'([\d,]+\.?\d*)\s+'  # Number 1
        r'([\d,]+\.?\d*)\s*'  # Number 2
        r'([\d,]+\.?\d*)?\s*$',  # Optional Number 3
        re.MULTILINE
    )
    
    # Pattern 2: Key-value pairs in tables
    # Example: "Total Employees : 10,485" or "Total Employees | 10,485"
    kv_pattern = re.compile(
        r'^([A-Za-z][A-Za-z\s\(\)]+?)'  # Key
        r'\s*[:\|]\s*'  # Separator
        r'([\d,]+\.?\d*%?)\s*$',  # Value
        re.MULTILINE
    )
    
    # Extract key-value pairs
    for match in kv_pattern.finditer(text):
        key, value = match.groups()
        tables.append({
            "type": "key_value",
            "key": key.strip(),
            "value": parse_number(value),
            "raw": match.group(0)
        })
    
    # Extract table rows
    for match in table_row_pattern.finditer(text):
        groups = match.groups()
        tables.append({
            "type": "table_row",
            "label": groups[0].strip(),
            "values": [parse_number(g) for g in groups[1:] if g],
            "raw": match.group(0)
        })
    
    return text, tables


def parse_number(s: str) -> Optional[float]:
    """
    Parse a string into a number, handling various formats.
    """
    if not s:
        return None
    
    # Remove non-numeric chars except decimal and minus
    s = s.strip()
    s = s.replace(',', '')  # Remove thousands separator
    s = s.replace('%', '')  # Remove percentage
    s = s.replace(' ', '')  # Remove spaces
    
    # Handle Indian numbering (lakhs, crores)
    lakh_match = re.search(r'([\d.]+)\s*lakh', s, re.IGNORECASE)
    crore_match = re.search(r'([\d.]+)\s*crore', s, re.IGNORECASE)
    
    if crore_match:
        return float(crore_match.group(1)) * 10_000_000
    if lakh_match:
        return float(lakh_match.group(1)) * 100_000
    
    # Standard number
    try:
        return float(s)
    except ValueError:
        return None


# ==========================================
# PART 3: ESG Metric Pattern Recognition
# ==========================================

ESG_PATTERNS = {
    # ===== ENVIRONMENTAL =====
    "scope1_tco2e": [
        r'scope[\s\-]*1[:\s]+(?:emissions?[:\s]*)?([\d,]+\.?\d*)\s*(?:tCO2e|MT|tonnes|tco2)',
        r'direct\s+(?:GHG\s+)?emissions?[:\s]*([\d,]+\.?\d*)\s*(?:tCO2e|MT)',
        r'scope[\s\-]*1[^\d]*([\d,]+\.?\d*)\s*(?:tCO2|MT)',
    ],
    "scope2_tco2e": [
        r'scope[\s\-]*2[:\s]+(?:emissions?[:\s]*)?([\d,]+\.?\d*)\s*(?:tCO2e|MT|tonnes|tco2)',
        r'indirect\s+(?:GHG\s+)?emissions?[:\s]*([\d,]+\.?\d*)\s*(?:tCO2e|MT)',
        r'scope[\s\-]*2[^\d]*([\d,]+\.?\d*)\s*(?:tCO2|MT)',
    ],
    "scope3_tco2e": [
        r'scope[\s\-]*3[:\s]+(?:emissions?[:\s]*)?([\d,]+\.?\d*)\s*(?:tCO2e|MT|tonnes|tco2)',
        r'other\s+indirect\s+emissions?[:\s]*([\d,]+\.?\d*)',
        r'scope[\s\-]*3[^\d]*([\d,]+\.?\d*)\s*(?:tCO2|MT)',
    ],
    "ghg_total_tco2e": [
        r'total\s+(?:GHG\s+)?emissions?[:\s]*([\d,]+\.?\d*)\s*(?:tCO2e|MT)',
        r'GHG\s+emissions?\s+total[:\s]*([\d,]+\.?\d*)',
    ],
    "energy_consumption_total_mwh": [
        r'total\s+energy\s+consumption[:\s]*([\d,]+\.?\d*)\s*(?:MWh|GJ|TJ)',
        r'energy\s+consumed[:\s]*([\d,]+\.?\d*)\s*(?:MWh|GJ)',
        r'energy\s+consumption[:\s]*([\d,]+\.?\d*)',
    ],
    "renewable_energy_percent": [
        r'renewable\s+energy[:\s]*([\d,]+\.?\d*)\s*%',
        r'([\d,]+\.?\d*)\s*%\s*(?:of\s+total\s+)?renewable',
        r'share\s+of\s+renewable[:\s]*([\d,]+\.?\d*)\s*%',
    ],
    "water_withdrawal_kl": [
        r'water\s+withdrawal[:\s]*([\d,]+\.?\d*)\s*(?:kL|ML|KL|kilolitre)',
        r'total\s+water\s+(?:consumed|used|withdrawn)[:\s]*([\d,]+\.?\d*)',
        r'water\s+consumption[:\s]*([\d,]+\.?\d*)\s*(?:kL|ML)',
    ],
    "water_recycled_percent": [
        r'water\s+recycled[:\s]*([\d,]+\.?\d*)\s*%',
        r'([\d,]+\.?\d*)\s*%\s+(?:of\s+)?water\s+recycled',
        r'recycled\s+water[:\s]*([\d,]+\.?\d*)\s*%',
    ],
    "waste_hazardous_generated_tonnes": [
        r'hazardous\s+waste[:\s]*(?:generated)?[:\s]*([\d,]+\.?\d*)\s*(?:MT|tonnes|kg)',
        r'total\s+hazardous[:\s]*([\d,]+\.?\d*)',
    ],
    "waste_non_hazardous_generated_tonnes": [
        r'non[\s\-]?hazardous\s+waste[:\s]*([\d,]+\.?\d*)\s*(?:MT|tonnes)',
    ],
    
    # ===== SOCIAL =====
    "total_employees": [
        r'total\s+employees?[:\s]*([\d,]+)(?!\s*%)',
        r'employee\s+(?:strength|headcount|count)[:\s]*([\d,]+)',
        r'total\s+(?:workforce|manpower)[:\s]*([\d,]+)',
        r'employees?\s+(?:and\s+workers?\s+)?total[:\s]*([\d,]+)',
    ],
    "permanent_employees": [
        r'permanent\s+(?:employees?|staff)[:\s\(D\)]*\s*([\d,]+)(?!\s*%)',
        r'total\s+permanent\s+employees?[:\s]*([\d,]+)',
    ],
    "contract_workers": [
        r'contract(?:ual)?\s+(?:workers?|employees?|staff)[:\s]*([\d,]+)',
        r'other\s+than\s+permanent[:\s]*([\d,]+)',
    ],
    "male_percent": [
        r'male[:\s]*([\d,]+\.?\d*)\s*%',
        r'([\d,]+\.?\d*)\s*%\s*male',
    ],
    "female_percent": [
        r'female[:\s]*([\d,]+\.?\d*)\s*%',
        r'([\d,]+\.?\d*)\s*%\s*female',
        r'women[:\s]*([\d,]+\.?\d*)\s*%',
    ],
    "women_in_board_percent": [
        r'women\s+(?:on\s+)?(?:the\s+)?board[:\s]*([\d,]+\.?\d*)\s*%',
        r'female\s+directors?[:\s]*([\d,]+\.?\d*)\s*%',
        r'([\d,]+\.?\d*)\s*%\s*women\s+(?:in|on)\s+board',
        r'board[^.]*women[:\s]*([\d,]+\.?\d*)\s*%',
    ],
    "attrition_rate_percent": [
        r'attrition\s+rate[:\s]*([\d,]+\.?\d*)\s*%',
        r'turnover\s+rate[:\s]*([\d,]+\.?\d*)\s*%',
        r'employee\s+turnover[:\s]*([\d,]+\.?\d*)\s*%',
    ],
    "training_hours_total": [
        r'total\s+training\s+hours?[:\s]*([\d,]+\.?\d*)',
        r'training\s+hours?\s+total[:\s]*([\d,]+\.?\d*)',
    ],
    "training_hours_per_employee": [
        r'training\s+hours?\s+per\s+employee[:\s]*([\d,]+\.?\d*)',
        r'average\s+training[:\s]*([\d,]+\.?\d*)\s*hours?',
        r'hours?\s+per\s+employee[:\s]*([\d,]+\.?\d*)',
    ],
    "ltifr_employees": [
        r'LTIFR[:\s]*([\d,]+\.?\d*)',
        r'lost\s+time\s+injury\s+frequency\s+rate[:\s]*([\d,]+\.?\d*)',
    ],
    "fatalities_employees_count": [
        r'(?:employee\s+)?fatalities?[:\s]*([\d]+)',
        r'deaths?[:\s]*([\d]+)',
    ],
    "csr_spend_million_inr": [
        r'CSR\s+(?:spend|expenditure)[:\s]*(?:INR\s*)?([\d,]+\.?\d*)\s*(?:million|crore|lakh)?',
        r'([\d,]+\.?\d*)\s*(?:million|crore|lakh)\s+(?:on\s+)?CSR',
        r'spent\s+on\s+CSR[:\s]*(?:INR\s*)?([\d,]+\.?\d*)',
    ],
    
    # ===== GOVERNANCE =====
    "board_size_count": [
        r'board\s+(?:size|strength)[:\s]*([\d]+)',
        r'([\d]+)\s+directors?\s+(?:on\s+)?(?:the\s+)?board',
        r'total\s+(?:board\s+)?directors?[:\s]*([\d]+)',
        r'board\s+of\s+directors[:\s]*([\d]+)',
    ],
    "independent_directors_count": [
        r'independent\s+directors?[:\s]*([\d]+)(?!\s*%)',
        r'([\d]+)\s+independent\s+directors?',
    ],
    "independent_directors_percent": [
        r'independent\s+directors?[:\s]*([\d,]+\.?\d*)\s*%',
        r'([\d,]+\.?\d*)\s*%\s*independent',
    ],
    "women_directors_count": [
        r'women\s+directors?[:\s]*([\d]+)(?!\s*%)',
        r'female\s+directors?[:\s]*([\d]+)(?!\s*%)',
    ],
    "women_directors_percent": [
        r'women\s+directors?[:\s]*([\d,]+\.?\d*)\s*%',
        r'female\s+directors?[:\s]*([\d,]+\.?\d*)\s*%',
    ],
    "board_meetings_count": [
        r'board\s+meetings?[:\s]*([\d]+)',
        r'([\d]+)\s+board\s+meetings?',
        r'meetings?\s+of\s+(?:the\s+)?board[:\s]*([\d]+)',
    ],
    "whistleblower_complaints_filed": [
        r'whistle[\s\-]?blower\s+complaints?[:\s]*(?:filed[:\s]*)?([\d]+)',
        r'complaints?\s+(?:under\s+)?whistle[\s\-]?blower[:\s]*([\d]+)',
    ],
}


def extract_metrics_regex(text: str) -> Dict:
    """
    Use regex patterns to pre-extract ESG metrics from text.
    Returns dict of {metric_name: extracted_value}.
    """
    results = {}
    text_lower = text.lower()
    
    # ========== ENVIRONMENTAL METRICS ==========
    
    # Scope 1 emissions: "Total Scope 1 emissions...equivalent3,39,941" 
    # Note: Number follows "equivalent" directly in BRSR format
    scope1_match = re.search(
        r'total\s+scope\s*1\s+emissions.*?equivalent\s*([\d,]+)',
        text, re.IGNORECASE | re.DOTALL
    )
    if scope1_match:
        # Handle Indian number format (3,39,941 = 339941)
        val = scope1_match.group(1).replace(',', '')
        results["scope1_tco2e"] = float(val)
    
    # Scope 2 emissions: "Total Scope 2 emissions...equivalent3,57,810"
    scope2_match = re.search(
        r'total\s+scope\s*2\s+emissions.*?equivalent\s*([\d,]+)',
        text, re.IGNORECASE | re.DOTALL
    )
    if scope2_match:
        val = scope2_match.group(1).replace(',', '')
        results["scope2_tco2e"] = float(val)
    
    # Total energy consumption
    energy_match = re.search(
        r'total\s+energy\s+consumption.*?([\d,]+\.?\d*)\s*(?:GJ|MWh|TJ)',
        text, re.IGNORECASE
    )
    if energy_match:
        results["energy_consumption_total_mwh"] = parse_number(energy_match.group(1))
    
    # Water withdrawal
    water_match = re.search(
        r'(?:total\s+)?water\s+(?:withdrawal|withdrawn).*?([\d,]+\.?\d*)\s*(?:kL|ML|KL)',
        text, re.IGNORECASE
    )
    if water_match:
        results["water_withdrawal_kl"] = parse_number(water_match.group(1))
    
    # ========== SOCIAL METRICS (BRSR TABLE FORMAT) ==========
    
    # Pattern: "1. Permanent (D) 10,485" - first number is the total
    permanent_emp_match = re.search(
        r'permanent\s*\(D\)\s*([\d,]+)\s+[\d,]+\s+[\d.]+\s+[\d,]+\s+[\d.]+',
        text, re.IGNORECASE
    )
    if permanent_emp_match:
        results["permanent_employees"] = parse_number(permanent_emp_match.group(1))
    
    # Total employees: "Total employees (D + E) 11,276"
    total_emp_match = re.search(
        r'total\s+employees?\s*\([^)]*\)\s*([\d,]+)',
        text, re.IGNORECASE
    )
    if total_emp_match:
        results["total_employees"] = parse_number(total_emp_match.group(1))
    
    # Board of Directors table: "Board of Directors 10 2 20.00"
    # Format: label total_count women_count women_percent
    board_match = re.search(
        r'board\s+of\s+directors\s+([\d]+)\s+([\d]+)\s+([\d.]+)',
        text, re.IGNORECASE
    )
    if board_match:
        results["board_size_count"] = parse_number(board_match.group(1))
        results["women_directors_count"] = parse_number(board_match.group(2))
        results["women_directors_percent"] = parse_number(board_match.group(3))
        results["women_in_board_percent"] = parse_number(board_match.group(3))
    
    # Turnover rate: "Permanent employees 7.37% 13.28% 8.26%"
    turnover_match = re.search(
        r'permanent\s+employees\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%',
        text, re.IGNORECASE
    )
    if turnover_match:
        results["attrition_rate_percent"] = parse_number(turnover_match.group(3))  # Total rate
    
    # ========== GOVERNANCE METRICS ==========
    
    # Board meetings: look for explicit count
    board_meetings_match = re.search(
        r'(?:number\s+of\s+)?board\s+meetings?\s*(?:held)?\s*[:\s]*([\d]+)',
        text, re.IGNORECASE
    )
    if board_meetings_match:
        results["board_meetings_count"] = parse_number(board_meetings_match.group(1))
    
    # Independent directors count from table or text
    indep_match = re.search(
        r'independent\s+directors?\s*[:\s]*([\d]+)(?:\s+out\s+of|\s+directors)?',
        text, re.IGNORECASE
    )
    if indep_match:
        results["independent_directors_count"] = parse_number(indep_match.group(1))
    
    # Standard pattern matching for remaining fields
    for metric, patterns in ESG_PATTERNS.items():
        if metric in results:  # Already found via special handling
            continue
        for pattern in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                value = parse_number(match.group(1))
                if value is not None and value > 0:  # Skip zero/null values
                    results[metric] = value
                    break
    
    return results


# ==========================================
# PART 4: Section Detection
# ==========================================

SECTION_MARKERS = {
    "Environment": [
        r'SECTION\s*[BC].*ENVIRONMENT',
        r'PRINCIPLE\s*6.*ENVIRONMENT',
        r'ENVIRONMENTAL\s+PERFORMANCE',
        r'GHG\s+EMISSIONS',
        r'ENERGY\s+MANAGEMENT',
        r'WATER\s+(?:MANAGEMENT|CONSUMPTION)',
    ],
    "Social": [
        r'SECTION\s*[BC].*SOCIAL',
        r'PRINCIPLE\s*[345].*EMPLOYEE',
        r'SOCIAL\s+PERFORMANCE',
        r'WORKFORCE\s+(?:DETAILS|DISCLOSURE)',
        r'EMPLOYEE\s+(?:WELL-?BEING|WELFARE)',
        r'HEALTH\s+AND\s+SAFETY',
    ],
    "Governance": [
        r'SECTION\s*A.*DISCLOSURE',
        r'CORPORATE\s+GOVERNANCE',
        r'BOARD\s+(?:OF\s+)?DIRECTORS',
        r'GOVERNANCE\s+STRUCTURE',
        r'PRINCIPLE\s*1.*ETHICS',
    ],
}


def detect_sections(text: str) -> Dict[str, str]:
    """
    Detect and extract ESG sections from text.
    Returns dict of {section_name: section_text}.
    """
    sections = {"Environment": "", "Social": "", "Governance": "", "General": ""}
    
    lines = text.split('\n')
    current_section = "General"
    section_buffer = []
    
    for line in lines:
        line_upper = line.upper().strip()
        
        # Check if line marks a new section
        new_section = None
        for section, patterns in SECTION_MARKERS.items():
            for pattern in patterns:
                if re.search(pattern, line_upper):
                    new_section = section
                    break
            if new_section:
                break
        
        if new_section and new_section != current_section:
            # Save current buffer to current section
            sections[current_section] += '\n'.join(section_buffer) + '\n'
            section_buffer = []
            current_section = new_section
        
        section_buffer.append(line)
    
    # Save final buffer
    sections[current_section] += '\n'.join(section_buffer)
    
    return sections


# ==========================================
# PART 5: Main Enhanced Preprocessing Pipeline
# ==========================================

def enhanced_clean_brsr(raw_text: str) -> str:
    """
    Enhanced BRSR text cleaning with better character recognition.
    """
    text = raw_text
    
    # Step 1: Unicode normalization
    text = normalize_unicode(text)
    
    # Step 2: Fix OCR errors
    text = fix_ocr_errors(text)
    
    # Step 3: Remove headers/footers
    # Annual Report footer patterns
    text = re.sub(r'\d+(?:st|nd|rd|th)\s+Annual\s+Report\s+\d{4}-\d{2,4}', '', text, flags=re.IGNORECASE)
    
    # Page numbers (various formats)
    text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', text)
    text = re.sub(r'Page\s*\d+\s*(?:of\s*\d+)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\|\s*\d+\s*\|', '', text)
    
    # Step 4: Preserve table structure
    # Convert multiple spaces in a row to tab (likely table columns)
    text = re.sub(r'(?<=[a-zA-Z])\s{3,}(?=[\d])', '\t', text)
    
    # Step 5: Normalize whitespace (but preserve newlines)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Step 6: Remove noise
    text = re.sub(r'[_\-]{10,}', '', text)  # Long dashes/underscores
    text = re.sub(r'\.{5,}', ' ', text)  # Dot leaders
    
    # Step 7: Trim lines
    text = '\n'.join(line.strip() for line in text.splitlines())
    text = text.strip()
    
    return text


def preprocess_for_extraction(raw_text: str) -> Dict:
    """
    Full preprocessing pipeline for ESG extraction.
    Returns dict with:
        - cleaned_text: Cleaned full text
        - sections: Dict of section texts
        - pre_extracted: Pre-extracted metrics from regex
        - tables: Detected table data
    """
    # Clean text
    cleaned = enhanced_clean_brsr(raw_text)
    
    # Detect sections
    sections = detect_sections(cleaned)
    
    # Extract tables
    _, tables = extract_tables(cleaned)
    
    # Pre-extract metrics using regex
    pre_extracted = extract_metrics_regex(cleaned)
    
    # Also extract from sections
    for section_name, section_text in sections.items():
        section_metrics = extract_metrics_regex(section_text)
        # Section-specific metrics have priority
        pre_extracted.update(section_metrics)
    
    return {
        "cleaned_text": cleaned,
        "sections": sections,
        "pre_extracted": pre_extracted,
        "tables": tables,
        "stats": {
            "original_length": len(raw_text),
            "cleaned_length": len(cleaned),
            "sections_found": [k for k, v in sections.items() if v.strip()],
            "metrics_pre_extracted": len(pre_extracted),
            "tables_detected": len(tables),
        }
    }


def process_file(raw_path: str, cleaned_folder: str = "cleaned_input_txt") -> str:
    """
    Process a raw ESG file with enhanced preprocessing.
    """
    os.makedirs(cleaned_folder, exist_ok=True)
    
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    result = preprocess_for_extraction(raw_text)
    
    # Save cleaned text
    company = os.path.splitext(os.path.basename(raw_path))[0]
    cleaned_path = os.path.join(cleaned_folder, f"{company}.txt")
    
    with open(cleaned_path, "w", encoding="utf-8") as f:
        f.write(result["cleaned_text"])
    
    # Save pre-extracted data
    preextract_path = os.path.join(cleaned_folder, f"{company}_preextract.json")
    import json
    with open(preextract_path, "w", encoding="utf-8") as f:
        json.dump({
            "pre_extracted": result["pre_extracted"],
            "stats": result["stats"]
        }, f, indent=2)
    
    print(f"✨ Enhanced preprocessing complete for {company}")
    print(f"   📄 Text: {result['stats']['original_length']:,} -> {result['stats']['cleaned_length']:,} chars")
    print(f"   📊 Pre-extracted {result['stats']['metrics_pre_extracted']} metrics via regex")
    print(f"   📋 Detected {result['stats']['tables_detected']} table entries")
    
    return cleaned_path


# ==========================================
# CLI Entry Point
# ==========================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        process_file(sys.argv[1])
    else:
        # Process all files in raw_input_txt
        raw_folder = "raw_input_txt"
        if os.path.exists(raw_folder):
            for fname in os.listdir(raw_folder):
                if fname.endswith(".txt"):
                    process_file(os.path.join(raw_folder, fname))
