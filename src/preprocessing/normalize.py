"""
Core normalization functions for business entity resolution.

Provides text cleaning, business name normalization, and address normalization.
Each function is designed to be composable and testable independently.
The main entry point is `preprocess_dataframe()` which applies the full pipeline.
"""

import re
import unicodedata
from typing import Optional

from src.preprocessing.constants import (
    LEGAL_SUFFIXES_TO_STRIP,
    ADDRESS_ABBREV_MAP,
    US_STATE_NAME_TO_ABBREV,
    NAME_NOISE_PATTERNS,
    ADDRESS_NOISE_PATTERNS,
    NAME_STOPWORDS,
)


# ══════════════════════════════════════════════════════════════════════
# GENERIC TEXT CLEANING
# ══════════════════════════════════════════════════════════════════════

def clean_text(text: Optional[str]) -> str:
    """
    Basic text cleaning applied to ALL fields.
    - Handle NaN/None → empty string
    - Lowercase
    - Normalize unicode (NFD → NFC), remove combining marks for comparison
    - Collapse whitespace
    """
    if text is None or (isinstance(text, float)):
        return ""

    text = str(text).strip().lower()

    # Normalize unicode — strip accents from Latin text only.
    # Devanagari/Kannada use combining marks (virama, vowel signs) that
    # NFKD decomposition breaks apart. So we process char-by-char:
    # only decompose and strip marks for Latin-range characters.
    text = unicodedata.normalize("NFC", text)
    result = []
    for c in text:
        if ord(c) < 0x0080:
            # Plain ASCII — keep as-is
            result.append(c)
        elif ord(c) < 0x0250:
            # Latin Extended range — decompose and strip accents
            decomposed = unicodedata.normalize("NFKD", c)
            result.append("".join(
                ch for ch in decomposed if not unicodedata.combining(ch)
            ))
        else:
            # Non-Latin (Devanagari, Kannada, etc.) — keep as-is
            result.append(c)
    text = "".join(result)

    # Replace common special chars
    text = text.replace("&", " and ")
    text = text.replace("@", " at ")
    text = text.replace("+", " plus ")

    # Remove non-alphanumeric chars (keep spaces, hyphens, periods, commas, slashes)
    # Also keep Unicode combining/spacing marks (Mn, Mc) — these are essential
    # for Devanagari, Kannada, and other Indic scripts (virama, vowel signs, anusvara).
    text = re.sub(r"[^\w\s\-\.,/\u0300-\u036f\u0900-\u097f\u0980-\u09ff\u0a00-\u0d7f\u0c80-\u0cff]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_text_keep_accents(text: Optional[str]) -> str:
    """
    Lighter cleaning that preserves accents — used for the embedding input
    where we want the model to see the original characters (especially French).
    """
    if text is None or (isinstance(text, float)):
        return ""

    text = str(text).strip().lower()
    text = unicodedata.normalize("NFC", text)
    text = text.replace("&", " and ")

    # Only remove extreme noise
    for pattern in NAME_NOISE_PATTERNS:
        text = text.replace(pattern, " ")

    text = re.sub(r"\s+", " ", text).strip()
    return text


# ══════════════════════════════════════════════════════════════════════
# BUSINESS NAME NORMALIZATION
# ══════════════════════════════════════════════════════════════════════

def remove_name_noise(name: str) -> str:
    """Remove known noise patterns from business names."""
    for pattern in NAME_NOISE_PATTERNS:
        name = name.replace(pattern, " ")

    # Remove URL-like patterns (e.g., "www.shivshakti.com")
    name = re.sub(r"https?://\S+", "", name)
    name = re.sub(r"www\.\S+", "", name)
    name = re.sub(r"\S+\.(com|org|net|in|co)\b", "", name)

    name = re.sub(r"\s+", " ", name).strip()
    return name


def strip_legal_suffixes(name: str) -> str:
    """
    Remove legal suffixes (Inc, LLC, Ltd, Pvt, etc.) from the business name.
    These add noise to matching — "Strategic Praetorian" should match
    "Strategic Praetorian Inc".
    """
    tokens = name.split()
    cleaned = []
    for token in tokens:
        # Check against the set of suffixes to strip
        if token.strip(".,") not in LEGAL_SUFFIXES_TO_STRIP:
            cleaned.append(token)

    return " ".join(cleaned).strip()


def normalize_business_name(name: Optional[str]) -> str:
    """
    Full business name normalization pipeline.
    Returns a cleaned, suffix-stripped, noise-removed name.
    """
    name = clean_text(name)
    if not name:
        return ""

    name = remove_name_noise(name)
    name = strip_legal_suffixes(name)

    # Remove remaining punctuation (periods, commas) for token matching
    name = re.sub(r"[.,\-/']", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


def get_name_tokens(name_clean: str) -> str:
    """
    Get sorted, deduplicated, stopword-filtered tokens from a cleaned name.
    Useful for token-overlap blocking and Jaccard similarity.

    "moore bitwise" → "bitwise moore"  (order-independent)
    """
    if not name_clean:
        return ""

    tokens = name_clean.split()
    tokens = [t for t in tokens if t not in NAME_STOPWORDS and len(t) > 1]
    tokens = sorted(set(tokens))
    return " ".join(tokens)


# ══════════════════════════════════════════════════════════════════════
# ADDRESS NORMALIZATION
# ══════════════════════════════════════════════════════════════════════

def remove_address_noise(addr: str) -> str:
    """Remove known noise patterns from addresses."""
    # Remove "##" prefix on street numbers
    addr = re.sub(r"##(\d)", r"\1", addr)

    # Remove "1/2" from street numbers (e.g., "19 1/2 Stardust Trail")
    addr = re.sub(r"(\d+)\s*1/2", r"\1", addr)

    # Remove landmark references (very noisy, not useful for matching)
    addr = re.sub(r"\bnear\s+\w+(\s+\w+){0,4}", "", addr, flags=re.IGNORECASE)
    addr = re.sub(r"\bopp\.?\s+\w+(\s+\w+){0,3}", "", addr, flags=re.IGNORECASE)
    addr = re.sub(r"\bbehind\s+\w+(\s+\w+){0,3}", "", addr, flags=re.IGNORECASE)
    addr = re.sub(r"\bbeside\s+\w+(\s+\w+){0,3}", "", addr, flags=re.IGNORECASE)

    # Remove "CDP" suffix (Census Designated Place)
    addr = re.sub(r"\bcdp\b", "", addr)

    # Remove "City of" prefix
    addr = re.sub(r"\bcity of\b", "", addr)

    addr = re.sub(r"\s+", " ", addr).strip()
    return addr


def expand_address_abbreviations(addr: str) -> str:
    """Expand common address abbreviations (Rd → Road, St → Street, etc.)."""
    tokens = addr.split()
    expanded = []
    for token in tokens:
        lookup_key = token.strip(".,")
        if lookup_key in ADDRESS_ABBREV_MAP:
            expanded.append(ADDRESS_ABBREV_MAP[lookup_key])
        else:
            expanded.append(token)
    return " ".join(expanded)


def extract_street_number(addr: str) -> str:
    """Extract the primary street number from an address."""
    # Match leading number or number after common prefixes
    match = re.search(r"\b(\d+)\b", addr)
    return match.group(1) if match else ""


def extract_postal_code(addr: str, country: str) -> str:
    """
    Extract postal/ZIP/PIN code from address.
    - US: 5-digit ZIP (optionally +4)
    - India: 6-digit PIN
    - France: 5-digit postal code
    """
    country_lower = country.lower() if country else ""

    if country_lower == "us":
        # US ZIP: 5 digits, optionally followed by -4 digits
        match = re.search(r"\b(\d{5})(?:-\d{4})?\b", addr)
        return match.group(1) if match else ""
    elif country_lower == "india":
        # Indian PIN: 6 digits
        match = re.search(r"\b(\d{6})\b", addr)
        return match.group(1) if match else ""
    elif country_lower == "france":
        # French postal code: 5 digits
        match = re.search(r"\b(\d{5})\b", addr)
        return match.group(1) if match else ""
    else:
        # Generic: try 5-6 digit codes
        match = re.search(r"\b(\d{5,6})\b", addr)
        return match.group(1) if match else ""


def extract_state_or_region(addr: str, country: str) -> str:
    """
    Extract state (US/India) or region (France) from address.
    Returns lowercase standardized form.
    """
    country_lower = country.lower() if country else ""
    addr_lower = addr.lower()

    if country_lower == "us":
        # Check for full state names first
        for state_name, abbrev in US_STATE_NAME_TO_ABBREV.items():
            if state_name in addr_lower:
                return abbrev
        # Check for 2-letter state abbreviations at word boundaries
        match = re.search(r"\b([A-Z]{2})\b", addr)
        if match:
            abbrev = match.group(1).lower()
            # Verify it's a real state abbreviation
            from src.preprocessing.constants import US_STATE_ABBREV
            if match.group(1) in US_STATE_ABBREV:
                return abbrev
    elif country_lower == "india":
        from src.preprocessing.constants import INDIAN_STATES
        for state in INDIAN_STATES:
            if state in addr_lower:
                return state
    elif country_lower == "france":
        from src.preprocessing.constants import FRENCH_REGIONS
        for region in FRENCH_REGIONS:
            if region in addr_lower:
                return region

    return ""


def normalize_address(addr: Optional[str], country: Optional[str] = "") -> str:
    """
    Full address normalization pipeline.
    Returns a cleaned, expanded, noise-removed address.
    """
    addr = clean_text(addr)
    if not addr:
        return ""

    addr = remove_address_noise(addr)
    addr = expand_address_abbreviations(addr)

    # Remove "unit", "po box", "apt" info markers (keep the actual number)
    addr = re.sub(r"\b(unit|po box|p\.?o\.? box)\s*", "", addr)

    # Remove remaining punctuation for cleaner token matching
    addr = re.sub(r"[.,#\-/']", " ", addr)
    addr = re.sub(r"\s+", " ", addr).strip()

    return addr


# ══════════════════════════════════════════════════════════════════════
# DATAFRAME-LEVEL PREPROCESSING
# ══════════════════════════════════════════════════════════════════════

def preprocess_dataframe(df):
    """
    Apply the full preprocessing pipeline to a source dataframe.

    Input columns:  entity_id, business_name, business_address, country
    Output columns: all original + name_clean, name_tokens, addr_clean,
                    street_number, postal_code, state_region, combined_text,
                    combined_text_for_embedding

    Parameters
    ----------
    df : pd.DataFrame
        Raw source dataframe with columns: entity_id, business_name,
        business_address, country

    Returns
    -------
    pd.DataFrame
        The dataframe with additional normalized columns.
    """
    import pandas as pd

    print(f"  Preprocessing {len(df):,} records...")

    # Fill NaN values
    df["business_name"] = df["business_name"].fillna("")
    df["business_address"] = df["business_address"].fillna("")
    df["country"] = df["country"].fillna("")

    # ── Business name normalization ──
    # Using list comprehensions instead of .apply() — avoids per-row
    # Series construction overhead, ~3-5× faster on millions of rows.
    print("    → Normalizing business names...")
    df["name_clean"] = [normalize_business_name(x) for x in df["business_name"]]
    df["name_tokens"] = [get_name_tokens(x) for x in df["name_clean"]]

    # ── Address normalization ──
    # zip-based comprehension replaces df.apply(axis=1) which created a
    # full Series per row — the single biggest bottleneck.
    print("    → Normalizing addresses...")
    df["addr_clean"] = [
        normalize_address(a, c)
        for a, c in zip(df["business_address"], df["country"])
    ]

    # ── Extract structured address components ──
    print("    → Extracting address components...")
    df["street_number"] = [extract_street_number(x) for x in df["addr_clean"]]
    df["postal_code"] = [
        extract_postal_code(str(a), str(c))  # Use original (may have ZIP in specific format)
        for a, c in zip(df["business_address"], df["country"])
    ]
    df["state_region"] = [
        extract_state_or_region(str(a), str(c))  # Use original for state detection
        for a, c in zip(df["business_address"], df["country"])
    ]

    # ── Combined text fields ──
    # For TF-IDF / char n-gram blocking (fully normalized)
    df["combined_text"] = df["name_clean"] + " " + df["addr_clean"]

    # For sentence embeddings (lighter cleaning, preserves accents/meaning)
    names_emb = [clean_text_keep_accents(x) for x in df["business_name"]]
    addrs_emb = [clean_text_keep_accents(x) for x in df["business_address"]]
    countries = [str(x).lower() for x in df["country"]]
    df["combined_text_for_embedding"] = [
        f"{n} | {a} | {c}" for n, a, c in zip(names_emb, addrs_emb, countries)
    ]

    print(f"  ✓ Done. Output columns: {df.columns.tolist()}")
    return df
