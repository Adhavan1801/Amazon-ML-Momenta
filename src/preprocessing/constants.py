"""
Constants for text normalization.
Abbreviation maps, legal suffixes, address standardization, stopwords.

Derived from manual inspection of matched pairs in the training data.
Noise patterns observed:
  - Name: "Corp" vs "Corporation", "Pvt" vs "Private", word reordering,
          typos ("Edterptises" for "Enterprises"), ".com" suffixes,
          "dba" prefixes, "[[LLC]]" bracket noise, "..." prefixes
  - Address: "Rd" vs "Road", "St" vs "Street", UPPER vs Title case,
             "##" prefix on street numbers, "1/2" in numbers,
             state abbreviations vs full names, "CDP" / "City" suffixes,
             landmark references ("Near Fortis Hospital")
"""

# ──────────────────────────────────────────────────────────────────────
# BUSINESS NAME — Legal suffix standardization
# ──────────────────────────────────────────────────────────────────────

# Map abbreviations → canonical form (all lowercase)
LEGAL_SUFFIX_MAP = {
    # English
    "corp": "corporation",
    "corp.": "corporation",
    "inc": "incorporated",
    "inc.": "incorporated",
    "ltd": "limited",
    "ltd.": "limited",
    "llc": "limited liability company",
    "l.l.c.": "limited liability company",
    "l.l.c": "limited liability company",
    "llp": "limited liability partnership",
    "l.l.p.": "limited liability partnership",
    "pvt": "private",
    "pvt.": "private",
    "co": "company",
    "co.": "company",
    "assoc": "associates",
    "assoc.": "associates",
    "intl": "international",
    "intl.": "international",
    "svc": "services",
    "svcs": "services",
    "svc.": "services",
    "mfg": "manufacturing",
    "mfg.": "manufacturing",
    "grp": "group",
    "grp.": "group",
    "tech": "technology",
    "sys": "systems",
    "sys.": "systems",
    "natl": "national",
    "natl.": "national",
    "dept": "department",
    "dept.": "department",
    "mgmt": "management",
    "mgmt.": "management",
    "dist": "distributors",
    "dist.": "distributors",
    "ent": "enterprises",
    "ent.": "enterprises",
    "hldg": "holdings",
    "hldg.": "holdings",
    "hldgs": "holdings",
    # French
    "sarl": "sarl",
    "sas": "sas",
    "sasu": "sasu",
    "eurl": "eurl",
    "sa": "sa",
    "sci": "sci",
    "ste": "societe",
    # Indian
    "pvt.ltd.": "private limited",
    "pvt.ltd": "private limited",
}

# Suffixes to REMOVE entirely for comparison (they add noise, not signal)
LEGAL_SUFFIXES_TO_STRIP = {
    "inc", "inc.", "incorporated",
    "llc", "l.l.c.", "l.l.c",
    "ltd", "ltd.", "limited",
    "llp", "l.l.p.", "l.l.p",
    "corp", "corp.", "corporation",
    "co", "co.", "company",
    "pvt", "pvt.",  "private",
    "pvt.ltd.", "pvt.ltd",
    "sarl", "sas", "sasu", "eurl", "sa", "sci",
    "societe",
}

# ──────────────────────────────────────────────────────────────────────
# ADDRESS — Abbreviation standardization
# ──────────────────────────────────────────────────────────────────────

ADDRESS_ABBREV_MAP = {
    # Street types
    "rd": "road", "rd.": "road",
    "st": "street", "st.": "street",
    "ave": "avenue", "ave.": "avenue",
    "dr": "drive", "dr.": "drive",
    "blvd": "boulevard", "blvd.": "boulevard",
    "ln": "lane", "ln.": "lane",
    "ct": "court", "ct.": "court",
    "pl": "place", "pl.": "place",
    "hwy": "highway", "hwy.": "highway",
    "pkwy": "parkway", "pkwy.": "parkway",
    "cir": "circle", "cir.": "circle",
    "ter": "terrace", "ter.": "terrace",
    "trl": "trail", "trl.": "trail",
    "sq": "square", "sq.": "square",
    "expy": "expressway",
    "fwy": "freeway",
    # Unit types
    "apt": "apartment", "apt.": "apartment",
    "ste": "suite", "ste.": "suite",
    "fl": "floor", "fl.": "floor",
    "rm": "room", "rm.": "room",
    "bldg": "building", "bldg.": "building",
    # Directions
    "n": "north", "n.": "north",
    "s": "south", "s.": "south",
    "e": "east", "e.": "east",
    "w": "west", "w.": "west",
    "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest",
    # French address types
    "bd": "boulevard",
    "av": "avenue",
    "imp": "impasse",
    "ch": "chemin",
}

# US State abbreviation → full name (for normalization)
US_STATE_ABBREV = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}

# Reverse map: full name → abbreviation (lowercase keys for matching)
US_STATE_NAME_TO_ABBREV = {v.lower(): k.lower() for k, v in US_STATE_ABBREV.items()}

# Indian states — for recognition (not exhaustive, covers major ones)
INDIAN_STATES = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
    "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram",
    "nagaland", "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu",
    "telangana", "tripura", "uttar pradesh", "uttarakhand", "west bengal",
    "delhi", "new delhi", "chandigarh", "puducherry", "jammu and kashmir",
    "ladakh",
}

# French regions
FRENCH_REGIONS = {
    "île-de-france", "ile-de-france", "nouvelle-aquitaine", "occitanie",
    "hauts-de-france", "auvergne-rhône-alpes", "auvergne-rhone-alpes",
    "grand est", "provence-alpes-côte d'azur", "provence-alpes-cote d'azur",
    "pays de la loire", "bretagne", "normandie", "bourgogne-franche-comté",
    "bourgogne-franche-comte", "centre-val de loire", "corse",
}

# Noise characters/patterns to strip from business names
NAME_NOISE_PATTERNS = [
    "[[", "]]",       # bracket noise: "[[LLC]]"
    "<<", ">>",       # angle bracket noise
    "...",             # prefix dots: "... Optimal Data"
    "--",              # prefix dashes: "-- Holloway Peak"
    "|",               # pipe separators (often followed by URLs)
    "www.",            # URL fragments in names
    ".com",            # domain suffixes used as names
    ".org",
    ".net",
    ".in",
    "dba ",            # "doing business as" prefix
    "d/b/a ",
]

# Address noise patterns
ADDRESS_NOISE_PATTERNS = [
    "##",              # "##3771 DEER VALLEY DRIVE"
    "near ",           # landmark references: "Near Fortis Hospital"
    "opp ",            # "Opposite ..."
    "behind ",         # "Behind ..."
    "beside ",
    "cdp",             # "FRANKLIN CDP" → just "FRANKLIN"
    "city of ",        # "CITY OF MENOMONIE" → "MENOMONIE"
]

# Tokens to remove from business names (stopwords for matching)
NAME_STOPWORDS = {
    "the", "a", "an", "of", "and", "&", "for", "in", "at", "by", "to",
    "on", "with", "from",
}
