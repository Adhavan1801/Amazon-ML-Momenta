"""Record normalisation (names + addresses), country-agnostic.

* Indic scripts -> Latin via a token dictionary learned from the TRAIN pairs
  (see learn_dict.py), falling back to rule-based transliteration (MIT lib).
* Accent folding with the standard library.
* Abbreviation expansion, legal-form canonicalisation, junk stripping,
  number canonicalisation, phonetic "skeleton" keys.
All dictionaries are hand-written general language knowledge.
"""
import re
import unicodedata
from functools import lru_cache

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# ------------------------------------------------------------------ scripts
_SCRIPTS = [(0x0900, 0x097F, sanscript.DEVANAGARI), (0x0980, 0x09FF, sanscript.BENGALI),
            (0x0A00, 0x0A7F, sanscript.GURMUKHI), (0x0A80, 0x0AFF, sanscript.GUJARATI),
            (0x0B00, 0x0B7F, sanscript.ORIYA), (0x0B80, 0x0BFF, sanscript.TAMIL),
            (0x0C00, 0x0C7F, sanscript.TELUGU), (0x0C80, 0x0CFF, sanscript.KANNADA),
            (0x0D00, 0x0D7F, sanscript.MALAYALAM)]
_INDIC_RE = re.compile(r"[ऀ-ൿ]")
_TAMIL_FIX = str.maketrans({"ன": "n", "ழ": "zh", "ற": "r", "ள": "l", "ஃ": ""})

INDIC_TOKEN_DICT: dict = {}     # filled by load_dicts()
INDIC_SEGMENT_DICT: dict = {}


def load_dicts(path):
    import json
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    INDIC_TOKEN_DICT.clear(); INDIC_TOKEN_DICT.update(d.get("token", {}))
    INDIC_SEGMENT_DICT.clear(); INDIC_SEGMENT_DICT.update(d.get("segment", {}))
    _translit_token.cache_clear()


def has_indic(s: str) -> bool:
    return bool(_INDIC_RE.search(s))


@lru_cache(maxsize=500_000)
def _translit_token(tok: str) -> str:
    if tok in INDIC_TOKEN_DICT:
        return INDIC_TOKEN_DICT[tok]
    if not _INDIC_RE.search(tok):
        return tok
    cp = next(ord(c) for c in tok if 0x0900 <= ord(c) <= 0x0D7F)
    sc = next(s for lo, hi, s in _SCRIPTS if lo <= cp <= hi)
    out = transliterate(tok.translate(_TAMIL_FIX) if sc == sanscript.TAMIL else tok, sc, sanscript.ITRANS)
    out = out.replace("M", "n").replace("~N", "n").replace("~n", "n").replace(".N", "n")
    out = out.lower()
    out = re.sub(r"[^a-z0-9]", "", strip_accents(out))
    # drop inherent schwa at word end (rama -> ram), keep short words
    if len(out) > 3 and out.endswith("a") and not out.endswith("aa"):
        out = out[:-1]
    return out


def indic_to_latin(s: str) -> str:
    return " ".join(_translit_token(t.strip(".,;:()[]-")) for t in s.split())


def strip_accents(t: str) -> str:
    t = (t.replace("ß", "ss").replace("œ", "oe").replace("Œ", "OE")
          .replace("æ", "ae").replace("Æ", "AE").replace("ø", "o").replace("Ø", "O")
          .replace("ł", "l").replace("đ", "d"))
    return "".join(ch for ch in unicodedata.normalize("NFKD", t) if not unicodedata.combining(ch))


# ------------------------------------------------------------------ vocab
NAME_ABBR = {
    "pvt": "private", "prvt": "private", "pvtltd": "private limited", "pte": "private",
    "ltd": "limited", "lmtd": "limited", "ltda": "limited", "ld": "limited",
    "corp": "corporation", "corpn": "corporation", "co": "company", "cos": "companies",
    "inc": "incorporated", "incorp": "incorporated", "intl": "international",
    "natl": "national", "mfg": "manufacturing", "mfrs": "manufacturers",
    "svcs": "services", "svc": "services", "tech": "technologies", "sys": "systems",
    "ent": "enterprises", "enterprise": "enterprises", "bros": "brothers",
    "assoc": "associates", "grp": "group", "hldgs": "holdings", "inds": "industries",
    "industry": "industries", "mktg": "marketing", "trdg": "trading", "engg": "engineering",
    "pharma": "pharmaceuticals", "mgmt": "management", "consultancy": "consultants",
    "cie": "compagnie", "ste": "societe", "sté": "societe", "ets": "etablissements",
    "st": "saint", "n": "and", "et": "and", "shree": "sri", "shri": "sri", "sree": "sri",
    "llc": "llc", "l": "l",
}
# canonical legal classes
LEGAL_CANON = {
    "private": "pvt", "limited": "ltd", "llp": "llp", "llc": "llc", "lp": "lp",
    "incorporated": "inc", "corporation": "corp", "company": "co", "plc": "plc", "pllc": "pllc",
    "pc": "pc", "opc": "opc", "gmbh": "gmbh", "sarl": "sarl", "sas": "sas", "sasu": "sasu",
    "sa": "sa", "eurl": "eurl", "sci": "sci", "snc": "snc", "scop": "scop", "selarl": "selarl",
    "earl": "earl", "gie": "gie", "eirl": "eirl", "huf": "huf", "esq": "esq",
}
NAME_STOP = {"the", "and", "of", "m", "s", "ms", "messrs", "a", "an", "de", "du", "des",
             "la", "le", "les", "l", "d", "dba", "aka", "smt", "mr", "mrs", "null", "none"}
_DOMAIN_RAW = re.compile(r"^\W*(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:com|net|org|co\.in|in|fr|biz|us|info|co|io)\W*$")
DOMAIN_RE = re.compile(r"^(?:https?\s*)?(?:www\s+)?([a-z0-9\s]+?)\s+(com|net|org|in|co\s+in|fr|biz|us|info|co|io)$")

ADDR_ABBR = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue", "avn": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "bld": "boulevard", "boul": "boulevard",
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place", "pkwy": "parkway", "hwy": "highway",
    "fwy": "freeway", "sq": "square", "ter": "terrace", "terr": "terrace", "cir": "circle",
    "trl": "trail", "ste": "suite", "fl": "floor", "flr": "floor", "apt": "apartment",
    "bldg": "building", "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest", "mt": "mount",
    "nr": "near", "opp": "opposite", "bhd": "behind", "ngr": "nagar", "clny": "colony",
    "col": "colony", "extn": "extension", "ext": "extension", "sec": "sector", "ph": "phase",
    "mkt": "market", "stn": "station", "rly": "railway", "indl": "industrial", "est": "estate",
    "cplx": "complex", "apts": "apartments", "soc": "society", "jn": "junction", "jct": "junction",
    "x": "cross", "crs": "cross", "blk": "block", "dist": "district", "distt": "district",
    "vill": "village", "vpo": "village", "hno": "", "h": "", "no": "", "nos": "", "door": "",
    "plot": "plot", "r": "rue", "ch": "chemin", "rte": "route", "imp": "impasse", "fbg": "faubourg",
    "res": "residence", "zi": "zone industrielle", "za": "zone artisanale", "bp": "", "cedex": "",
    "twp": "township", "hts": "heights", "pt": "point", "twnshp": "township",
    "unit": "", "suite": "", "po": "", "box": "", "c": "", "o": "",
    "null": "", "none": "", "nil": "", "tel": "", "ph.": "",
}
ADDR_STOP = {"the", "of", "and", "de", "du", "des", "la", "le", "les", "l", "d", "a", "near",
             "opposite", "behind", "next", "to", "beside", "besides", "in", "at", "on", "off",
             "via", "post", "office", "floor", "ground", "first", "second", "third"}

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire",
    "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee",
    "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
    "pr": "puerto rico",
}
IN_STATES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam", "br": "bihar",
    "cg": "chhattisgarh", "ct": "chhattisgarh", "ga": "goa", "gj": "gujarat", "hr": "haryana",
    "hp": "himachal pradesh", "jh": "jharkhand", "jk": "jammu and kashmir", "ka": "karnataka",
    "kl": "kerala", "mp": "madhya pradesh", "mh": "maharashtra", "mn": "manipur",
    "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland", "od": "odisha", "or": "odisha",
    "pb": "punjab", "rj": "rajasthan", "sk": "sikkim", "tn": "tamil nadu", "tg": "telangana",
    "ts": "telangana", "tr": "tripura", "up": "uttar pradesh", "uk": "uttarakhand",
    "ut": "uttarakhand", "wb": "west bengal", "dl": "delhi", "ch": "chandigarh",
    "py": "puducherry", "ld": "lakshadweep", "an": "andaman and nicobar", "la": "ladakh",
    "orissa": "odisha", "pondicherry": "puducherry", "uttaranchal": "uttarakhand",
    "new delhi": "delhi", "nct of delhi": "delhi",
}
FR_REGIONS = ["auvergne rhone alpes", "bourgogne franche comte", "bretagne", "centre val de loire",
              "corse", "grand est", "hauts de france", "ile de france", "normandie",
              "nouvelle aquitaine", "occitanie", "pays de la loire", "provence alpes cote d azur"]
FR_DEPTS = ["ain", "aisne", "allier", "ardeche", "ardennes", "ariege", "aube", "aude", "aveyron",
            "bouches du rhone", "calvados", "cantal", "charente", "charente maritime", "cher",
            "correze", "cote d or", "cotes d armor", "creuse", "dordogne", "doubs", "drome", "eure",
            "eure et loir", "finistere", "gard", "haute garonne", "gers", "gironde", "herault",
            "ille et vilaine", "indre", "indre et loire", "isere", "jura", "landes", "loir et cher",
            "loire", "haute loire", "loire atlantique", "loiret", "lot", "lot et garonne", "lozere",
            "maine et loire", "manche", "marne", "haute marne", "mayenne", "meurthe et moselle",
            "meuse", "morbihan", "moselle", "nievre", "nord", "oise", "orne", "pas de calais",
            "puy de dome", "pyrenees atlantiques", "hautes pyrenees", "pyrenees orientales",
            "bas rhin", "haut rhin", "rhone", "haute saone", "saone et loire", "sarthe", "savoie",
            "haute savoie", "paris", "seine maritime", "seine et marne", "yvelines", "deux sevres",
            "somme", "tarn", "tarn et garonne", "var", "vaucluse", "vendee", "vienne",
            "haute vienne", "vosges", "yonne", "territoire de belfort", "essonne", "hauts de seine",
            "seine saint denis", "val de marne", "val d oise"]
FR_DEPT_REGION = {
    "nord": "hauts de france", "pas de calais": "hauts de france", "somme": "hauts de france",
    "aisne": "hauts de france", "oise": "hauts de france",
    "gironde": "nouvelle aquitaine", "landes": "nouvelle aquitaine", "dordogne": "nouvelle aquitaine",
    "charente": "nouvelle aquitaine", "charente maritime": "nouvelle aquitaine",
    "lot et garonne": "nouvelle aquitaine", "pyrenees atlantiques": "nouvelle aquitaine",
    "deux sevres": "nouvelle aquitaine", "vienne": "nouvelle aquitaine", "haute vienne": "nouvelle aquitaine",
    "correze": "nouvelle aquitaine", "creuse": "nouvelle aquitaine",
    "loire atlantique": "pays de la loire", "maine et loire": "pays de la loire", "mayenne": "pays de la loire",
    "sarthe": "pays de la loire", "vendee": "pays de la loire",
    "paris": "ile de france", "seine et marne": "ile de france", "yvelines": "ile de france",
    "essonne": "ile de france", "hauts de seine": "ile de france", "seine saint denis": "ile de france",
    "val de marne": "ile de france", "val d oise": "ile de france",
    "rhone": "auvergne rhone alpes", "isere": "auvergne rhone alpes", "loire": "auvergne rhone alpes",
    "ain": "auvergne rhone alpes", "savoie": "auvergne rhone alpes", "haute savoie": "auvergne rhone alpes",
    "drome": "auvergne rhone alpes", "ardeche": "auvergne rhone alpes", "allier": "auvergne rhone alpes",
    "cantal": "auvergne rhone alpes", "haute loire": "auvergne rhone alpes", "puy de dome": "auvergne rhone alpes",
    "bouches du rhone": "provence alpes cote d azur", "var": "provence alpes cote d azur",
    "vaucluse": "provence alpes cote d azur",
    "haute garonne": "occitanie", "herault": "occitanie", "gard": "occitanie", "aude": "occitanie",
    "pyrenees orientales": "occitanie", "tarn": "occitanie", "gers": "occitanie", "lot": "occitanie",
    "aveyron": "occitanie", "lozere": "occitanie", "ariege": "occitanie", "hautes pyrenees": "occitanie",
    "tarn et garonne": "occitanie",
    "bas rhin": "grand est", "haut rhin": "grand est", "moselle": "grand est", "marne": "grand est",
    "meurthe et moselle": "grand est", "vosges": "grand est", "aube": "grand est", "ardennes": "grand est",
    "meuse": "grand est", "haute marne": "grand est",
    "finistere": "bretagne", "morbihan": "bretagne", "ille et vilaine": "bretagne", "cotes d armor": "bretagne",
    "calvados": "normandie", "manche": "normandie", "orne": "normandie", "eure": "normandie",
    "seine maritime": "normandie",
    "loiret": "centre val de loire", "indre et loire": "centre val de loire", "loir et cher": "centre val de loire",
    "cher": "centre val de loire", "indre": "centre val de loire", "eure et loir": "centre val de loire",
    "cote d or": "bourgogne franche comte", "doubs": "bourgogne franche comte", "jura": "bourgogne franche comte",
    "saone et loire": "bourgogne franche comte", "nievre": "bourgogne franche comte", "yonne": "bourgogne franche comte",
    "haute saone": "bourgogne franche comte", "territoire de belfort": "bourgogne franche comte",
}
_STATE_FULL = set(US_STATES.values()) | set(IN_STATES.values()) | set(FR_REGIONS) | (set(FR_DEPTS) - {"paris"})

_ws = re.compile(r"\s+")
_DOTTED = re.compile(r"\b([a-z](?:\.[a-z])+)\.?(?![a-z])")
_nonalnum = re.compile(r"[^a-z0-9 ]+")
_ord = re.compile(r"\b(\d+)\s*(st|nd|rd|th|er|e|eme|ème)\b")
_alnum_split = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")


def basic(text: str) -> str:
    if not text:
        return ""
    if _INDIC_RE.search(text):
        text = indic_to_latin(text)
    t = strip_accents(text).lower()
    t = t.replace("&", " and ").replace("+", " and ").replace("@", " at ")
    t = re.sub(r"(?<=\w)'s\b", "s", t)
    t = _DOTTED.sub(lambda m: m.group(1).replace(".", ""), t)   # s.a.s / l.l.c. / p.v.t -> sas / llc / pvt
    t = t.replace("'", "").replace("’", "")
    t = _nonalnum.sub(" ", t)
    return _ws.sub(" ", t).strip()


def _join_initials(tokens):
    out, buf = [], []
    for tok in tokens:
        if len(tok) == 1 and tok.isalpha():
            buf.append(tok)
            continue
        if len(buf) >= 2:
            out.append("".join(buf))
        else:
            out.extend(buf)
        buf = []
        out.append(tok)
    if len(buf) >= 2:
        out.append("".join(buf))
    else:
        out.extend(buf)
    return out


_SKEL_MAP = [("sch", "s"), ("tch", "c"), ("ph", "f"), ("bh", "b"), ("dh", "d"), ("th", "t"),
             ("kh", "k"), ("gh", "g"), ("sh", "s"), ("ch", "c"), ("zh", "l"), ("ck", "k"),
             ("q", "k"), ("x", "ks"), ("w", "v"), ("z", "s"), ("y", "i"), ("j", "g")]


@lru_cache(maxsize=300_000)
def skeleton(tok: str) -> str:
    """Consonant skeleton; robust to vowel typos and transliteration noise."""
    if not tok or tok.isdigit():
        return tok
    t = tok
    for a, b in _SKEL_MAP:
        t = t.replace(a, b)
    t = t.replace("c", "k")
    head, rest = t[0], re.sub(r"[aeiouh]", "", t[1:])
    s = head + rest
    return re.sub(r"(.)\1+", r"\1", s)


def canon_num(n: str) -> str:
    n = n.lstrip("0")
    return n or "0"


@lru_cache(maxsize=300_000)
def norm_name(raw: str):
    """-> (core, full, legal_classes(str,sorted), nums(str), is_domain, has_indic)"""
    indic = bool(raw) and has_indic(raw)
    is_dom = 0
    m = _DOMAIN_RAW.match(strip_accents(raw).lower()) if raw else None
    if m:
        t = basic(m.group(1)).replace(" ", "")
        is_dom = 1
    else:
        t = basic(raw)
        t = re.sub(r"\b(d b a|dba|doing business as|trading as|t a|aka|a k a)\b", " dba ", t)
    toks = _join_initials(t.split())
    nums, exp = [], []
    for tok in toks:
        if tok.isdigit():
            if len(tok) < 7:               # long digit runs are phone numbers -> drop
                nums.append(canon_num(tok))
            continue
        exp.extend(NAME_ABBR.get(tok, tok).split())
    legal = sorted({LEGAL_CANON[t] for t in exp if t in LEGAL_CANON})
    # "pvt" alone is part of "pvt ltd"
    core = [t for t in exp if t not in LEGAL_CANON and t not in NAME_STOP]
    if not core:
        core = [t for t in exp if t not in NAME_STOP] or exp
    return (" ".join(core), " ".join(exp), " ".join(legal), " ".join(nums), is_dom, int(indic))


def _replace_segments(raw: str) -> str:
    if not INDIC_SEGMENT_DICT or not _INDIC_RE.search(raw):
        return raw
    segs = [s.strip() for s in raw.split(",")]
    return ", ".join(INDIC_SEGMENT_DICT.get(s, s) for s in segs)


@lru_cache(maxsize=300_000)
def norm_addr(raw: str, country: str = ""):
    """-> (core, state, nums(str)). A comma segment that is only a state/region name
    or code is moved to `state`; everything else stays in `core`."""
    if not raw:
        return ("", "", "")
    raw = _replace_segments(raw)
    c = country.lower()
    code_map = US_STATES if c in ("us", "usa") else IN_STATES if c == "india" else {}
    states, out = set(), []
    for seg in raw.split(","):
        t = basic(seg)
        if not t:
            continue
        if t in code_map:
            states.add(code_map[t]); continue
        if t in _STATE_FULL or t in IN_STATES:
            t = IN_STATES.get(t, t)
            states.add(FR_DEPT_REGION.get(t, t)); continue
        t = _ord.sub(r"\1", t)
        t = _alnum_split.sub(" ", t)
        for tok in t.split():
            v = ADDR_ABBR.get(tok, tok)
            if v:
                out.extend(v.split())
    nums = [canon_num(w) for w in out if w.isdigit()]
    core = [canon_num(w) if w.isdigit() else w for w in out if w not in ADDR_STOP]
    return (" ".join(core), " ".join(sorted(states)), " ".join(dict.fromkeys(nums)))
