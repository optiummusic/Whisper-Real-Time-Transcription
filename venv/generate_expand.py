# Зависимости:
#   pip install pymorphy3 lemminflect morfeusz2 spacy mlconjug3 toml
#
#   Модели morfeusz2 (для Polish):
#     morfeusz2 поставляется со встроенным словарём SGJP — дополнительных
#     загрузок не требуется.
#
#   Модели spacy (для German / French / Spanish / Polish-опционально):
#     python -m spacy download de_core_news_sm
#     python -m spacy download fr_core_news_sm
#     python -m spacy download es_core_news_sm

    
import re
import json
import toml
from functools import lru_cache

from pymorphy3 import MorphAnalyzer
import lemminflect
import spacy
from verbecc import CompleteConjugator, LangCodeISO639_1 as Lang
# ─── Model Loading ────────────────────────────────────────────────────────────

print("Загрузка лингвистических моделей (это может занять несколько секунд)...")

uk_morph = MorphAnalyzer(lang="uk")
ru_morph = MorphAnalyzer(lang="ru")

# Загружаем легковесные модели spaCy
nlp_de = spacy.load("de_core_news_sm", disable=["ner", "parser"])
nlp_fr = spacy.load("fr_core_news_sm", disable=["ner", "parser"])
nlp_es = spacy.load("es_core_news_sm", disable=["ner", "parser"])
nlp_pl = spacy.load("pl_core_news_sm", disable=["ner", "parser"])

# Инициализируем спрягатели для романских языков
conj_fr = CompleteConjugator(lang=Lang.fr)
conj_es = CompleteConjugator(lang=Lang.es)

# ─── Constants ────────────────────────────────────────────────────────────────

BASE_LANG_RE = re.compile(r"^[a-z]{2}$")

NUMERAL_PROTECT = {"три", "один", "одна", "одне", "два", "дві"}
EN_NUMERAL_WORDS = {"one", "two", "three", "four", "five"}
EN_SKIP = {
    "hello", "good morning", "good evening",
    "thank you", "yes", "no", "please", "sorry",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _norm(word: str) -> str:
    return word.strip().lower()


def _is_multiword(word: str) -> bool:
    return " " in word or "-" in word


def _to_list(value) -> list:
    return value if isinstance(value, list) else []


# ─── pymorphy3: Ukrainian & Russian ───────────────────────────────────────────

@lru_cache(maxsize=50_000)
def _parse_uk_ru(lang: str, word: str):
    analyzer = uk_morph if lang == "uk" else ru_morph
    return analyzer.parse(word)


def _gen_pymorphy(lang: str, word: str) -> set[str]:
    word = _norm(word)
    if not word or _is_multiword(word):
        return {word} if word else set()

    forms = {word}
    try:
        parses = _parse_uk_ru(lang, word)
        if not parses:
            return forms

        best_parse = max(parses, key=lambda p: p.score)

        if word in NUMERAL_PROTECT:
            allowed = {"NUMR", "ADJF"}
            best_parse = next(
                (p for p in parses if p.tag.POS in allowed), best_parse
            )

        base_pos = best_parse.tag.POS
        if base_pos not in {"NOUN", "VERB", "ADJF", "ADJS", "NUMR"}:
            return forms

        for f in best_parse.lexeme:
            if f.tag.POS == base_pos:
                forms.add(f.word.lower())

    except Exception as e:
        print(f"Ошибка [{lang}] '{word}': {e}")

    return forms


def gen_uk(word: str) -> set[str]:
    return _gen_pymorphy("uk", word)


def gen_ru(word: str) -> set[str]:
    return _gen_pymorphy("ru", word)


# ─── lemminflect: English ─────────────────────────────────────────────────────

def gen_en(word: str) -> set[str]:
    word = _norm(word)
    if not word or _is_multiword(word):
        return {word} if word else set()

    forms = {word}

    if word in EN_SKIP:
        return forms

    try:
        if word in EN_NUMERAL_WORDS:
            inf = lemminflect.getInflection(word, "NNS")
            if inf:
                forms.update(map(str.lower, inf))
            return forms

        for tag in ("NNS", "VBD", "VBG", "VBN", "VBP", "VBZ"):
            inf = lemminflect.getInflection(word, tag)
            if inf:
                forms.update(map(str.lower, inf))

    except Exception as e:
        print(f"Ошибка [en] '{word}': {e}")

    return forms


# ─── morfeusz2: Polish ────────────────────────────────────────────────────────

def gen_pl(word: str) -> set[str]:
    word = _norm(word)
    if not word or _is_multiword(word):
        return {word} if word else set()

    forms = {word}
    try:
        doc = nlp_pl(word)
        lemma = doc[0].lemma_.lower()
        forms.add(lemma)

        # Простая эвристика для польского (существительные)
        # Добавим типичные окончания, чтобы ловить больше вариантов при поиске
        if doc[0].pos_ == "NOUN":
            base = lemma
            # Для слова 'pies' корень меняется на 'ps', но регулярками это не поймать.
            # Зато для большинства других слов сработает:
            forms.update({
                base + "a",   # kota, psa
                base + "y",   # koty
                base + "i",   # ptaki
                base + "ów",  # kotów
                base + "em",  # kotem
                base + "owi", # kotowi
                base + "ach", # kotach
            })
            # Можно еще жестко захардкодить: если оканчивается на 'es', корень без 'es' + 's' (pies -> ps)
            if base.endswith("ies"):
                root = base[:-3] + "s"
                forms.update({root + "a", root + "u", root + "y", root + "ów", root + "em"})

    except Exception as e:
        print(f"Ошибка [pl] '{word}': {e}")

    return forms


# ─── spaCy: German ────────────────────────────────────────────────────────────

_DE_UMLAUT = str.maketrans({"a": "ä", "o": "ö", "u": "ü"})
_DE_NOUN_SUFFIXES  = ("", "e", "en", "er", "es", "n", "s", "em")
_DE_UMLAUT_SUFFIXES = ("", "e", "en", "er", "es")

def _de_noun_forms(lemma: str) -> set[str]:
    forms = set()
    base = lemma.lower()
    for suffix in _DE_NOUN_SUFFIXES:
        forms.add(base + suffix)
    ubase = base.translate(_DE_UMLAUT)
    if ubase != base:
        for suffix in _DE_UMLAUT_SUFFIXES:
            forms.add(ubase + suffix)
    return forms

def gen_de(word: str) -> set[str]:
    word = _norm(word)
    if not word or _is_multiword(word):
        return {word} if word else set()
    forms = {word}
    try:
        doc = nlp_de(word)
        lemma = doc[0].lemma_.lower()
        pos = doc[0].pos_
        forms.add(lemma)
        if pos in ("NOUN", "PROPN"):
            forms |= _de_noun_forms(lemma)
    except Exception as e:
        print(f"Ошибка [de] '{word}': {e}")
    return forms


# ─── spaCy + mlconjug3: French ────────────────────────────────────────────────

_FR_PLURAL_RULES: list[tuple[str, list[str]]] = [
    ("al",  ["aux", "ale", "ales"]),
    ("eau", ["eaux", "eaus"]),
    ("eu",  ["eux", "eus"]),
    ("if",  ["ive", "ifs", "ives"]),
    ("eux", ["euse", "euses"]),
    ("e",   ["es"]),
]

def _fr_noun_adj_forms(word: str) -> set[str]:
    forms = {word}
    for ending, suffixes in _FR_PLURAL_RULES:
        if word.endswith(ending):
            stem = word[: -len(ending)]
            for s in suffixes:
                forms.add(stem + s)
            return forms
    forms.update({word + "s", word + "es"})
    return forms

def gen_fr(word: str) -> set[str]:
    word = _norm(word)
    if not word or _is_multiword(word): return {word}
    
    forms = {word}
    try:
        doc = nlp_fr(word)
        token = doc[0]
        lemma = token.lemma_.lower()
        forms.add(lemma)

        if token.pos_ in ("VERB", "AUX"):
            # В 2.0 метод возвращает объект с методом to_json() или get_data()
            # По умолчанию conjugate возвращает ВСЕ формы
            data = conj_fr.conjugate(lemma).get_data()
            
            # Проходим по всем наклонениям и временам
            for mood_tenses in data['moods'].values():
                for tense_forms in mood_tenses.values():
                    for f in tense_forms:
                        if isinstance(f, str) and f:
                            # Очищаем от местоимений (je, tu...), если нужно только слово
                            clean_f = f.split()[-1].lower()
                            forms.add(clean_f)
        else:
            forms |= _fr_noun_adj_forms(lemma)
    except Exception as e:
        print(f"Ошибка [fr] '{word}': {e}")
    return forms


# ─── spaCy + mlconjug3: Spanish ───────────────────────────────────────────────

_ES_ACCENT = str.maketrans("áéíóúüÁÉÍÓÚÜ", "aeiouuAEIOUU")

def _es_noun_adj_forms(word: str) -> set[str]:
    forms = {word}

    if word.endswith("o"):
        forms.update({word[:-1] + "a", word[:-1] + "os", word[:-1] + "as"})
    elif word.endswith("a"):
        forms.update({word[:-1] + "o", word[:-1] + "os", word[:-1] + "as"})
    elif word.endswith("z"):
        forms.add(word[:-1] + "ces")
    elif word.endswith("ión"):
        forms.add(word[:-3] + "iones")
    elif word[-1] in "aeiouáéíóú":
        forms.add(word + "s")
    else:
        forms.update({word + "s", word + "es"})

    no_acc = word.translate(_ES_ACCENT)
    if no_acc != word:
        forms.update({no_acc, no_acc + "s", no_acc + "es"})

    return forms

def gen_es(word: str) -> set[str]:
    word = _norm(word)
    if not word or _is_multiword(word): return {word}
    
    forms = {word}
    try:
        doc = nlp_es(word)
        token = doc[0]
        lemma = token.lemma_.lower() 
        forms.add(lemma)

        if token.pos_ in ("VERB", "AUX"):
            # В 2.0 метод возвращает объект с методом to_json() или get_data()
            # По умолчанию conjugate возвращает ВСЕ формы
            data = conj_es.conjugate(lemma).get_data()
            
            # Проходим по всем наклонениям и временам
            for mood_tenses in data['moods'].values():
                for tense_forms in mood_tenses.values():
                    for f in tense_forms:
                        if isinstance(f, str) and f:
                            # Очищаем от местоимений (je, tu...), если нужно только слово
                            clean_f = f.split()[-1].lower()
                            forms.add(clean_f)
        else:
            forms |= _es_noun_adj_forms(lemma)
    except Exception as e:
        print(f"Ошибка [fr] '{word}': {e}")
    return forms


# ─── Dispatch ─────────────────────────────────────────────────────────────────

def gen_identity(word: str) -> set[str]:
    word = _norm(word)
    return {word} if word else set()

LANG_GENERATORS: dict[str, callable] = {
    "en": gen_en,
    "uk": gen_uk,
    "ru": gen_ru,
    "de": gen_de,
    "fr": gen_fr,
    "es": gen_es,
    "pl": gen_pl,
}

def get_forms(word: str, lang: str) -> set[str]:
    word = _norm(word)
    if not word:
        return set()
    return LANG_GENERATORS.get(lang, gen_identity)(word)


# ─── Concept Expansion ────────────────────────────────────────────────────────

def expand_concept(concept: dict) -> dict:
    base: dict[str, str] = {}
    custom: dict[str, list] = {}

    for k, v in concept.items():
        if BASE_LANG_RE.match(k):
            base[k] = _norm(v)
        elif k == "custom" and isinstance(v, dict):
            custom = v

    generated: dict[str, list] = {}

    for lang, base_word in base.items():
        forms = get_forms(base_word, lang)

        extra = {
            _norm(f)
            for f in forms
            if _norm(f) and _norm(f) != base_word
        }

        custom_words = set(map(_norm, _to_list(custom.get(lang, []))))
        extra -= custom_words

        if extra:
            generated[lang] = sorted(extra)

    return {
        "base": base,
        "custom": custom,
        "generated": generated,
    }


# ─── TOML Output ──────────────────────────────────────────────────────────────

def _q(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)

def write_toml(data, concepts, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# ─── dictionary/main.lxdb.toml ────────────────────────────────────────────────\n")
        f.write("# Source dictionary for the LXDB binary format.\n")
        f.write("#\n")
        f.write("# To compile:\n")
        f.write("#   lxdb compile dictionary/main.lxdb.toml dictionary/main.lxdb\n")
        f.write("#\n")
        f.write("# The compiled .lxdb file is what gets loaded at runtime.\n")
        f.write("# This TOML file is the human-editable source of truth.\n")
        f.write("# ─────────────────────────────────────────────────────────────────────────────\n\n")

        f.write("[meta]\n")
        f.write(f'version = {int(data["meta"]["version"])}\n')
        f.write(f'description = {_q(data["meta"]["description"])}\n\n')

        f.write("# BCP-47 language codes → human-readable names.\n")
        f.write("# Add/remove languages here. The compiler handles all wiring automatically.\n")
        f.write("[languages]\n")
        for k, v in data["languages"].items():
            f.write(f"{k} = {_q(v)}\n")

        f.write("\n")
        f.write("# ─── Concepts ─────────────────────────────────────────────────────────────────\n")
        f.write("# Each [[concepts]] block is ONE semantic concept.\n")
        f.write("# Keys are BCP-47 language codes from [languages] above.\n")
        f.write("# Missing languages are silently skipped (no translation stored for that lang).\n")
        f.write("# Multi-word phrases are allowed — they hash as a unit (ngram support).\n")
        f.write("# Generated keyword is generated by python script, custom is, well, custom added.\n")
        f.write("# ─────────────────────────────────────────────────────────────────────────────\n\n")

        for concept in concepts:
            f.write("[[concepts]]\n")

            for k, v in concept["base"].items():
                f.write(f"{k} = {_q(v)}\n")

            if concept["custom"]:
                f.write("\n[concepts.custom]\n")
                for lang, words in concept["custom"].items():
                    arr = ", ".join(_q(w) for w in words)
                    f.write(f"{lang} = [{arr}]\n")

            if concept["generated"]:
                f.write("\n[concepts.generated]\n")
                for lang, words in concept["generated"].items():
                    arr = ", ".join(_q(w) for w in words)
                    f.write(f"{lang} = [{arr}]\n")

            f.write("\n")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    input_file  = "../dictionary/main.lxdb.toml"
    output_file = "../dictionary/expanded.toml"

    with open(input_file, "r", encoding="utf-8") as f:
        data = toml.load(f)

    expanded = [expand_concept(c) for c in data.get("concepts", [])]
    write_toml(data, expanded, output_file)

    print("Готово: custom сохранён, generated добавлен, формат соблюдён.")

if __name__ == "__main__":
    main()