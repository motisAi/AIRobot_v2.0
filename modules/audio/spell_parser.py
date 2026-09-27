"""Turn spoken, spelled-out speech into an exact string (e.g. a Wi-Fi password).

Convention (spoken by the user):
  - a letter, or its NATO word ("alpha".."zulu")  -> that letter (lowercase)
  - "capital"/"cap"/"upper"/"uppercase" before it -> uppercase that letter
  - "lower"/"lowercase"/"small" before it         -> force lowercase
  - number words ("five") or digits               -> digits
  - symbols by name: at, dot, dash, underscore, hash, dollar, star, ...

Speech-to-text merges/garbles isolated letters unpredictably, so this is a
best-effort parser meant to be paired with a spoken read-back + confirmation in
the caller. NATO words are far more reliable than bare letters.
"""

from __future__ import annotations

NATO = {
    'alpha': 'a', 'alfa': 'a', 'bravo': 'b', 'charlie': 'c', 'delta': 'd',
    'echo': 'e', 'foxtrot': 'f', 'golf': 'g', 'hotel': 'h', 'india': 'i',
    'juliet': 'j', 'juliett': 'j', 'kilo': 'k', 'lima': 'l', 'mike': 'm',
    'november': 'n', 'oscar': 'o', 'papa': 'p', 'quebec': 'q', 'romeo': 'r',
    'sierra': 's', 'tango': 't', 'uniform': 'u', 'victor': 'v', 'whiskey': 'w',
    'xray': 'x', 'yankee': 'y', 'zulu': 'z',
}
NUM = {'zero': '0', 'oh': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
       'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'}
SYM = {
    'at': '@', 'dot': '.', 'point': '.', 'period': '.', 'dash': '-', 'hyphen': '-',
    'minus': '-', 'underscore': '_', 'hash': '#', 'hashtag': '#', 'pound': '#',
    'dollar': '$', 'star': '*', 'asterisk': '*', 'percent': '%', 'ampersand': '&',
    'and': '&', 'plus': '+', 'slash': '/', 'backslash': '\\', 'exclamation': '!',
    'bang': '!', 'question': '?', 'equals': '=', 'equal': '=', 'colon': ':',
    'semicolon': ';', 'comma': ',', 'tilde': '~', 'caret': '^', 'space': ' ',
    'pipe': '|', 'apostrophe': "'", 'quote': "'",
}
_UPPER = {'capital', 'cap', 'upper', 'uppercase', 'caps', 'big'}
_LOWER = {'lower', 'lowercase', 'small'}
# multi-word phrases collapsed before tokenising
_PHRASES = {
    'exclamation mark': '!', 'exclamation point': '!', 'question mark': '?',
    'at sign': '@', 'number sign': '#', 'full stop': '.', 'open paren': '(',
    'close paren': ')', 'open bracket': '(', 'close bracket': ')',
    'double quote': '"',
}


def parse_spelled(text: str) -> str:
    """Best-effort convert one spoken chunk into the characters it represents."""
    if not text:
        return ''
    low = ' ' + text.lower().strip() + ' '
    for phrase, ch in _PHRASES.items():
        low = low.replace(' ' + phrase + ' ', ' \x00' + ch + ' ')
    out = []
    case = None
    for tok in low.split():
        t = tok.strip('.,')
        if not t:
            continue
        if t.startswith('\x00'):        # pre-resolved multi-word symbol
            out.append(t[1:]); case = None; continue
        if t in _UPPER:
            case = 'up'; continue
        if t in _LOWER:
            case = 'low'; continue
        ch = None
        if t in NATO:
            ch = NATO[t]
        elif t in NUM:
            ch = NUM[t]
        elif t in SYM:
            ch = SYM[t]
        elif len(t) == 1 and t.isalnum():
            ch = t
        elif t.isdigit():               # "2024" spoken as one token
            out.extend(list(t)); case = None; continue
        elif t.isalpha():               # STT merged letters into a word
            ch = t
        if ch is None:
            continue
        if ch.isalpha():
            ch = ch.upper() if case == 'up' else ch.lower()
        out.append(ch)
        case = None
    return ''.join(out)


_SPEAK_SYM = {'@': 'at', '.': 'dot', '-': 'dash', '_': 'underscore', '#': 'hash',
              '$': 'dollar', '*': 'star', '!': 'exclamation', '?': 'question mark',
              '&': 'and', '+': 'plus', '/': 'slash', '%': 'percent', ' ': 'space',
              '=': 'equals', ':': 'colon', ';': 'semicolon', ',': 'comma'}


def spell_out(s: str) -> str:
    """Human read-back of a string, so Stella can confirm what she heard."""
    parts = []
    for c in s:
        if c.isupper():
            parts.append(f"capital {c.lower()}")
        elif c.isdigit() or (c.isalpha() and c.islower()):
            parts.append(c)
        else:
            parts.append(_SPEAK_SYM.get(c, c))
    return ", ".join(parts)
