"""Shared lightweight input preparation for previews and alignment execution."""
from dataclasses import dataclass, replace
from functools import lru_cache
import re
from .kana import to_hiragana, is_kana, split_morae, is_kana_text
from .profile import JaHepburnProfile, is_hold

VERSION = 'alignment-input/1'


@dataclass(frozen=True)
class InputOptions:
    latin_normalize: bool = True
    strip_tone_digits: bool = False

    def __post_init__(self):
        if type(self.latin_normalize) is not bool or type(self.strip_tone_digits) is not bool:
            raise ValueError('input options must be boolean')


@lru_cache(maxsize=1)
def romanizer():
    from uroman import Uroman
    return Uroman()


def reading_pieces(reading, options=InputOptions()):
    """Internal script/mora pieces, with original reading code point ranges."""
    if not options.latin_normalize:
        return [(reading, 0, len(reading), False)]
    # Keep combining marks and half-width voicing attached before normalization.
    import unicodedata
    clusters = []
    for i, c in enumerate(reading):
        if clusters and (unicodedata.category(c).startswith('M') or c in 'ﾞﾟ'):
            clusters[-1][0] += c
            clusters[-1][2] = i+1
        else:
            clusters.append([c, i, i+1])
    runs = []
    for text, a, b in clusters:
        norm = to_hiragana(text)
        kana = bool(norm) and all(is_kana(c) or c == '~' for c in norm)
        if runs and runs[-1][3] == kana:
            runs[-1][0] += text; runs[-1][2] = b
        else:
            runs.append([text, a, b, kana])
    result = []
    for text, a, b, kana in runs:
        if kana:
            # Derive normalized mora boundaries from whole source clusters.
            group = [c for c in clusters if a <= c[1] < b]
            norm_ranges = [(c, x, y) for raw, x, y in group for c in to_hiragana(raw)]
            offset = 0
            for mora in split_morae(to_hiragana(text)):
                count = len(mora.text)
                result.append((mora.text, norm_ranges[offset][1], norm_ranges[offset+count-1][2], True))
                offset += count
        else:
            result.append((text, a, b, False))
    return result or [(reading, 0, len(reading), False)]


def split_request(request, options=InputOptions()):
    """Return internal units and their stable original-unit mapping; no editor mutation."""
    lines, origins = [], {}
    for line in request.lines:
        units = []
        for unit in line.units:
            pieces = reading_pieces(unit.reading, options)
            for i, (text, a, b, kana) in enumerate(pieces):
                uid = unit.unit_id if len(pieces) == 1 else f'{unit.unit_id}/piece/{i}'
                origins[uid] = unit.unit_id
                units.append(replace(unit, unit_id=uid, reading=text, reading_language='ja' if kana else 'unknown',
                                     reading_range=(a,b) if b>a else None, input_group=unit.input_group or unit.unit_id,
                                     source_range=(unit.source_range[0]+a,unit.source_range[0]+b) if unit.reading_source=='body' and b>a and unit.source_range[1]-unit.source_range[0]==len(unit.reading) else unit.source_range))
        lines.append(replace(line, units=tuple(units)))
    return replace(request, lines=tuple(lines)), origins


def model_texts(readings, options=InputOptions(), groups=None):
    pieces = [reading_pieces(r, options) for r in readings]
    flat = [p for group in pieces for p in group]
    kana_values = JaHepburnProfile().unit_texts([p[0] if p[3] else '' for p in flat], ['ja']*len(flat), [[] for _ in flat])
    group_ids = [group for group, pieces_for_unit in zip(groups if groups is not None else range(len(readings)), pieces) for _ in pieces_for_unit]
    values = []
    for p, kana_text in zip(flat, kana_values):
        value = kana_text if p[3] else romanizer().romanize_string(p[0]) if options.latin_normalize and p[0] else p[0]
        values.append(value)
    for i, value in enumerate(values):
        if options.strip_tone_digits:
            following = next((values[j] for j in range(i+1,len(values)) if group_ids[j]==group_ids[i] and values[j]), '')
            pattern = r'\d+(?=\s|$)' if not following or following[0].isspace() else r'\d+(?=\s)'
            value = re.sub(pattern, '', value)
        # Whitespace is a separator; punctuation/diacritics are checked, never silently erased.
        values[i] = ''.join(value.lower().split())
    out, offset = [], 0
    for group in pieces:
        out.append(''.join(values[offset:offset+len(group)])); offset += len(group)
    return out


def profile_request(request, vocabulary, blank, special_tokens=(), encoder=None, options=InputOptions()):
    texts, tokens, ranges, problems = {}, [], {}, {}
    for line in request.lines:
        values = model_texts([u.reading for u in line.units], options, [u.input_group or u.unit_id for u in line.units])
        for u, text in zip(line.units, values):
            texts[u.unit_id] = text
            begin = len(tokens)
            encoded = encoder(text, vocabulary) if encoder is not None and text else ([vocabulary[c] for c in text] if text and all(c in vocabulary and c not in special_tokens and c not in '| ' and vocabulary[c] != blank for c in text) else None)
            if encoded:
                if any(type(t) is not int or t == blank or t not in vocabulary.values() for t in encoded):
                    raise ValueError('encoder produced invalid target IDs')
                tokens.extend(encoded)
            else:
                problems[u.unit_id] = 'profile_hold' if options.latin_normalize and is_hold(to_hiragana(u.reading)) else 'unresolved_reading'
            ranges[u.unit_id] = (begin, len(tokens))
    return texts, tokens, ranges, problems


def prepare_input(request, vocabulary, blank, special_tokens=(), options=InputOptions()):
    internal, origins = split_request(request, options)
    texts, tokens, ranges, problems = profile_request(internal, vocabulary, blank, special_tokens, options=options)
    return {'request': internal, 'origins': origins, 'texts': texts, 'tokens': tokens, 'ranges': ranges,
            'problems': problems, 'version': VERSION,
            'unknown_characters': {uid: sorted({c for c in text if c not in vocabulary or c in special_tokens or vocabulary.get(c) == blank}) for uid, text in texts.items() if uid in problems}}
