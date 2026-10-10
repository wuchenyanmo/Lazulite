"""Independent Japanese requests and original-axis results; no editor model types."""
from dataclasses import dataclass, field
from math import isfinite


@dataclass(frozen=True, slots=True)
class StartHint:
    time: float
    source: str
    route: str | None = None

    def __post_init__(self):
        if not isfinite(self.time) or self.time < 0 or self.source not in {'imported', 'edited', 'automatic', 'unknown'}:
            raise ValueError('invalid start hint')


@dataclass(frozen=True, slots=True)
class JapaneseUnit:
    unit_id: str
    source_range: tuple[int, int]
    reading: str
    reading_language: str
    reading_source: str
    reading_range: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class JapaneseLine:
    line_id: str
    text: str
    units: tuple[JapaneseUnit, ...]
    start_hint: StartHint | None = None


@dataclass(frozen=True, slots=True)
class JapaneseAlignmentRequest:
    lines: tuple[JapaneseLine, ...]

    def __post_init__(self):
        line_ids, unit_ids = set(), set()
        for line in self.lines:
            if not line.line_id or line.line_id in line_ids:
                raise ValueError('duplicate or empty line ID')
            line_ids.add(line.line_id)
            for unit in line.units:
                if not unit.unit_id or unit.unit_id in unit_ids:
                    raise ValueError('duplicate or empty unit ID')
                unit_ids.add(unit.unit_id)
                a, b = unit.source_range
                if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(line.text):
                    raise ValueError('source_range outside Unicode code point range')
                if unit.reading_language not in {'ja', 'en', 'latin', 'unknown'} or not unit.reading_source:
                    raise ValueError('explicit reading language/source required')
                if not isinstance(unit.reading, str):
                    raise ValueError('reading must be a string')
                if unit.reading_range is not None:
                    x, y = unit.reading_range
                    if type(x) is not int or type(y) is not int or not 0 <= x < y:
                        raise ValueError('invalid reading_range')

    @classmethod
    def from_dict(cls, value):
        if value.get('schema') != 'lazulite-japanese-request-v1':
            raise ValueError('unsupported Japanese request schema')
        return cls(tuple(JapaneseLine(l['line_id'], l['text'], tuple(JapaneseUnit(
            **{**u, 'source_range': tuple(u['source_range']),
               'reading_range': tuple(u['reading_range']) if u.get('reading_range') is not None else None}
        ) for u in l['units']), StartHint(**l['start_hint']) if l.get('start_hint') is not None else None)
                         for l in value['lines']))


@dataclass(frozen=True, slots=True)
class Endpoint:
    time: float
    source: str
    basis_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Timing:
    start: Endpoint | None = None
    end: Endpoint | None = None


@dataclass(frozen=True, slots=True)
class Fragment:
    start: float
    end: float
    piece_index: int


@dataclass(frozen=True, slots=True)
class JapaneseIssue:
    object_id: str
    code: str
    values: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class JapaneseUnitResult:
    unit_id: str
    line_id: str
    reading: str
    model_text: str
    raw_timing: Timing
    proposed_timing: Timing
    fragments: tuple[Fragment, ...] = ()
    concat_interval: tuple[float, float] | None = None
    crosses_removed_gap: bool = False
    removed_gap_seconds: float = 0
    character_path: tuple[dict, ...] = ()
    issues: tuple[JapaneseIssue, ...] = ()


@dataclass(frozen=True, slots=True)
class JapaneseAlignmentResult:
    units: tuple[JapaneseUnitResult, ...]
    status: str
    lines: tuple[dict, ...]
    coverage: dict
    calibration: dict
    execution: dict
    issues: tuple[JapaneseIssue, ...] = ()
    schema: str = 'lazulite-japanese-result-v1'
