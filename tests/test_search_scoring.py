"""离线候选夹具覆盖评分精度、筛选边界和四个来源的排序。"""

from pathlib import Path

import pytest

from lazulite.lyrics.providers.kugou import candidate_from_kugou_result
from lazulite.lyrics.providers.lrclib import candidate_from_lrclib_result
from lazulite.lyrics.providers.netease import candidate_from_netease_result
from lazulite.lyrics.providers.qqmusic import candidate_from_qq_result
from lazulite.lyrics.search import SearchScoreFields, score_search_candidate
from lazulite.lyrics.service import search_lyrics
from lazulite.models.audio import AudioMetadata


def make_candidate(source, candidate_id, name, query):
    """构造固定的来源响应；不请求在线服务。"""
    factories = {
        "netease": (candidate_from_netease_result,
                    {"id": candidate_id, "name": name, "dt": 180_000}),
        "qqmusic": (candidate_from_qq_result,
                    {"mid": candidate_id, "name": name, "interval": 180}),
        "kugou": (candidate_from_kugou_result,
                  {"hash": candidate_id, "songname": name, "duration": 180}),
        "lrclib": (candidate_from_lrclib_result,
                   {"id": candidate_id, "trackName": name, "duration": 180}),
    }
    factory, payload = factories[source]
    return factory(payload, title=query, duration=180, artist=None, album=None)


class FrozenProvider:
    priority = 1

    def __init__(self, source, rows):
        self.source_name = source
        self.rows = rows

    def search(self, *, title, score_title=None, **kwargs):
        return [make_candidate(self.source_name, key, name, score_title or title)
                for key, name in self.rows]


@pytest.mark.parametrize("source", ["netease", "qqmusic", "kugou", "lrclib"])
def test_provider_float_scores_determine_top_three(source):
    provider = FrozenProvider(source, [
        ("4", "love"), ("3", "a story"),
        ("2", "lovely story"), ("1", "story"),
    ])
    metadata = AudioMetadata(Path("unused.wav"), "love story", None, None, 180)
    result = search_lyrics(metadata, providers=(provider,))
    assert [item.candidate_id for item in result] == ["1", "2", "3"]
    assert [item.match_score for item in result] == pytest.approx(
        [86.6666666667, 84.3636363636, 83.6199095023])


def test_threshold_uses_unrounded_score_and_ties_use_candidate_id():
    provider = FrozenProvider("netease", [
        ("b", "鳥の詩 (TV Size)"), ("a", "鳥の詩 (TV Size)"),
    ])
    metadata = AudioMetadata(Path("unused.wav"), "鳥の詩", None, None, 180)
    result = search_lyrics(metadata, providers=(provider,), min_match_score=75)
    assert [item.candidate_id for item in result] == ["a", "b"]
    assert [item.match_score for item in result] == pytest.approx([77.14285714285714, 77.14285714285714])
    assert search_lyrics(metadata, providers=(provider,), min_match_score=77.2) == ()


@pytest.mark.parametrize("duration, expected", [
    (None, 0), (164.9, 0), (165, 100), (195, 100), (195.1, 0),
])
def test_duration_boundary_preserves_alias_and_traditional_matching(duration, expected):
    fields = SearchScoreFields(duration, ("unrelated", "鸟之诗"), ("Lia",), ("AIR",))
    assert score_search_candidate(
        title="鳥之詩", duration=180, fields=fields, artist="Lia", album="AIR"
    ) == pytest.approx(expected)


def test_empty_fields_have_no_score_and_missing_fields_reweight():
    assert score_search_candidate(
        title="", duration=180, fields=SearchScoreFields(180, ("",), (), ())
    ) == 0
    fields = SearchScoreFields(180, ("歌",), (), ())
    assert score_search_candidate(title="歌", duration=180, fields=fields) == 100
    # 提供了查询字段但来源没有该字段时，保留其权重并计零分。
    assert score_search_candidate(
        title="歌", duration=180, artist="Lia", album="AIR", fields=fields
    ) == pytest.approx(70)

@pytest.mark.parametrize('query,candidate',[('桜','樱'),('國々','国国'),('裏','里')])
def test_cjk_views_include_japanese_traditional_and_iteration(query,candidate):
    fields=SearchScoreFields(180,(candidate,),(),())
    assert score_search_candidate(title=query,duration=180,fields=fields)==100
    assert score_search_candidate(title=query,duration=196,fields=fields)==0


def test_title_only_fallback_runs_after_empty_artist_query():
    queries=[]
    class Provider:
        source_name='netease'
        priority=1
        def search(self, *, title, artist=None, **kwargs):
            queries.append((title,artist))
            if artist:return []
            return [make_candidate('netease','one',title,title)]
    metadata=AudioMetadata(Path('unused.wav'),'Song','Singer',None,180)
    rows=search_lyrics(metadata,providers=(Provider(),))
    assert rows[0].candidate_id=='one'
    assert queries[0]==('Song','Singer') and queries[-1]==('Song',None)

@pytest.mark.parametrize('error',[RuntimeError('network'),TypeError('programming')])
def test_search_stops_query_variants_on_failure_and_preserves_programming_errors(error):
    queries=[]
    class Provider:
        source_name='netease'
        priority=1
        def search(self, **kwargs):
            queries.append(kwargs)
            raise error
    with pytest.raises(type(error)):
        search_lyrics(AudioMetadata(Path('unused.wav'),'Song (Live)','Singer','Album',180),providers=(Provider(),))
    assert len(queries)==1


def test_album_does_not_duplicate_keywords_but_remains_available_for_scoring():
    queries=[]
    class Provider:
        source_name='qqmusic'
        priority=1
        def search(self, **kwargs):
            queries.append(kwargs)
            return []
    search_lyrics(AudioMetadata(Path('unused.wav'),'Song','Singer','Album',180),providers=(Provider(),))
    assert [(q['title'],q['artist'],q['album']) for q in queries]==[('Song','Singer',None),('Song',None,None)]
    assert all(q['score_album']=='Album' for q in queries)

def test_score_punctuation_and_featuring_labels_do_not_change_artist_identity():
    fields=SearchScoreFields(180,('Song',),('Singer Guest',),())
    assert score_search_candidate(title='Song',duration=180,artist='Singer (feat. Guest)',fields=fields)==pytest.approx(100)
    from lazulite.lyrics.search import _score_variants
    assert 'feature' in _score_variants('feature')
