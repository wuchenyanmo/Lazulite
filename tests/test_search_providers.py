"""搜索接口与关键词的离线行为回归。"""
import pytest
from lazulite.lyrics.providers.qqmusic import QQMusicProvider
from lazulite.lyrics.providers.kugou import KugouProvider
from lazulite.lyrics.providers.netease import NeteaseProvider

@pytest.mark.parametrize('fallback',[False,True])
def test_qq_search_uses_full_results_and_fallback(monkeypatch,fallback):
    provider=QQMusicProvider();calls=[]
    song={'mid':'mid','id':1,'name':'歌','interval':180,'singer':[{'name':'歌手'}],'album':{'name':'专辑'}}
    def request(url,**kwargs):
        calls.append((url,kwargs))
        if kwargs.get('body'):
            param=kwargs['body']['req_0']['param']
            assert param['query']=='歌 歌手' and param['num_per_page']==30
            return {'code':1} if fallback else {'code':0,'req_0':{'code':0,'data':{'body':{'song':{'list':[song]}}}}}
        assert url.endswith('search_for_qq_cp') and kwargs['params']['n']==30
        return {'code':0,'data':{'song':{'list':[song]}}}
    monkeypatch.setattr(provider,'_request',request)
    rows=provider.search('歌',180,'歌手','专辑')
    assert rows[0].candidate_id=='mid' and rows[0].match_score==100
    assert len(calls)==(2 if fallback else 1)

@pytest.mark.parametrize('fallback',[False,True])
def test_kugou_keywords_fallback_and_artist_split(monkeypatch,fallback):
    import lazulite.lyrics.providers.kugou as module
    def get(session,url,params,**kwargs):
        assert params['keyword']=='歌 歌手' and params['pagesize']==30
        if 'mobiles.' in url:
            if fallback: raise RuntimeError('offline')
            return {'data':{'info':[{'hash':'h','songname':'歌','singername':'歌手 / B','duration':180}]}}
        return {'data':{'lists':[{'FileHash':'h','SongName':'歌','SingerName':'歌手 / B','Duration':180}]}}
    monkeypatch.setattr(module,'_get',get)
    row=KugouProvider().search('歌',180,'歌手')[0]
    assert row.artist=='歌手 / B' and row.match_score==pytest.approx(100)
