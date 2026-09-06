"""酷狗搜索与歌词下载适配器。"""
from __future__ import annotations
import base64
import warnings
import requests
from requests.exceptions import RequestException
from lazulite.lyrics.lrc import LrcDocument, read_lrc
from lazulite.lyrics.normalize import KUGOU_ARTIST_SEPARATORS, split_text
from lazulite.lyrics.search import SearchScoreFields, score_search_candidate
from .base import SearchCandidate

_HEADERS={"User-Agent":"Mozilla/5.0","Referer":"https://www.kugou.com/"}
_SEARCH="https://mobilecdn.kugou.com/api/v3/search/song"; _LYRIC_SEARCH="https://lyrics.kugou.com/search"; _DOWNLOAD="https://lyrics.kugou.com/download"

def _get(session, url, params):
    try:
        response=session.get(url,params=params,headers=_HEADERS,timeout=(5,10)); response.raise_for_status(); return response.json()
    except (RequestException,ValueError) as error:
        warnings.warn(f"酷狗请求失败，已跳过当前请求: {error}",RuntimeWarning); return {}

def candidate_from_kugou_result(item, *, title, duration, artist, album):
    artists=tuple(split_text(item.get("singername"),KUGOU_ARTIST_SEPARATORS)); names=tuple(str(item.get(key) or "").strip() for key in ("songname","songname_original","filename","othername") if item.get(key))
    fields=SearchScoreFields(float(item.get("duration") or 0),names,artists,(str(item.get("album_name") or "").strip(),))
    candidate_id=str(item.get("hash") or "")
    if not candidate_id: raise ValueError("酷狗搜索结果缺少 hash")
    return SearchCandidate("kugou",candidate_id,names[0] if names else ""," / ".join(artists) or None,str(item.get("album_name") or "").strip() or None,fields.duration,score_search_candidate(title=title,duration=duration,fields=fields,artist=artist,album=album),dict(item))

class KugouProvider:
    source_name="kugou"; priority=100
    def __init__(self,*,session=None): self._session=session or requests.Session()
    def search(self,title,duration,artist=None,album=None,score_title=None,score_artist=None,score_album=None):
        payload=_get(self._session,_SEARCH,{"format":"json","keyword":title,"page":1,"pagesize":20,"showtype":1})
        return sorted((candidate_from_kugou_result(item,title=score_title or title,duration=duration,artist=score_artist if score_artist is not None else artist,album=score_album if score_album is not None else album) for item in ((payload.get("data") or {}).get("info") or []) if item.get("hash")),key=lambda item:item.match_score,reverse=True)
    def fetch(self,candidate):
        query=f"{candidate.artist} - {candidate.title}" if candidate.artist else candidate.title
        candidates=_get(self._session,_LYRIC_SEARCH,{"ver":1,"man":"yes","client":"pc","keyword":query,"duration":round((candidate.duration or 0)*1000),"hash":candidate.candidate_id}).get("candidates") or []
        for item in candidates:
            payload=_get(self._session,_DOWNLOAD,{"ver":1,"client":"pc","id":item.get("id"),"accesskey":item.get("accesskey"),"fmt":"lrc","charset":"utf8"})
            try: text=base64.b64decode(payload.get("content") or "").decode("utf-8",errors="replace")
            except (ValueError,TypeError): continue
            lyrics=read_lrc(text)
            if lyrics.document.lines: return lyrics
        return None
