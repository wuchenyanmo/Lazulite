import numpy as np
import requests
import json
from Lazulite.Search.SearchMusic import search_163_music_file
from Lazulite.Lyric import LyricLineStamp

HEADER = {'User-Agent':"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"}
# 双语歌词API
LYRIC_163_API = "https://music.163.com/api/song/lyric?os=pc&id={song_id}&lv=-1&tv=-1"

def get_163_lyric(song_id: str | int) -> LyricLineStamp | None:
    url = LYRIC_163_API.format(song_id = song_id)
    response = requests.get(url,headers = HEADER, timeout = (5, 7))
    res = response.json()
    if("lrc" not in res):
        return None
    if("nolyric" in res and res["nolyric"]):
        return None
    if("pureMusic" in res and res["pureMusic"]):
        return None
    if(not res['lrc']['lyric']):
        return None
    lyr = LyricLineStamp(res['lrc']['lyric'])
    if("tlyric" in res and res['tlyric']['lyric']):
        lyr.load_translation(res['tlyric']['lyric'])
    return lyr
