import requests
import json
from urllib.parse import quote
from itertools import chain
from fuzzywuzzy import fuzz
from mutagen.mp4 import MP4
import numpy as np
import pandas as pd
import os

HEADER = {'User-Agent':"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"}
SEARCH_LIMIT = 20
# 第三方网易云搜索API，信息更全
SEARCH_163_API_THIRD = "https://apis.netstart.cn/music/cloudsearch?keywords={name}"
# 官方网易云搜索API
SEARCH_163_API = "https://music.163.com/api/search/get?s={name}&type=1&offset=0&limit={limit}"

def parse_163_artist_dict(artist_dict: dict) -> list[str]:
    '''
    解析返回结果'ar'属性中的单个对象，返回包括歌手名字、翻译名字、别名在内的列表，去除重复
    '''
    artist_list = [artist_dict['name']]
    if('tns' in artist_dict):
        artist_list.extend(artist_dict['tns'])
    if('alia' in artist_dict):
        artist_list.extend(artist_dict['tns'])
    if('alias' in artist_dict):
        artist_list.extend(artist_dict['tns'])
    
    return list(set(artist_list))

def match_163_search_result(name: str, duration: float, result: dict, 
                            artist: str | None = None, album: str | None = None,
                            name_weight: float = 0.7, album_weight: float = 0.2, artist_weight: float = 0.1,
                            full_match_weight: float = 0.2, duration_threshold: float = 15.0) -> float:
    '''
    根据歌曲名、艺术家、专辑返回匹配分数
    
    duration_threshold: 若时长相差超过该值，则强制返回0，单位为秒
    '''
    def fuzz_match(str1: str, str2: str):
        return fuzz.partial_ratio(str1, str2) * (1 - full_match_weight) + fuzz.ratio(str1, str2) * full_match_weight
    if(np.abs(result['dt'] / 1000 - duration) > duration_threshold):
        return 0.
    if('alia' in result):
        result_names = [result['name'], *result['alia']]
        score = {'name': np.max([fuzz_match(name, res_name) for res_name in result_names])}
    else:
        score = {'name': fuzz_match(name, result['name'])}
        
    if(artist is None):
        artist_weight = 0
        score['artist'] = 0
    else:
        result_artists = [parse_163_artist_dict(d) for d in result['ar']]
        result_artists = list(chain(*result_artists))
        score['artist'] = np.max([fuzz_match(artist, res_artist) for res_artist in result_artists])
    
    if((album is None) or ('al' not in result)):
        album_weight = 0
        score['album'] = 0
    else:
        result_albums = [result['al']['name'], *result['al']['tns']]
        score['album'] = np.max([fuzz_match(album, res_album) for res_album in result_albums])
        
    return (name_weight * score['name'] + artist_weight * score['artist'] + album_weight * score['album'])/(name_weight + artist_weight + album_weight)
    
    
def search_163_music(name: str,
                     duration: float,
                     artist: str | None = None,
                     album: str | None = None):
    url = SEARCH_163_API_THIRD.format(name = quote(name))
    response = requests.get(url,headers = HEADER, timeout = (5, 7))
    res = response.json()
    
    res_list = res['result']['songs']
    for r in res_list:
        r['match sore'] = match_163_search_result(name, duration, r, artist, album)
    res_list.sort(key = lambda r: r['match sore'], reverse = True)

    return res_list
    
def search_163_music_file(file: os.PathLike):
    audio = MP4(file)
    name = audio.tags['©nam'][0]
    artist = audio.tags['©ART'][0]
    album = audio.tags['©alb'][0]
    duration = audio.info.length
    return search_163_music(name, duration, artist, album)
