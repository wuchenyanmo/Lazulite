import json
import pytest
from lazulite.alignment.japanese.backend import describe_model, MMS_VOCAB
from lazulite.alignment.japanese.input import InputOptions, model_texts, prepare_input
from lazulite.models.japanese import JapaneseAlignmentRequest, JapaneseLine, JapaneseUnit


def test_model_inspection_is_lightweight_and_confines_weights(tmp_path):
    (tmp_path/'model.pt').write_bytes(b'not a serialized model')
    declaration=tmp_path/'alignment-model.json'
    declaration.write_text(json.dumps({'type':'torchaudio-mms-fa','weights':'model.pt'}))
    info=describe_model(tmp_path)
    assert info['columns']==28 and info['blank']==0 and info['frames'].count(400)==1
    assert list(MMS_VOCAB)==list("-aienoutsrmkldghybpwcvjzf'qx")
    for name in ('../outside.pt', '/absolute.pt', None):
        declaration.write_text(json.dumps({'type':'torchaudio-mms-fa','weights':name}))
        with pytest.raises(ValueError): describe_model(tmp_path)


@pytest.mark.parametrize('normalize,strip,expected',[(True,True,['ga','k','ka','', 'nei','lu']), (True,False,['ga','k','ka','','nei55','lu']), (False,True,['ｶﾞ','っ','か','ー','nei','lǜ']), (False,False,['ｶﾞ','っ','か','ー','nei55','lǜ'])])
def test_input_switches_and_context(normalize,strip,expected):
    assert model_texts(['ｶﾞ','っ','か','ー','nei55','lǜ'],InputOptions(normalize,strip))==expected


def test_mixed_input_mapping_and_unknowns():
    req=JapaneseAlignmentRequest((JapaneseLine('l','漢字', (JapaneseUnit('u',(0,2),'がnei214!','unknown','ruby'),)),))
    prepared=prepare_input(req,MMS_VOCAB,0,('-',),InputOptions(True,True))
    assert prepared['origins']=={'u/piece/0':'u','u/piece/1':'u'}
    assert prepared['request'].lines[0].units[0].reading_range==(0,2)
    assert prepared['texts']=={'u/piece/0':'ga','u/piece/1':'nei214!'}
    assert prepared['unknown_characters']['u/piece/1']==['!','1','2','4']
    assert model_texts(['nei214 hou55'],InputOptions(False,True))==['neihou']


def test_tone_stripping_does_not_treat_script_boundaries_as_whitespace():
    req=JapaneseAlignmentRequest((JapaneseLine('l','字',(JapaneseUnit('u',(0,1),'nei5か','unknown','ruby'),)),))
    p=prepare_input(req,MMS_VOCAB,0,options=InputOptions(True,True))
    assert list(p['texts'].values())==['nei5','ka']
    assert model_texts(['nei5か'],InputOptions(True,True))==['nei5ka']


def test_other_architectures_are_not_supported_models(tmp_path):
 from lazulite.alignment.japanese.backend import UnsupportedModel
 (tmp_path/'config.json').write_text('{"model_type":"bert"}')
 with pytest.raises(UnsupportedModel):describe_model(tmp_path)
