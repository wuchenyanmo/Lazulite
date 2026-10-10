import numpy as np
from lazulite.alignment.japanese.stages import FrameMap

class Backend:
    vocab={c:i+1 for i,c in enumerate('abcdefghijklmnopqrstuvwxyz')}
    blank=0
    vocabulary_size=27
    frames=FrameMap(16000,320,400)
    special_tokens=()
    description={'adapter':'fixture'}
    def __init__(self): self.calls=0
    def emissions(self,audio,**kwargs):
        self.calls+=1
        return np.full((self.frames.count(audio.samples.shape[1]),self.vocabulary_size),-np.log(self.vocabulary_size),np.float32)
