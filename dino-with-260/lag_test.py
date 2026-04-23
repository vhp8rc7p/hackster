# Source - https://stackoverflow.com/a/54596970
# Posted by Anton Panchishin
# Retrieved 2026-04-22, License - CC BY-SA 4.0

# !pip install image
# !pip install opencv-python
# !pip install pyscreenshot

import numpy as np
from time import time


resolutions = [
    (0, 0, 100,100),(0, 0, 200,100),
    (0, 0, 200,200),(0, 0, 400,200),
    (0, 0, 400,400),(0, 0, 800,400)
]


import numpy as np
import pyscreenshot as ImageGrab



def mss_test(shape) :
    average = time()
    import mss
    sct = mss.mss()
    mon = {"top": shape[0], "left": shape[1], "width": shape[2]-shape[1], "height": shape[3]-shape[0]}
    for _ in range(5):
        printscreen =  np.asarray(sct.grab(mon))
    average_ms = int(1000*(time()-average)/5.)
    return average_ms, printscreen.shape




def pil_test(shape) :
    average = time()
    from PIL import ImageGrab
    for _ in range(5):
        printscreen =  np.array(ImageGrab.grab(bbox=shape))
    average_ms = int(1000*(time()-average)/5.)
    return average_ms, printscreen.shape




def pyscreenshot_test(shape):
    average = time()
    import pyscreenshot as ImageGrab
    for _ in range(5):
        printscreen = np.asarray( ImageGrab.grab(bbox=shape) )
    average_ms = int(1000*(time()-average)/5.)
    return average_ms, printscreen.shape


named_function_pair = zip("mss_test,pil_test,pyscreenshot_test".split(","),
    [mss_test,pil_test,pyscreenshot_test])

for name,function in named_function_pair:
    results = [ function(res) for res in resolutions ]
    print("Speed results for using",name)
    for res,result in zip(resolutions,results) :
        speed,shape = result
        print(res,"took",speed,"ms, produced shaped",shape)

