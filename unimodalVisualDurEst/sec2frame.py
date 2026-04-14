def sec2frames(sec,frameRate=60):
    return round(sec*frameRate)

def frames2sec(frames,frameRate=60):
    return round(frames/frameRate,4)
