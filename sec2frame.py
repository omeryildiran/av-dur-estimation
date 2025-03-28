def sec2frames(sec,frameRate=60):
    
    return round(sec*frameRate)

def frames2sec(frames,frameRate=60):
    return round(frames/frameRate,5)

#print(sec2frames(-0.025,120))