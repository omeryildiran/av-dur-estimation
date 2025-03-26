
from sec2frame import sec2frames, frames2sec
from ohatcher_audio_gen import AudioCueGenerator
from generateAudio import generateAudioClass

# Initialize the stimulus component

#audio_cue_gen = AudioCueGenerator(sampleRate=sampleRate)

genAudio = generateAudioClass(sampleRate=sampleRate)

while not endExpNow and stopped_stair_count!=(len(all_staircases)):
    """ Prepare trial routine""" 
    #region [rgba(0, 30, 10, 0.200)] # reddish for preparation
    print(f"Trial {trialN} out of {total_trial_num}")

    def chose_stair(stair_n=0):
        tmp_stair=all_staircases[stair_n]#np.random.choice(all_staircases)
        if tmp_stair.stair_stopped==False:
            if tmp_stair.method!="lapse_rate":
                return tmp_stair
            else:
                return tmp_stair
        else:
            return chose_stair((stair_n+1)%len(all_staircases))


    stair=chose_stair(trialN%len(all_staircases))
    if trialN%len(all_staircases)==0:
        shuffle(all_staircases)
    current_stair=stair.method

    # Get the current trial
    if current_stair=="lapse_rate":
        if not lapse_ended:
            if lapse_rate_conds.shape[0]>0:
                standardDur=lapse_rate_conds[0][0]
                riseDur=lapse_rate_conds[0][1]
                deltaDurPercent=lapse_rate_conds[0][2]
                lapse_rate_conds=lapse_rate_conds[1:]
            else:
                stopped_stair_count+=1
                lapse_ended=True
                stair.stair_stopped=True
                print('lapse rate ended')
                continue
        else:
            continue

    else:
        # Pop the last elemnt and assign it to the current trial
        standardDur = standardDurs.pop()
        riseDur = riseDurs.pop()

        deltaDurPercent = round(stair.next_trial(),4) # delta dur in terms of percentage of the standard duration (0.1, 0.2, 0.3, 0.4, 0.5)

    deltaDurS= round(standardDur*deltaDurPercent,4)  # delta dur in terms of seconds
    testDurS = standardDur + deltaDurS


    deltaDurs[trialN] = deltaDurPercent
    testDurs[trialN] = testDurS
    realDeltaDurs[trialN] = deltaDurS
    conflictDur=conflictDurs[trialN]    
    conflictDurHalf=conflictDur/2

    # Assign values calculate directly now
    order = int(np.random.choice([1,2])) # or orders[trialN] # presentation order of test. 1: comparison first, 2: comparison second
    preDur = np.random.uniform(0.4, 0.65)
    postDur = np.random.uniform(0.4, 0.65)
    isiDur = np.random.uniform(0., 0.9)

    preDurFrames=sec2frames(preDur, frameRate)
    postDurFrames=sec2frames(postDur, frameRate)
    isiDurFrames=sec2frames(isiDur, frameRate)
    testDurFrames=sec2frames(testDurS, frameRate)
    standardDurFrames=sec2frames(standardDur, frameRate)
    riseDurFrames=sec2frames(riseDur, frameRate)
    deltaDurFrames=sec2frames(deltaDurS, frameRate)


    # In this specific experiment, test is audio and standard is visual
    # visual stimulus
    # so that if the test is in the first place, the standard is in the second place and visual should be in the second place
    # if the test is in the second place, the standard is in the first place and visual should be in the first place
    
    if order==1: # test(audio) in the first place visual stim is in the 2nd place
        # Test times
        onsetAud=preDurFrames
        offsetAud=onsetAud+testDurFrames
        # standard times
        onsetVisual=offsetAud+isiDurFrames
        offsetVisual=onsetVisual+standardDurFrames

    elif order==2: # test (audio) in the second place so visual stim is in the 1st place
        # standard times
        onsetVisual=preDurFrames
        offsetVisual=onsetVisual+standardDurFrames
        # Test times
        onsetAud=offsetVisual+isiDurFrames
        offsetAud=onsetAud+testDurFrames 

    print(f"Onset visual: {onsetVisual}, Offset visual: {offsetVisual}, Onset audio: {onsetAud}, Offset audio: {offsetAud}")

    #recalculate durations in seconds (frames to seconds)
    preDur=frames2sec(preDurFrames, frameRate)
    postDur=frames2sec(postDurFrames, frameRate)
    isiDur=frames2sec(isiDurFrames, frameRate)
    testDurS=frames2sec(testDurFrames, frameRate)
    standardDur=frames2sec(standardDurFrames, frameRate)
    riseDur=frames2sec(riseDurFrames, frameRate)
    deltaDurS=frames2sec(deltaDurFrames, frameRate)

    # audio stimulus (simple white noise with duration of testDurS)
    audio_stim = genAudio.generateNoise(dur=testDurS, noise_type='white')
    audio_stim_sound=sound.Sound(value=audio_stim, sampleRate=sampleRate, stereo=True)
    audio_stim_sound.setVolume(volume)
    # uncomment to see the plots
    t=np.linspace(0,len(audio_stim)/sampleRate,len(audio_stim))
    # plt.plot(t,audio_stim)
    # plt.show()

    # For testing purposes uncomment the following line
    if ExpTesting:
        audio_stim_sound=sound.Sound('A', sampleRate=sampleRate, stereo=False,secs=0.0001) 

    print(f'Current Stair: {current_stair}, Standard Dur: {standardDur}, Test Dur: {testDurS}, Rise Dur: {riseDur},Test in: {order} place,  Delta Dur: {deltaDurPercent},  deltaDurS: {deltaDurS}')

    total_dur_of_stim = preDur+testDurS+isiDur+standardDur+postDur
    totalDurFrames=sec2frames(total_dur_of_stim, frameRate)
    total_stim_durs.append(total_dur_of_stim) # save the total duration of the audio stimulus

    if ExpTraining and trialN>0:
        # draw correct or incorrect text
        if is_correct:
            feedback_text = "Correct!"
            color='green'
        else:
            feedback_text = "Incorrect!"
            color='red'
        
        feedback_text_comp = visual.TextStim(win, text=feedback_text, color=color, height=50)
        feedback_text_comp.draw()
        win.flip()
        # comment for testing
        core.wait(0.5) if ExpTesting==False else None


    trialN += 1
    # have a rest screen
    if trialN % 30 == 0 and trialN > 0:
        block_num = trialN // 30
        total_blocks = total_trial_num // 30
        rest_text = f"""You can have a break now.
        You are in block {block_num} out of {total_blocks}.
        Press any key to continue."""
        rest_text_comp = visual.TextStim(win, text=rest_text, color='white', height=30)
        rest_text_comp.draw()
        win.flip()
        # comment for testing
        event.waitKeys() if ExpTesting==False else None


    # Check if the experiment is over
    if endExpNow or event.getKeys(keyList=['escape']):
        core.quit()


    # clear the screen and wait for 100 ms
    # #time to record refresh rate
    # t0 = trialClock.getTime()
    # win.flip()
    # t1 = trialClock.getTime()
    # print(f"Time to flip: {t1-t0}")
    # print(f"Frame Rate: {1/(t1-t0)}")

    #core.wait(0.05)



    win.flip(clearBuffer=True)

   
    #endregion

    """ Run Trial Routine """
    continueRoutine = True
    tPreTrial = globalClock.getTime()
    frameStart = 0
    frameN = -1
    # draw the fixation cross
    core.wait(0.15) if ExpTesting==False else None
    # fixation.draw()    
    # win.flip()
    # core.wait(0.033) if ExpTesting==False else None
    # startEndAudioCue.play()
    
    # Timers just before the trial starts
    trialClock.reset()
    globalClock.reset()
    t_start=globalClock.getTime()

    while continueRoutine:

        frameN += 1
        t = trialClock.getTime()
        # draw the fixation cross
        # visual stimulus
        if frameN==onsetVisual:
            visualStim.setAutoDraw(True)
            tVisualStimStart = t
        elif frameN==offsetVisual:
            visualStim.setAutoDraw(False)
            tVisualStimEnd = t
        
        # audio stimulus
        if onsetAud==frameN:
            audio_stim_sound.play()
            tAudStart =t
            audio_stim_sound.status = STARTED
        elif offsetAud==frameN:
            #continue
            audio_stim_sound.stop()
            


        if audio_stim_sound.status == STARTED:
            if audio_stim_sound.isPlaying == False:
                tAudEnd = t
                tAudDurRecorded = tAudEnd - tAudStart
                audio_stim_sound.status = FINISHED


        
        # check if all the stimuli are finished
        if visualStim.status == FINISHED and audio_stim_sound.status == FINISHED and frameN > totalDurFrames:
            continueRoutine = False
                #audio_stim_sound.stop()
                #break
        


        # check for quit (typically the Esc key)
        if event.getKeys(keyList=["escape"]):
            visualStim.setAutoDraw(False)
            endExpNow = True
            break
        win.flip()

    print('trial ended')
    stair_num_reversal=stair.reversals
    stair_is_reversal=stair.is_reversal
    # endregion
    print(f"dur of visual stim: {tVisualStimEnd-tVisualStimStart}, dur of audio stim: {tAudDurRecorded}")
    # region [rgba(40, 10, 3, 0.80)]

    """ SAVE TRIAL DATA BEFORE RESPONSE"""
    # independent variables
    exp_data[trialN, 0] = standardDur
    exp_data[trialN, 1] = riseDur
    exp_data[trialN, 19] = conflictDur

    # free variables
    exp_data[trialN, 2] = order
    exp_data[trialN, 3] = preDur
    exp_data[trialN, 4] = postDur
    exp_data[trialN, 5] = isiDur
    exp_data[trialN, 6] = trialN


    # stair dependent variables
    exp_data[trialN, 7] = round(total_dur_of_stim,6)
    exp_data[trialN, 8] = deltaDurPercent
    exp_data[trialN, 9] = deltaDurS
    exp_data[trialN, 10] = testDurS
    exp_data[trialN, 11] = maxIntensityBurst

    # stair data
    exp_data[trialN, 12] = current_stair
    exp_data[trialN, 16]= stair_num_reversal # num of reversals
    exp_data[trialN, 17]= stair_is_reversal # is this trial a reversal (1: reversal 0: not reversal)



    # # recorded durations
    exp_data[trialN, 20] = tAudStart
    exp_data[trialN, 21] = tAudEnd
    exp_data[trialN, 22] = tAudDurRecorded

    exp_data[trialN, 23] = tVisualStimStart
    exp_data[trialN, 24] = tVisualStimEnd
    exp_data[trialN, 25] = tVisualStimEnd - tVisualStimStart




    # endregion

    # clear the screen
    fixation.setAutoDraw(False)
    waitingResponse = True
    # clear event buffer
    event.clearEvents(eventType='keyboard')
    responseKeys.clearEvents()
    # response timer
    t_start = globalClock.getTime()
    # Two interval forced choice response

    # Create a text stimulus with an audio symbol (Unicode)
    modalityPostCue=None #np.random.choice(['A','V'])
    exp_data[trialN, 26] = modalityPostCue

    if modalityPostCue=='A':
        postCue = audioIcon 
    elif modalityPostCue=='V':
        postCue = visualIcon 
    # RESPONSE TEXT
    response_text = "1st Longer (<-) vs 2nd Longer (->)"
    response_text_comp = visual.TextStim(win, text=response_text, color='white', height=30)

    # region [rgba(40, 10, 30, 0.30)]
    core.wait(0.15) if ExpTesting==False else None
    #noise_audio.play()
    


    """ Start the response routine """
    while waitingResponse and not endExpNow:
        response_text_comp.draw()
        #postCue.draw()

        win.flip()

        response = responseKeys.getKeys(keyList=['left', 'right'], waitRelease=False)
        class simKeys:
            def __init__(self, keyList, rtRange):
                self.name=np.random.choice(keyList)
                self.rt = np.random.choice(np.linspace(rtRange[0], rtRange[1])/1000)

        #  for testing purposesSimulate a key press
        if not response and ExpTesting:
            #fake a response responseKeys 
            response=[simKeys(keyList=['left', 'right'], rtRange=[200,1000])]


        # record the response
        if response:
            #noise_audio.stop()
            responses[trialN] = 1 if response[0].name=='left' else 2  # 1 for first longer, 2 for second longer
            is_correct=(testDurS>standardDur and responses[trialN]==order) or (testDurS<standardDur and responses[trialN]!=order) # 1 for correct, 0 for incorrect
            is_corrects[trialN] = is_correct
            print(f"Response: {response[0].name}, Order: {order}, is_correct: {is_correct}")
            response_rts[trialN] = globalClock.getTime() - t_start

            exp_data[trialN, 13] = responses[trialN] # response as num
            exp_data[trialN, 14] = is_correct 
            exp_data[trialN, 15] = round(response_rts[trialN],3)
            exp_data[trialN, 18] = response[0].name # response key as name



            # constrain the conditions matrix to the max current trial
            exp_data_saved = exp_data[:trialN+1, :]
            # save exp_data as DataFrame
            data_saved = pd.DataFrame(exp_data_saved, columns=[
                'standardDur', 'riseDur', 'order', 'preDur', 'postDur', 'isiDur', 'trial_num',
                'totalDur', 'delta_dur_percents', 'deltaDurS', 'testDurS', 'intensities',
                'current_stair', 'responses', 'is_correct', 'response_rts' , 
                'stair_num_reversal', 'stair_is_reversal', 'response_keys', 'conflictDur',
                'recordedOnsetAudioTest', 'recordedOffsetAudioTest', 'recordedDurAudioTest',
                'recordedOnsetVisualStandard', 'recordedOffsetVisualStandard', 'recordedDurVisualStandard',

                'modalityPostCueTest'
            ])            
            data_saved.to_csv(filename + '.csv')


            waitingResponse = False

            # update staircase
            stair.update_staircase(is_correct)

            if stair.stair_stopped:
                stopped_stair_count+=1

        if endExpNow or event.getKeys(keyList=["escape"]):
            endExpNow = True
            break

# endregion
# clear the screen
win.flip()

waitEndExp = True
# End of Experiment Routine
while waitEndExp:
    #save data
    # data.to_csv(filename + '.csv')
    # sio.savemat(filename + '.mat', {'data': conditions_matrix})
    # end of experiment text
    print("end of exp")
    end_text = """End of the experiment.
    Thank you for your participation. Press any key to exit."""
    end_text_comp = visual.TextStim(win, text=end_text, color='white', height=30)
    end_text_comp.draw()
    win.flip()
    event.waitKeys()
    waitEndExp = False
core.quit()





