
from sec2frame import sec2frames, frames2sec
from ohatcher_audio_gen import AudioCueGenerator

# Initialize the stimulus component
sampleRate = 48000
audio_cue_gen = AudioCueGenerator(sampleRate=sampleRate)

while not endExpNow and stopped_stair_count!=(len(all_staircases)):
    """ Prepare trial routine""" 
    #region [rgba(1, 30, 1, 0.240)] # reddish for preparation
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

    print('\n')
    # Get the current trial
    if current_stair=="lapse_rate":
        if not lapse_ended:
            if lapse_rate_conds.shape[0]>0:
                standardDur=lapse_rate_conds[0][0]
                riseDur=lapse_rate_conds[0][1]
                conflictDur=lapse_rate_conds[0][2]
                deltaDurPercent=lapse_rate_conds[0][3]
                print(f'delta dur percent: {deltaDurPercent}')
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
    isiDur = np.random.uniform(0.5, 0.9)

    # audiovisual pse difference from bimodal experiment
    avPSEframes=sec2frames(avPSEseconds, frameRate)
    avPSEframesHalf=avPSEframes/2
    pseAdditionHalf=-1*signPSE*avPSEframesHalf


    preDurFrames=sec2frames(preDur, frameRate)
    postDurFrames=sec2frames(postDur, frameRate)
    isiDurFrames=sec2frames(isiDur, frameRate)
    testDurFrames=sec2frames(testDurS, frameRate)
    standardDurFrames=sec2frames(standardDur, frameRate)
    riseDurFrames=sec2frames(riseDur, frameRate)
    deltaDurFrames=sec2frames(deltaDurS, frameRate)
    conflictDurFrames=sec2frames(conflictDur, frameRate)
    conflictDurFramesHalf=sec2frames(conflictDurHalf, frameRate)



    print(f'conflict half {conflictDurFramesHalf}')


    
    if order==1: # test in the first place, visual stimulus 1 is the test
        # Test times
        onset1=preDurFrames +1*signPSE*avPSEframesHalf
        offset1=onset1+testDurFrames -1*signPSE*avPSEframesHalf
        # standard times +  ADD CONFLICT DURATION TO THE STANDARD DURATION
        onset2=offset1+isiDurFrames-conflictDurFramesHalf +1*signPSE*avPSEframesHalf # we subsctract the conflict duration half thus the test will start earlier if conflict is positive and later if conflict is negative
        offset2=onset2+standardDurFrames+conflictDurFramesHalf-1*signPSE*avPSEframesHalf

    elif order==2: # test in the second place, visual stimulus 2 is the test
        # standard times
        onset1=preDurFrames-conflictDurFramesHalf+1*signPSE*avPSEframesHalf # we subsctract the conflict duration half thus the test will start earlier if conflict is positive and later if conflict is negative
        offset1=onset1+standardDurFrames+conflictDurFramesHalf-1*signPSE*avPSEframesHalf
        # Test times
        onset2=offset1+isiDurFrames+1*signPSE*avPSEframesHalf
        offset2=onset2+testDurFrames-1*signPSE*avPSEframesHalf

    print(f'Onset1: {onset1}, Offset1: {offset1}, Onset2: {onset2}, Offset2: {offset2}')

    #recalculate durations in seconds (frames to seconds)
    preDur=frames2sec(preDurFrames, frameRate)
    postDur=frames2sec(postDurFrames, frameRate)
    isiDur=frames2sec(isiDurFrames, frameRate)
    testDurS=frames2sec(testDurFrames, frameRate)
    standardDur=frames2sec(standardDurFrames, frameRate)

    
    riseDur=frames2sec(riseDurFrames, frameRate)
    deltaDurS=frames2sec(deltaDurFrames, frameRate)
    conflictDur=frames2sec(conflictDurFramesHalf*2, frameRate)

    print('break')

    # audio stimulus
    audio_stim= audio_cue_gen.whole_stimulus(
            test_dur=testDurS, standard_dur=standardDur, noise_type='white',
            order=order, 
            pre_dur=preDur, post_dur=postDur, isi_dur=isiDur, 
            intensity=maxIntensityBurst, rise_dur=0.005, 
            intensity_background=riseDur)

    audio_stim_sound=sound.Sound(value=audio_stim, sampleRate=sampleRate, stereo=True)
    t=np.linspace(0,len(audio_stim)/sampleRate,len(audio_stim))

    # plt.plot(t,audio_stim)
    # plt.show()

    # For testing purposes uncomment the following line
    if ExpTesting:
        audio_stim_sound=sound.Sound('A', sampleRate=sampleRate, stereo=False,secs=0.0001) 



    print(f'Current Stair: {current_stair}, Standard Dur: {standardDur}, Test Dur: {testDurS}, Rise Dur: {riseDur},Test in: {order} place,  Delta Dur: {deltaDurPercent},  deltaDurS: {deltaDurS}')
    visualStim=visual.Circle(win, radius=visualStimSize, fillColor=True, lineColor='black', colorSpace='rgb', units='pix',
                        pos=(0, 0))
    visualStim.lineWidht=5

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

    # timers
    trialClock.reset()
    globalClock.reset()
    t_start=globalClock.getTime()

    win.flip(clearBuffer=True)

   
    #endregion

    """ Run Trial Routine """
    audio_stim_sound.volume = volume
    continueRoutine = True
    tPreTrial = globalClock.getTime()
    frameStart = 0
    frameN = -1
    visualStim.setAutoDraw(True)

    while continueRoutine:
        frameN += 1
        t = trialClock.getTime()
        # draw the fixation cross
        # Predur
        if frameN < preDurFrames:
            #visualStim.setAutoDraw(False)
            visualStim.fillColor = "gray"

        # First Stimulus onset1
        elif onset1== frameN:# before
            #visualStim.setAutoDraw(True)
            visualStim.fillColor = "black"
            tVisualStim1Start = t

        # Interstimulus interval
        elif offset1== frameN:#< onset2:
            #visualStim.setAutoDraw(False)
            visualStim.fillColor = "gray"
            tVisualStim1End = t
            

        # Second Stimulus onset2
        elif onset2==frameN:# < offset2:
            #visualStim.setAutoDraw(True)
            visualStim.fillColor = "black"
            tVisualStim2Start =t
        
        # Post Stimulus
        elif offset2==frameN:# < frameN<=totalDurFrames:
            #visualStim.setAutoDraw(False)
            visualStim.fillColor = "gray"
            tVisualStim2End = t
        
        elif frameN >= totalDurFrames:
            visualStim.setAutoDraw(False)
            visualStim.status = FINISHED
        # Audio stimulus
        # audio stimulus
        if audio_stim_sound.status == NOT_STARTED and t >= 0:
            audio_stim_sound.play()

            audio_stim_sound.status = STARTED
            t_start = globalClock.getTime()
        elif audio_stim_sound.status == STARTED:
            if audio_stim_sound.isPlaying == False:
                t_dur=t-t_start
                audio_stim_sound.status = FINISHED
        
        # check if all the stimuli are finished
        if visualStim.status == FINISHED and audio_stim_sound.status == FINISHED:
            continueRoutine = False
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
    exp_data[trialN,16]= stair_num_reversal # num of reversals
    exp_data[trialN,17]= stair_is_reversal # is this trial a reversal (1: reversal 0: not reversal)

    # recorded durations
    if order==1:# test in the first place, visual stimulus 1 is the test
        exp_data[trialN, 20] = round(tVisualStim1Start,4)
        exp_data[trialN, 21] = round(tVisualStim1End,4)
        exp_data[trialN, 22] = round(tVisualStim1End-tVisualStim1Start,4)

        exp_data[trialN, 23] = round(tVisualStim2Start,4)
        exp_data[trialN, 24] = round(tVisualStim2End,4)
        exp_data[trialN, 25] = round(tVisualStim2End-tVisualStim2Start,4)

    elif order==2: # test in the second place, visual stimulus 2 is the test
        exp_data[trialN, 20] = round(tVisualStim2Start,4)
        exp_data[trialN, 21] = round(tVisualStim2End,4)
        exp_data[trialN, 22] = round(tVisualStim2End-tVisualStim2Start,4)

        exp_data[trialN, 23] = round(tVisualStim1Start,4)
        exp_data[trialN, 24] = round(tVisualStim1End,4)
        exp_data[trialN, 25] = round(tVisualStim1End-tVisualStim1Start,4)




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
    modalityPostCue=np.random.choice(['A','V'])
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
        postCue.draw()

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
                'recordedOnsetVisualTest', 'recordedOffsetVisualTest', 'recordedDurVisualTest',
                'recordedOnsetVisualStandard', 'recordedOffsetVisualStandard', 'recordedDurVisualStandard','modalityPostCue'
            ])            
            data_saved.to_csv(filename + '.csv')

            # save as mat file with the same variable names
        # save as mat file with the same variable names
            sio.savemat(
                filename + '.mat', 
                {
                    'standardDur': exp_data_saved[:, 0],
                    'riseDur': exp_data_saved[:, 1],
                    'order': exp_data_saved[:, 2],
                    'preDur': exp_data_saved[:, 3],
                    'postDur': exp_data_saved[:, 4],
                    'isiDur': exp_data_saved[:, 5],
                    'trial_num': exp_data_saved[:, 6],
                    'totalDur': exp_data_saved[:, 7],
                    'delta_dur_percents': exp_data_saved[:, 8],
                    'deltaDurS': exp_data_saved[:, 9],
                    'testDurS': exp_data_saved[:, 10],
                    'intensities': exp_data_saved[:, 11],
                    'current_stair': exp_data_saved[:, 12],
                    'responses': exp_data_saved[:, 13],
                    'is_correct': exp_data_saved[:, 14],
                    'response_rts': exp_data_saved[:, 15],
                    'stair_num_reversal': exp_data_saved[:, 16],
                    'stair_is_reversal': exp_data_saved[:, 17],
                    'response_keys': exp_data_saved[:, 18],
                    'conflictDur': exp_data_saved[:, 19],

                    'recordedOnsetVisualTest': exp_data_saved[:, 20],
                    'recordedOffsetVisualTest': exp_data_saved[:, 21],
                    'recordedDurVisualTest': exp_data_saved[:, 22],
                    'recordedOnsetVisualStandard': exp_data_saved[:, 23],
                    'recordedOffsetVisualStandard': exp_data_saved[:, 24],
                    'recordedDurVisualStandard': exp_data_saved[:, 25]



                }
            )

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





