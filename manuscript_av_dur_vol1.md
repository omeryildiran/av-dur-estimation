# Out of sync: Duration judgments with conflicting audiovisual cues
This repository contains the code and data for the manuscript "Out of sync: Duration judgments with conflicting audiovisual cues". The manuscript explores how conflicting audiovisual cues affect duration judgments, revealing that participants' estimates are biased towards the duration of the visual stimulus when it conflicts with the auditory stimulus.

## Abstract
Abstract will go here. Following is placeholder abstract.

How does the brain compute the duration of events and integrate conflicting temporal cues from different modalities? To date, it is still an unresolved issue how our brain computes the duration of events and how the brain integrates events from different modalities.
Previous studies have shown that observers integrate duration information across modalities in a statistically optimal manner, weighting cues by their relative reliability. However, when there is strong conflict between modalities, it is still unclear what integration strategy the brain adopts.
In this research, we focus on this issue and specifically, we investigate the relation between the integration and observers’ beliefs about the common cause of the presented two modality cues.
In this research, we focus on this issue and specifically investigate the relationship between integration and observers beliefs about the common cause of the presented auditory and visual cues. Using both low and high auditory signal-to-noise ratios, we first estimated sensory noise in unimodal auditory and visual duration discrimination tasks. We then conducted a cross modal baseline experiment to measure and correct for modality-specific biases. These corrections were embedded into the main audiovisual experiment, which included seven visual conflict conditions ranging from $ - 250$ to $+250$ ms. Participants performed a two interval forced choice task, judging which interval was longer, with instructions to base their responses just on the auditory component.
We found that when there was no audiovisual conflict, the observed bias in the main experiment centered near zero, confirming the effectiveness of our cross-modal correction. Results showed that under high auditory noise, participants' duration judgments were more strongly biased toward the visual cue, consistent with integration. In contrast, when auditory noise was low, visual conflicts had minimal influence.
These findings suggest that audiovisual duration judgments flexibly depend on both sensory uncertainty and causal inference. Our results extend Bayesian models of multisensory integration into the temporal domain, highlighting how the brain dynamically adapts its integration strategy when confronted with modality-specific noise and cross-modal conflict.



## Methods:

\section{Methods and Materials}

\subsection{Subjects}
We recruited twelve participants for the experiment, two of them were the authors O.Y. and L.N., two of them were other lab members while the rest were partially naive participants who are not familiar with the research area but in general experienced on general psychophysics experiments. Testing was conducted in accordance with the ethical standards approved by the Institutional Review Board at New York University (protocol number FY2016–595). 
All participants provided informed consent prior to their involvement in the study and were compensated $\$15$ per hour for participation, except for the first author himself.

\subsection{Aparatus}

All the experiments were conducted on a 21-inch FD Trinitron CRT Sony GDM-5402 at 120Hz refresh rate with resolution of 800x600 pixel rate. We used low resolution as the resolution was not important for our experimental purpose and the visual stimulus was clearly visible white disk, so we freed up horizontal and vertical scan budget for higher refresh rates. The experiments were designed and executed using Python and mainly using the Python PsychoPy package and we did not used the Builder or Coder Gui (Peirce, 2007). Participants were positioned approximately 60 cm away from the computer screen, with their head stabilized by a chin rest.

\subsection{Stimuli}
The visual stimuli were a white disk, at the screen center added to a gray background. Size of the disk was 1.5 degrees of visual angle(deg) so the disk subtends to $\sim$3 deg diameter on the retina. Throughout one trial revolution disk was always displayed on the screen but in order to define the disk as visual signal, disk had to be filled white and when it is center is not filled(shown as only white circle) it is defined as visual background noise phase.

After trying with many different methods for creating an auditory stimulus, we found that the most effective and consistent way to manipulate perceived auditory duration noise was to use a method of adding increased white noise to the auditory signal which was similarly done by (Hartcher-O’Brien, 2014) and by Shi et al.,(2013) in which they used a pure tone on top of pink noise background which to manipulate the signal to noise ratio they just increased or decreased the intensity comparison signal (46 dB) compared to the standard signal (68 dB). 

In our case hovever, we have also differed the characteristics of the auditory signal and noise so to be sure that signal is a detectable/differentiable audio even when the noise is louder than the signal. 

To generate the auditory stimulus we generated a 48kHz white noise. Each event was a noise burst amplitude-modulated by a raised-cosine (cosine ramp up, flat peak, cosine ramp down). Rise/fall ramps are short (5 ms); peak duration equals the intended stimulus duration. Envelope scales baseline (1x) up to a peak set by an intensity parameter(1/0.1 high signal-to-noise ratio (SNR) or 1/1.2 low SNR). 

On each trial, we generated two full-length audio tracks at 48 kHz: a continuous background noise track and a signal track that spanned the entire trial. White noise served as the carrier. The signal track was assembled by concatenating pre-cue noise, a standard burst, ISI noise, and a test burst, ensuring temporal continuity across the trial. 
The pre-cue duration ranged from 0.20–0.45 s, the ISI from 0.40–0.90 s, and the post-cue duration from 0.20–0.45 s (uniformly sampled). 
The pre-cue , post-cue and ISI noise were all generated with the same intensity which was either 0.1 or 1.2 times the peak intensity of the signal burst, depending on the condition. After concatenating the audio array we have added very brief zero-padding "jitter" (0.0001s (0.1ms)) at the beginning and end of each audio array to prevent clicks at the start and end of the audio. 

Then the background noise was filtered with 10-610 Hz (4th order SOS badpass) and the signal was filtered with 150-775Hz (4th order SOS bandpass) to make the signal more detectable. Test and standard bursts are matched in variance (RMS) to equate overall level. The two tracks were then summed, and the final waveform is peak-normalized to ±1 before playback to ensure that the peak amplitude of the signal is always the same across trials.
Order is counterbalanced via standard(fixed to 0.5 s) and test (varyed based on participant response using staircase procedure) signal was randomized across trials.

On each trial, we generated two full-length audio tracks at 48 kHz: a continuous background noise track and a signal track that spanned the entire trial. White noise served as the carrier. The signal track was assembled by concatenating pre-cue noise, a standard burst, ISI noise, and a test burst, ensuring temporal continuity across the trial. Each burst was a noise segment amplitude-modulated by a raised-cosine envelope with 5 ms cosine onset/offset ramps; the flat peak matched the intended event duration. Pre-/ISI-/post-noise durations were drawn uniformly (pre: 0.20–0.45 s; ISI: 0.40–0.90 s; post: 0.20–0.45 s). Noise level per condition set the SNR: 0.1× (high SNR) or 1.2× (low SNR) relative to the signal burst peak amplitude; the same level was applied to the pre-/ISI-/post segments. After construction, the background track was band-pass filtered to 10–610 Hz and the signal track to 150–775 Hz (both 4th‑order SOS) to enhance signal detectability. Test and standard bursts were RMS-matched prior to mixing to equate overall level. The two tracks were then summed, peak-normalized to ±1, and padded with 0.1 ms zeros at the start and end to prevent clicks. Audio was presented in stereo.

The audio is presented in stereo. 

