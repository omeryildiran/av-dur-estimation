"""
Docstring for article manuscript
This file will be used to write the manuscript for the article.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% ELIFE ARTICLE TEMPLATE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PREAMBLE 
\documentclass[9pt,lineno]{elife}
\usepackage{booktabs,tabularx}
\usepackage{threeparttable}

\usepackage[utf8]{inputenc}
% Use the onehalfspacing option for 1.5 line spacing
% Use the doublespacing option for 2.0 line spacing
% Please note that these options may affect formatting.
% Additionally, the use of the \newcommand function should be limited.

\usepackage{lipsum} % Required to insert dummy text
\usepackage[version=4]{mhchem}
\usepackage{siunitx}
\DeclareSIUnit\Molar{M}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% ARTICLE SETUP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\title{Duration judgments with conflicting audiovisual cues}

\author[1*]{Omer Faruk Yildiran}
\author[2]{Long Ni}
\author[3]{Michael S. Landy}
\affil[1]{New York University}

\corr{omer.yildiran@nyu.edu}{OFY}



\presentadd[\authfn{3}]{Psychology, New York University, NY, USA}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% ARTICLE START
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\begin{document}

\maketitle

\begin{abstract}

How does the brain integrate conflicting temporal information across sensory modalities? Using small conflicts (±50 ms), Hartcher-O’Brien et al. (2014) showed that observers integrate audiovisual duration cues optimally. Does causal inference lead to a breakdown of audiovisual integration when duration conflicts are large? Two levels of auditory signal-to-noise ratio (SNR) were used. We confirmed in a unimodal 2-IFC task that lower auditory SNR increased auditory duration discrimination threshold. Cross-modal duration discrimination revealed a consistent bias: auditory intervals were perceived as longer than physically matched visual intervals. This bias was taken into account in the main bimodal experiment. In the bimodal task, participants compared the auditory durations of two bimodal stimuli. Test stimulus: the two cues were consistent (same apparent duration). Standard stimulus: the two cues differed by seven conflict durations (range: ±250 ms). Auditory duration percepts shifted systematically toward the visual duration, especially when auditory uncertainty was high. The shift was proportional to the amount of cue conflict, inconsistent with the predictions of causal inference. We compared fits of several models. A heuristic model in which the observer probabilistically switches between the visual and auditory cues was preferred for most participants. However, AIC differences across models were small. Simulations revealed why the models were difficult to distinguish: Given the observed sensory noise, competing models can only be discriminated using unreasonably large conflicts. Within the reasonable conflict range used here, all models produced largely overlapping, near-linear patterns of shift as a function of the degree of cue conflict. In conclusion, observers do not rely on causal inference when judging durations under our conditions. The heuristic, cue-switching is favored, but the large uncertainty of duration estimates fundamentally limits our ability to discriminate computational models in this domain.


\end{abstract}


\newpage
\section{Introduction}

%1- General info about time
Accurately estimating the timing of events is essential for many everyday behaviors, from playing a musical instrument to catching a fly ball. Humans rely on multiple sensory modalities to perceive the timing and duration of events in the world. Prevailing models of timing mechanisms often rely on a clock analogy, particularly pacemaker–accumulator models \citep{gibbon1977scalar, treisman1963temporal}. These models propose that the brain contains an internal clock, where a pacemaker emits regular pulses that are accumulated over time to estimate duration. However, there is no concrete evidence for the existence of such a mechanism in the brain. While these models successfully explain timing properties such as Vierdot's law \citep{jazayeri2010temporal,lejeune2009vierordt,mamassian2010s}, which accounts for overestimation of short durations and underestimation of long durations, neural response patterns predicted by these models do not align well with actual neural data. 
Unlike vision or audition, however, time does not have a dedicated sensory receptor; rather, temporal perception is inferred based on signals elicited from a bunch of modalities.

% 2- Why multisensory and shortly saying what we have done
In natural environments, sounds and sights rarely occur in isolation; temporal judgments should often be made across modalities. This indirect encoding makes time perception inherently harder to investigate and more variable. The brain estimates time using cues embedded in visual, auditory, and tactile inputs, each with its own level of precision and bias. Even when stimuli are relatively clear, differences in the physical properties and neural conduction pathways of vision and audition create inherent temporal discrepancies. Light travels faster than sound, but paradoxically, auditory signals often reach the cortex more quickly than visual ones due to faster transduction and shorter conduction delays \citep{zoefel2017oscillatory}. Auditory stimuli typically arrive at central processing regions within 8–10 ms, whereas visual signals take approximately 20–40 ms to reach the cortex from the retina \citep{ king1985integration,vroomen2010perception}. In this study, we investigated how the brain processes duration perception when presented with conflicting information across modalities, specifically examining whether and how observers integrate auditory and visual timing cues under such conflict.

% 3- Why it is important
Despite the inherent difficulty of temporal estimation, the brain must operate with remarkable precision and consistency when timing our actions, which is a necessity for behaviors ranging from playing a musical instrument to catching a ball. Consider the act of driving: braking too early may leave you halted awkwardly in traffic, while braking too late could result in a collision, so timing is critical. Achieving this kind of temporal precision relies not only on motor coordination but also on the brain’s ability to accurately perceive and integrate temporal cues from multiple senses.

% 4 - Bayesian inference for integration 
When sensory information is available from multiple modalities, the brain faces the challenge of integrating these cues into a single percept. Research across diverse sensory systems has shown that this integration is not arbitrary: rather, the brain tends to weight each cue according to its reliability. This strategy—known as statistically optimal cue integration—helps minimize uncertainty in the final estimate, minimizing the total uncertainty of the fused percept \citep{ernst2002humans, maloney1989statistical, young1993perturbation, landy1995measurement}. This optimal cue integration strategy has been documented across a wide array of sensory tasks, including vision-touch integration for object slant estimation \citep{ernst2002humans, hillis2002combining}, taste-smell combination for flavor perception \citep{maier2020adaptive}, and audio-visual integration for spatial localization \citep{alais2004ventriloquist, negen2018bayes}. These findings underscore the generality of Bayesian cue combination as a core principle of sensory processing \citep{alais2019cue, landy2011ideal, landy2001ideal, rossi2021mechanisms, wolpert2011principles}, providing a robust normative framework for understanding multi-sensory perception.



% 5 - Not always fusion, starting causal inference
In many previous experiments, sensory cues across modalities were deliberately aligned in space and time, or only slightly mismatched, effectively biasing the brain toward assuming a single source. The introduction of low conflict conditions is a general way of examining how different cues are combined with respect to their reliabilities. Under such low‑conflict conditions, it is reasonable to assume a common cause and to apply statistical fusion (e.g., via MLE). However, everyday perception often presents greater cue disparities, whether from environmental factors (e.g., room acoustics), technological delays (e.g., audio–video lag), or neural processing differences. In those cases, the brain must address two interrelated questions: Are these disparate cues from the same source? And if so, how should they be combined? Rather than integrating across the board, the brain may dynamically shift from fusion to segregation depending on the degree of mismatch, an insight that has prompted the development of Bayesian causal-inference models. These models explain how integration diminishes as spatial or temporal disparity grows, as notably demonstrated in the ventriloquist-illusion paradigm \citep{kording2007causal, shams2005sound}. Under this account, integration is not automatic; instead, it is contingent on the inferred probability that the cues arise from a common cause \citep{badde2018vision, badde2021causal, badde2023multisensory, hong2021causal, kording2007causal, li2024uncertainty, li2025precision}. When this probability is high, the brain combines cues; when it is low, it treats them separately.

% 6- there are also neural evidence for causal inference
At the neural level, cortical hierarchies have been shown to implement integration or segregation differently across stages. Specifically, early sensory areas tend to process each cue independently, while posterior intraparietal regions integrate signals under a common-cause assumption. Crucially, only anterior intraparietal areas combine sensory estimates weighted by both cue reliability and the inferred likelihood of a shared cause, consistent with hierarchical Bayesian causal inference \citep{rohe2019neural}.

% 7 - Emphasizing the gap about duration and causal inference
While causal-inference models have been widely and successfully applied to spatial tasks, their application to temporal perception, particularly duration judgments, is far less explored. In essence, these models posit that perception emerges from a combination of common-cause inference and likelihood-based integration; that is, the brain infers whether different sensory cues share a source and, if so, integrates them accordingly.

% 8 - Duration integration is mainly investigated using MLE by hartcher
Indeed, \cite{hartcher2014duration} provided foundational evidence supporting this idea under low-conflict conditions: when audiovisual signals specify the same interval, observers combine duration information in a statistically optimal fashion. However, this study did not probe how the brain behaves when conflicts between modalities become substantial. In other words, even though perceptual integration aligns with MLE under modest disparities, we still lack an understanding of how high-disparity scenarios affect duration perception. This is the gap our study addresses.

% 9 - introducing our study what we have done
In the present study, we investigated how observers judge duration when presented with conflicting audiovisual(AV) cues. Specifically, we manipulated two key factors: the degree of cross-modal conflict (by varying the visual duration) and auditory reliability (by introducing high vs. low auditory-noise conditions). Although participants were explicitly instructed to base their judgments on the auditory signal, our design allowed us to test whether irrelevant visual information continued to influence their decisions and whether this influence varied as a function of the auditory signal-to-noise ratio.

By incorporating cross-modal bias correction, unimodal threshold estimation, and carefully controlled conflict conditions, our approach offers a strong test of whether observers flexibly integrate cues based on their inferred causal structure. In doing so, we extend causal-inference models to the domain of time perception, providing new insight into how the brain switch between integration and segregation when confronted with conflicting duration information.




\section{Results}

\subsection{Cue Reliability}

To characterize sensory precision, all participants completed two unimodal duration-discrimination tasks: one using auditory stimuli and the other using visual stimuli. Following each task, we fitted log-normal psychometric functions to each participant's data using maximum-likelihood estimation. The log-normal observer model is more appropriate for duration data than cumulative Gaussian functions because it naturally incorporates Weber's law of scalar variability behavior and gives zero probability for negative durations, consistent with scalar timing theory. The psychometric model included three key parameters: lapse rate ($\lambda$), representing attentional lapses; discrimination parameter ($\sigma$), reflecting the steepness of the psychometric curve in log-space and duration sensitivity; and bias parameter ($\mu$), representing systematic over- or under-estimation of durations in log-space. In the log-normal model, the probability of judging the test duration as longer is given by: $$P(\text{``test longer''}) = \frac{\lambda}{2} + (1-\lambda) \cdot \Phi\left(\frac{\ln(t/s) - \mu}{\sigma}\right), $$ where $t$ and $s$ are test and standard durations, respectively, and $\Phi$ is the standard normal cumulative distribution function. In the unimodal experiments the bias parameter ($\mu$) was fixed at zero because, in our two interval forced choice (2IFC) design with randomized interval presentation, participants could not systematically bias their responses toward one interval over the other. Both the test and standard intervals were auditory stimuli of the same reliability (both high-noise or both low-noise). Thus participants can not have any bias towards perceiving one standard/test longer. Any bias would affect both test and standard stimuli equally and thus it makes sense to fix $\mu$ to zero. 


\begin{figure}
    \centering
    \includegraphics[width=0.7\linewidth]{assets/figures/auditory and visual cue psychometric fits LOGNORMAL.png}
    \caption{\textbf{Unimodal visual and auditory duration-discrimination psychometric functions.} Probability of choosing the test interval as longer than the standard interval (500~ms) across three sensory conditions. Data points represent mean responses across 12 participants with error bars indicating standard error of the mean. Solid lines show fitted cumulative lognormal functions. Black curve: visual condition; blue: high-noise auditory condition; red: low-noise auditory condition. Discrimination precision varied substantially across modalities and noise levels ($\sigma_{\text{visual}} =$~194~ms, $\sigma_{\text{auditory\_high}} = $~359~ms, $\sigma_{\text{auditory\_low}} = $~86~ms), with the auditory low-noise condition yielding the highest precision and auditory high-noise condition showing the poorest discrimination performance.}
    \label{fig:unimodal data fits}
\end{figure}

The analysis revealed substantial effects of the signal-to-noise ratio of the auditory cue. The estimated noise parameter ($\sigma$) was significantly higher in the \textit{high-noise} auditory condition ($\sigma =$359~ms) compared to the \textit{low-noise} auditory condition ($\sigma = $86~ms), confirming that our background noise manipulation successfully reduced auditory reliability.

We also quantified these results by bootstrapping data and doing statistical analysis on the retrieved parameter estimates. We have resampled the data by randomly selecting  80\% of the data for 1000 times and fitting to the same psychometric function over again. This allowed us to have 1000 estimates for each parameter and we have performed an independent t-test on the bootstrap distributions of the $\sigma$ parameter and the results revealed a statistically significant difference between the two conditions ($t(1998) = 49.724, p <.0001$ ).


Visual reliability was not manipulated during the experiment. However, the visual psychometric function fits yielded a $\sigma$ value of 194~ms, indicating that visual duration sensitivity was worse than the low-noise auditory condition and better than the high-noise auditory condition. 


\subsubsection{Weber Fractions and Relative Precision}
Weber fractions, calculated as $\sigma$/standard duration, quantified the relative precision of duration discrimination across conditions. Auditory high-reliability conditions yielded the best performance (Weber fraction = 172~ms), while visual conditions showed intermediate precision (Weber fraction = 388~ms), and auditory low-reliability conditions produced the poorest performance (Weber fraction = 712~ms). 

These results confirm that the experimental design successfully created distinct reliability conditions across auditory and visual modalities, enabling us to assess how cue reliability influences integration behavior under audiovisual conflict.



\subsection{Modality-specific duration biases}

To estimate participants' inherent modality-specific biases in duration perception, we included a cross-modal duration-comparison task.

Psychometric fits revealed systematic biases in cross-modal judgments that depended on the auditory noise level (Figure~\ref{fig:crossModalData}). Under the low-noise condition, the PSE shift was $-105~ms$($\mu=-0.238$, indicating that the auditory interval had to be physically shorter than the visual interval to be perceived as equal in duration.

Under the high-noise condition, the PSE shift was $-77~ms$($\mu=-0.169$ , still indicating a visual bias but substantially reduced compared to the low noise condition, representing a 15.5\% underestimation. This suggests that while participants maintained a preference for auditory duration to be longer estimates, increasing the auditory noise reduced the magnitude of this crossmodal bias.

The precision of duration judgments, as reflected by the psychometric function slope ($\sigma$), also varied systematically with noise level. Low noise auditory signals yielded more precise judgments ($\sigma$ = 217~ms) compared to high noise signals ($\sigma$ = 388~ms), confirming that signal degradation impaired duration-discrimination performance. 

These results demonstrate that participants show reliable crossmodal biases in duration perception that are modulated by signal quality. There is a bias of perceived auditory relative to visual duration that was consistent across the two noise conditions. Thus, if we present both auditory and visual stimuli simultaneously in the bimodal experiment, with identical physical durations, participants should treat this as having a cue conflict (equal to the estimated bias). These individual bias estimates provide important baseline information for interpreting the main audiovisual-conflict experiment, where deviations from optimal integration may reflect both sensory uncertainty and inherent modality-specific biases.


\begin{figure}
    \centering
    \includegraphics[width=.8\linewidth]{assets/figures/cross modal result LOGNORMAL.png}
    \caption{ \textbf{Cross-modal duration-comparison psychometric functions.} Psychometric functions showing the probability of choosing the auditory test interval as longer than the visual standard interval (500 ms) as a function of test duration. Data points represent mean responses across 12 participants with error bars indicating standard error of the mean. Solid lines show fitted cumulative Gaussian functions. Blue line: low-noise condition (SNR = 0.1); Red line: high-noise condition (SNR = 1.2). The vertical dashed line indicates the standard duration (500 ms), while the horizontal dashed line marks chance performance (0.5). Points of subjective equality (PSE) were shifted leftward for both conditions (low noise: $\mu$ = -170; high noise: $\mu$ = -90), indicating a systematic bias for auditory duration estimates which is when both the auditory and the visual stimuli have the same duration auditory stimuli are perceived as 170~ms(low noise condition) or 90~ms(high noise condition) longer. Higher noise reduced both the magnitude of bias and discrimination precision ($\sigma_{\text{low}}$ = 0.46 vs $\sigma_{\text{high}}$ = 0.81).}
    \label{fig:crossModalData}
\end{figure}


\subsection{Main experiment: Cue-conflict effects}

In the main bimodal experiment, participants were asked to judge which of two intervals was longer based solely on the auditory stimulus. Despite this instruction, participants’ judgments were systematically influenced by the duration of the visual stimulus---especially under high auditory noise. 

To investigate how participants combined auditory and visual cues during duration judgments, we analyzed psychometric functions across varying levels of audiovisual conflict and auditory reliability.

Figure~\ref{fig:psychometric_conflict_fits} shows the full set of psychometric functions for each conflict level, separately for low (left) and high (right) auditory noise conditions. As visual duration increased, curves shifted horizontally toward longer durations, indicating that the irrelevant visual cue biased auditory judgments. This integration effect was more pronounced under high noise, where the auditory signal was less reliable, qualitatively consistent with reliability-weighted cue combination. Under low noise, the auditory signal dominated, and psychometric functions showed smaller shifts, reflecting a stronger reliance on the auditory modality.

\begin{figure}
    \centering
    \includegraphics[width=1\linewidth]{assets/figures/bimodal psychometric fits.png}
      \caption{\textbf{Psychometric functions for audiovisual duration judgments.} Each curve corresponds to one of seven audiovisual conflict levels ($-$250 to $+$250~ms). Dots show participant-level responses pooled across 12 participants, with marker size scaled to the number of trials. Under high auditory noise (right), psychometric curves shift progressively with conflict, indicating visual influence. Under low noise (left), shifts are minimal, reflecting auditory dominance.}
    \label{fig:psychometric_conflict_fits}
\end{figure}

To quantify the integration behavior, we extracted the PSE ($\mu$) from each fitted psychometric curve and plotted it as a function of conflict level (Figure~\ref{fig:conflict_pse}). Under high auditory noise, PSEs showed a nearly linear relationship with conflict. In contrast, under low noise, PSEs remained relatively stable near zero, indicating auditory dominance and resistance to visual bias. The error bars reflect the standard error from 100 bootstraps. For the bootstrapping we used a parametric sampling method where we generated simulation data using the fitted psychometric function parameters and fitted these simulation data again to the same psychometric function to get and save all 100 (the number of bootstraps) combinations of parameters to find the credible interval of a parameter fit.

We also quantified these observations using linear regression analysis, which revealed that visual conflict significantly influenced auditory duration perception in a noise-dependent manner. Under high auditory-noise conditions, visual conflicts showed a very strong positive correlation with PSE shifts ($r$ = 0.963, $p$ $<$ 0.001), with a steep slope of 0.606~ms PSE change per ms of visual conflict. In contrast, under low-noise conditions, the correlation remained strong ($r$ = 0.911, $p$ = 0.004), but had a much shallower slope (of 0.144~ms per ms conflict).

Together, these results demonstrate that audiovisual duration integration is flexible and reliability-dependent. When the auditory signal was degraded by noise, participants' judgments were systematically biased toward the visual cue. These behavioral patterns are consistent with predictions from Bayesian cue integration models that weight sensory evidence according to uncertainty.

This behavioral pattern aligns with optimal-integration theory in which perceived duration is computed as a reliability-weighted average of the measurements from different modalities. As such, in the low auditory-noise condition, participants had a very small bias towards the visual duration, when compared to the high auditory-noise condition where the participants were biased prominently towards the visual duration.

However, one essential aspect of the data is the hint of nonlinearity for the largest conflicts, especially when the visual duration was larger than the auditory duration. In the high auditory-noise condition, when the visual duration is longer than the auditory duration by 170~ms or more, the visual bias levels off. This non-linear trend is a hint of the signature of the effect of causal inference. That is, when the conflict is sufficiently large, the brain infers a low probability for these two stimuli to be coming from the same source. As a result, the brain gives a higher weight the auditory duration measurement, leading to reduced bias towards the visual duration. 

This behavioral pattern aligns with the principles of Bayesian causal inference: the brain integrates signals when they are similar and likely share a common cause, but segregates them when they differ too much.

We plan to model duration-discrimination using the Bayesian causal-inference model. We have not completed the modeling part of our study. In the next paragraphs, we will explain the main logic of our models and report preliminary modeling results here.


\begin{figure}
    \centering
    \includegraphics[width=1\linewidth]{assets/figures/bimodal pse shifts data.png}
    \caption{\textbf{Effect of audiovisual conflict on perceived auditory duration.} Group-averaged point of subjective equality (PSE) estimates as a function of audiovisual duration conflict, shown separately for high-noise (left) and low-noise (right) auditory conditions. Each point reflects the mean PSE across 12 participants; error bars show bootstrapped 95\% confidence intervals. Under high noise, auditory duration judgments are pulled toward the visual duration, particularly for moderate conflicts. For the largest conflicts, the bias saturates, consistent with cue segregation. Under low noise, PSEs remain stable across all conflict levels, suggesting auditory dominance.}
    \label{fig:conflict_pse}
\end{figure}


\subsection{Modeling results}

To test how well different observer models could explain participants' behavior in the audiovisual duration-conflict task, we fit six causal-inference models to the behavioral data. These models varied along two key dimensions: (1) the form of sensory noise and the noise assumed by the participant (Gaussian noise in linear time, log-space Gaussian noise, and a log-linear mismatch) and (2) observers integration strategy (Bayesian causal inference, forced-fusion or heuristic models). All of these models have three lapse-rate ($\lambda$) parameters (one per session), two the auditory duration-noise ($\sigma_{a}$) parameters (one for each auditory SNR),the visual duration-noise ($\sigma_{v}$) parameter.

\subsubsection{Observer measurement}

The first dimension of our model space concerns how auditory and visual durations are internally represented and corrupted by sensory noise. We considered three observer models that differ in the assumed form of sensory variability and in whether the observer’s inference mechanism correctly reflects the structure of that variability.

\paragraph{Gaussian observer (linear time).}
In the Gaussian observer, perceptual measurements of duration are assumed to be corrupted by additive Gaussian noise in physical time:
\begin{equation}
m_a \sim \mathcal{N}(S_a,\sigma_a^2), \qquad
m_v \sim \mathcal{N}(S_v,\sigma_v^2),
\end{equation}
where $S_a$ and $S_v$ denote the physical auditory and visual durations, and $\sigma_a$ and $\sigma_v$ are modality-specific noise parameters. This model implies duration-independent uncertainty and therefore does not implement scalar variability. Although this assumption is inconsistent with much of the timing literature, which shows that temporal uncertainty scales with duration \citep{haigh2021role,killeen1987optimal,brannon2008electrophysiological,walker1981auditory}, we include the Gaussian observer as a benchmark for comparison.

\paragraph{Log-space Gaussian observer (scalar variability).}
To capture Weber-like scaling of temporal uncertainty, the log-space Gaussian observer assumes that duration is encoded in logarithmic space and corrupted by additive Gaussian noise:
\begin{equation}
m_a \sim \mathcal{N}(\log S_a,\sigma_a^2), \qquad
m_v \sim \mathcal{N}(\log S_v,\sigma_v^2).
\end{equation}
Under this model, measurements in physical time are lognormally distributed, such that the standard deviation of perceived duration increases approximately proportionally with the true duration (scalar variability). Importantly, inference—including cue integration and causal inference—is performed in log-duration space, making the observer’s inference matched to the encoding process.

\paragraph{Log--linear mismatch observer.}
The log--linear mismatch model assumes the same sensory encoding as the log-space Gaussian observer, namely log-Gaussian measurements as in Eq.~(2). However, the observer’s inference system is mis-specified: instead of operating in log space, the observer exponentiates the measurements,
\begin{equation}
\tilde m_a = \exp(m_a), \qquad \tilde m_v = \exp(m_v),
\end{equation}
and treats the resulting quantities as if they were corrupted by additive Gaussian noise in linear time:
\begin{equation}
\tilde m_a \sim \mathcal{N}(S_a,\sigma_a^2), \qquad
\tilde m_v \sim \mathcal{N}(S_v,\sigma_v^2).
\end{equation}
Thus, although sensory encoding exhibits scalar variability, cue integration and causal inference are carried out as if duration were encoded linearly with constant variance. This model captures a plausible computational mismatch in which multiplicative sensory noise is not correctly taken into account during inference, without introducing additional free parameters or altering the structure of the causal inference model.



\subsubsection{Integration strategy}

Given the internal sensory noise raised from auditory and visual estimates, as second main deminsion our models differed in how they combined information across modalities and how they mapped internal estimates onto behavioral responses.  We considered five classes of integration and decision strategies, ranging from Bayesian causal inference to heuristic cue selection.

\paragraph{Bayesian causal inference.}



"""


