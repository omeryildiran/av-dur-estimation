# MAP vs BLS in our causal-inference duration model

This note explains the disagreement about the prior/modeling section in plain language.

The main issue is:

> When the model assumes a bounded uniform prior, should the segregated auditory estimate be written as the raw auditory measurement `m_a`, or as an estimated auditory duration `\hat{s}_a`?

The answer depends on whether the model uses a **MAP estimator** or a **BLS/posterior-mean estimator**.

---

## 1. The variables

In our model:

$$
m_a
$$

means the noisy internal auditory measurement.

$$
m_v
$$

means the noisy internal visual measurement.

These are not the true stimulus durations. They are noisy measurements inside the observer's brain.

The model assumes something like:

$$
m_a \sim \mathcal{N}(y_a, \sigma_a^2)
$$

where:

- `y_a` is the true auditory duration in log-duration space
- `m_a` is the noisy auditory measurement
- `\sigma_a` is auditory sensory noise

Similarly:

$$
m_v \sim \mathcal{N}(y_v, \sigma_v^2)
$$

---

## 2. What is a prior?

A prior is the observer's expectation about what durations are possible before seeing/hearing the stimulus.

In the causal-inference model, we use a **bounded uniform prior**, also called a **box prior**.

That means:

$$
p(y) =
\begin{cases}
\frac{1}{t_{\max}-t_{\min}}, & t_{\min} < y < t_{\max} \\
0, & \text{otherwise}
\end{cases}
$$

In words:

> The observer thinks all durations inside the allowed range are equally likely, and durations outside the range are impossible.

So the prior is flat inside the box and zero outside.

---

## 3. What is the posterior?

After the observer gets a noisy measurement `m_a`, the observer combines:

1. the measurement likelihood
2. the prior

to get the posterior:

$$
p(y_a \mid m_a) \propto p(m_a \mid y_a)p(y_a)
$$

In words:

> Given the noisy measurement, what true duration values are plausible?

Because the prior is a box prior, the posterior is basically a Gaussian centered near `m_a`, but cut off at the prior boundaries.

So if `m_a` is near the edge of the allowed range, part of the Gaussian gets chopped off.

---

## 4. The key question: how do we turn the posterior into one estimate?

The posterior is a whole distribution.

But the model needs one final estimate:

$$
\hat{s}
$$

There are different ways to choose this estimate.

The two relevant ones are:

1. **MAP**
2. **BLS / posterior mean**

This is where the PI/postdoc disagreement comes from.

---

## 5. MAP estimator

MAP means:

> Maximum a posteriori estimate.

In simple words:

> Choose the value where the posterior is highest.

So MAP asks:

> What is the single most likely duration after combining the measurement and prior?

Mathematically:

$$
\hat{s}^{MAP} = \arg\max_y p(y \mid m)
$$

For a bounded uniform prior, if the measurement `m_a` is inside the allowed range, the peak of the posterior is still at `m_a`.

Why?

Because the prior is flat inside the box. It does not pull the peak left or right. It only says values outside the box are impossible.

So under MAP:

$$
\hat{s}_{a}^{MAP} = m_a
$$

as long as `m_a` is inside the prior range.

This is why the PI says:

> A MAP estimate with a box prior will still be the same as the measurement.

That is correct for MAP.

---

## 6. BLS / posterior-mean estimator

BLS means:

> Bayesian least squares.

In this context, BLS usually means using the **posterior mean**.

In simple words:

> Average over all possible true durations, weighted by how likely they are under the posterior.

Mathematically:

$$
\hat{s}^{BLS} = E[y \mid m]
$$

This is different from MAP.

MAP uses the peak of the posterior.

BLS uses the mean of the posterior.

With a box prior, the posterior mean can differ from `m_a`, especially near the prior boundaries.

Example:

Suppose `m_a` is close to the lower boundary. The posterior Gaussian is centered near `m_a`, but the part below the lower boundary is cut off. This leaves more posterior mass above `m_a` than below it.

So the posterior mean shifts upward.

Therefore under BLS:

$$
\hat{s}_{a}^{BLS} \neq m_a
$$

near the boundaries.

This is what the postdoc means when they say:

> Because you assume a box prior, `\hat{s}_a` might not always be the same as `m_a`. If the measurement is close to the prior boundaries, the posterior will be substantially truncated.

That is correct for BLS/posterior mean.

---

## 7. Why both people are kind of right

The PI is right if our model uses a MAP estimate.

The postdoc is right if our model uses a BLS/posterior-mean estimate.

So the real question is not:

> Is `m_a` right or wrong?

The real question is:

> What estimator are we claiming the observer uses?

If we claim MAP, then using `m_a` is defensible.

If we claim BLS/posterior mean, then using `m_a` is not fully correct near the prior boundaries.

---

## 8. What our current model seems to do

Our appendix currently computes the posterior probability of common cause using the bounded uniform prior:

$$
p(C=1 \mid m_a,m_v)
$$

To compute that, we marginalize over possible true durations within the prior range.

So the prior matters for computing:

$$
p(m_a,m_v \mid C=1)
$$

and:

$$
p(m_a,m_v \mid C=2)
$$

But then the final estimate is written as:

$$
\hat{s}
=
p(C=1 \mid m_a,m_v)\hat{s}_{C=1}
+
(1-p(C=1 \mid m_a,m_v))m_a
$$

This means:

> In the separate-causes case, the observer uses the raw auditory measurement `m_a`.

That is equivalent to using a MAP estimate under a bounded uniform prior.

So the current model is basically a **MAP causal-inference model**.

---

## 9. Why the notation causes confusion

The equation currently says:

$$
(1-p(C=1 \mid m_a,m_v))m_a
$$

The postdoc wants:

$$
(1-p(C=1 \mid m_a,m_v))\hat{s}_a
$$

This is not just a notation issue.

It depends on what `\hat{s}_a` means.

If:

$$
\hat{s}_a = \hat{s}_{a}^{MAP}
$$

then:

$$
\hat{s}_a = m_a
$$

and the PI is right.

But if:

$$
\hat{s}_a = \hat{s}_{a}^{BLS}
$$

then:

$$
\hat{s}_a = E[y_a \mid m_a]
$$

and this is not always equal to `m_a`.

Then the postdoc is right.

---

## 10. Common-cause case

Under common cause, the model combines auditory and visual measurements:

$$
\hat{s}_{C=1}
=
\frac{J_a}{J_a+J_v}m_a
+
\frac{J_v}{J_a+J_v}m_v
$$

where:

$$
J_a = \frac{1}{\sigma_a^2}
$$

and:

$$
J_v = \frac{1}{\sigma_v^2}
$$

This is a reliability-weighted average.

If auditory noise is low, `J_a` is large, so the estimate is closer to `m_a`.

If visual noise is low, `J_v` is large, so the estimate is closer to `m_v`.

This formula is also MAP-like if we ignore boundary effects.

But with a bounded prior and BLS readout, the common-cause estimate would technically be the posterior mean of a truncated Gaussian, not exactly this simple weighted average near the boundaries.

---

## 11. Why the postdoc says the model is odd

The postdoc is noticing this:

1. We use a box prior to compute causal probabilities.
2. But then we use simple measurement-based estimates like `m_a`.

They are asking:

> Are we using the box prior only for computing `p(C=1)`, but not for computing the final duration estimate?

That can look inconsistent if the paper does not clearly say what estimator is being used.

But it is not necessarily wrong.

It is valid if we say:

> We use the bounded prior to compute causal-structure probabilities, and then use MAP estimates under each causal structure.

With MAP estimates, the bounded uniform prior does not change the estimate unless measurements fall outside the prior support.

---

## 12. Why Jazayeri & Shadlen matters

Jazayeri & Shadlen used a Bayesian model of interval timing where the prior affects estimates.

They found that human timing behavior was better explained by a Bayesian least-squares/posterior-mean type estimator than by a simple measurement-only estimator.

This matters because temporal perception often shows central-tendency effects:

> Short intervals are overestimated and long intervals are underestimated.

That kind of bias naturally comes from a prior combined with a BLS/posterior-mean estimate.

So the postdoc is saying:

> Since this is a duration-perception paper, and since prior-based BLS models have worked well in timing, maybe we should also try a BLS version of the causal-inference model.

That is a reasonable suggestion.

But it would be a new model variant, not just a small notation change.

---

## 13. What we should probably say in the manuscript

The safest fix is to make the estimator explicit.

Instead of writing only:

$$
\hat{s}
=
p(C=1 \mid m_a,m_v)\hat{s}_{C=1}
+
(1-p(C=1 \mid m_a,m_v))m_a
$$

we can write:

$$
\hat{s}^{MAP}
=
p(C=1 \mid m_a,m_v)\hat{s}_{C=1}^{MAP}
+
(1-p(C=1 \mid m_a,m_v))\hat{s}_{a,C=2}^{MAP}
$$

Then define:

$$
\hat{s}_{a,C=2}^{MAP}=m_a
$$

for measurements inside the prior range.

Then add:

> Under the bounded uniform prior, the MAP estimate equals the maximum-likelihood estimate for measurements within the prior support. Therefore, in the segregated case, the auditory estimate reduces to the auditory measurement, `m_a`. A posterior-mean/BLS readout with the same bounded prior would yield estimates that can differ from `m_a` near the prior boundaries; this alternative readout was not used in the present model.

This directly addresses both comments.

---

## 14. Plain-language summary

The prior does two possible jobs:

1. It helps decide whether the auditory and visual signals came from the same source.
2. It can also bias the final duration estimate.

Our current model definitely uses the prior for job 1.

Whether it uses the prior for job 2 depends on the estimator.

With MAP:

> The box prior usually does not shift the estimate, so `m_a` is okay.

With BLS:

> The box prior can shift the estimate, especially near boundaries, so `\hat{s}_a` should not simply be replaced by `m_a`.

So:

> Current model = defensible as a MAP model, but the manuscript should say this clearly.

And:

> BLS model = reasonable alternative worth testing, especially because timing papers like Jazayeri & Shadlen suggest posterior-mean estimators can explain temporal biases.

---

## 15. One-sentence answer to the disagreement

The PI is right that a MAP estimate with a box prior reduces to the measurement `m_a`, but the postdoc is right that a BLS/posterior-mean estimate with the same box prior would not necessarily equal `m_a`; therefore the manuscript needs to explicitly state that the current causal-inference model uses a MAP readout, or else implement and compare a BLS version.
