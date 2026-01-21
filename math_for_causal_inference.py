"""
Docstring for math_for_causal_inference

Doc modelling of causal inference mathematics.

We are trying to model the best possible model for audiovisual duration perception in case of conflicting cues informing different durations. In our experiment, we had two modality cues auditory and visual. Each of them elicits latent duration y. For simplicity we would like to describe the model in which the observer perceives the latent duration y with a normal gaussian noise and the prior over y is bounded by perceived minimum ($t_{min}$ ) and maximum durations of a stimulus interval ($t_{max}$). 

## 0: Generative model
## Step 0.1: Generative model for same source

```mermaid  
graph LR;  
id1((s_av))--> id((m_a))
id1((s_av))--> id2((m_v))
id((m_a)) --> idfus((av fused est))
id2((m_v)) --> idfus((av fused est))

```

where:
$$
p(m_{av}|s_{av})= N(m_{av},s_{av},\sigma_{av}^2)
$$

It is easy to solve however we are interested in the assumption of separate sources, because it is essentially a categorization problem.
## Step 0.2: Generative model for separate source

```mermaid  
graph LR;  
id1((s_av))--> sv((s_v))
id1((s_av))--> sa((s_a))
sa-->idma((m_a))
sv-->idmv((m_v))
idma((m_a))-->idmaf((aud est.))

```


## Step 0.3: Generative model for category (a step back)

```mermaid
graph LR;  
pc(c)-->pc1
pc(c)-->pc2
pc1((c=1))<-->id1((s_av))
id1--> id((m_a))
id1((s_av))--> id2((m_v))
id((m_a)) --> idfus((av fused est))
id2((m_v)) --> idfus((av fused est))

pc2((c=2))<-->sv((s_v))
pc2<-->sa
sa((s_a))
sa-->idma((m_a))
sv-->idmv((m_v))
idma((m_a))-->idmaf((aud est.))
idmaf--> pcavg(category avg)
idfus-->pcavg

```


- s_a, s_v: true durations of **auditory** and **visual** stimuli
- m_a, m_v: noisy **measurements** (in observer’s representation space)
- σ_a, σ_v: sensory noise standard deviations


$$
m_a \sim N(s_a,\sigma_a^2) $$
$$
m_v \sim N(s_v,\sigma_v^2) 
$$

# 1 - Bayesian Inference

## 1.1) Likelihood of common cause $p(m_a,m_v| C=1 )$

So if the the durations are coming from same source instead of $s_a$ and $s_v$ we can say that there is a single latent duration $y$. Thus the measurements would arise from normal disributions around y:

$$
m_a \sim N(y,\sigma_a^2) 
\tag 1{}$$
$$
m_v \sim N(y,\sigma_v^2) 
\tag 2
$$

If we assume “true” value y has a **flat prior** on $[ t_{min},  t_{max}]$:
$$
p(y) = \frac{1}{ t_{max}- t_{min}} 
$$
We want to find the likelihood of common causes we need a single integrand bounded between $\hat t_{min}$ and $\hat t_{max}$:
$$p(m_a, m_v \mid C=1) = \int_{ t{min}}^{ t_{max}} p(m_a \mid y) p(m_v \mid y) p(y) dy
\tag 3$$
When we take the prior part of the equation out of integral:

$$
p(m_a, m_v \mid C=1) = \frac{1}{ t_{max}- t_{min}} \int_{ t{min}}^{ t_{max}} p(m_a \mid y) p(m_v \mid y) dy
\tag 4
$$
We can write the likelihoods within the integral as gaussians:

$$
p(m_a, m_v \mid C=1) = \frac{1}{ t_{max}- t_{min}} \int_{ t{min}}^{ t_{max}} N(m_a;y,\sigma_a^2)N(m_v;y,\sigma_v^2) dy
\tag 5
$$
Because a Gaussian is symmetric in its arguments (mathematically ):
$$
N(m_a;y,\sigma_a^2)=N(y;m_a,\sigma_a^2) \tag 6
$$
it is identical to write the likelihood as:

$$
p(m_a, m_v \mid C=1) = \frac{1}{ t_{max}- t_{min}} \int_{ t{min}}^{ t_{max}} N(y;m_a,\sigma_a^2)N(y;m_v,\sigma_v^2) dy
\tag 7
$$
## 1.2 Product of two Gaussians in the **same variable**

Within the integrand is product of two gaussians in the same variable so let:
$$f(y) = \mathcal{N}(y; m_a, \sigma_a^2) \mathcal{N}(y; m_v, \sigma_v^2) = \frac{1}{2\pi \sigma_a \sigma_v} exp[-\frac{1}{2}(\frac{(y-m_a)^2}{\sigma_a^2}+\frac{(y-m_v)^2}{\sigma_v^2})]
\tag 8$$
We’re trying to rewrite that in the form of a **Gaussian in y times probably a a constant** that doesn’t depend on y.
When we write the exponent and complete the square 

$$

log(f(y))= -\frac{1}{2} [\frac{(y-m_a)^2}{\sigma_a^2}+\frac{(y-m_v)^2}{\sigma_v^2}]-\frac{1}{2}log(2\pi\sigma_a^2)-\frac{1}{2}log(2\pi\sigma_v^2)

\tag 9
$$
$$

 = -\frac{1}{2} \left( \left(\frac{1}{\sigma_a^2} + \frac{1}{\sigma_v^2}\right) y^2 - 2 \left(\frac{m_a}{\sigma_a^2} + \frac{m_v}{\sigma_v^2}\right) y + \left(\frac{m_a^2}{\sigma_a^2} + \frac{m_v^2}{\sigma_v^2}\right) \right) - \text{norm consts}, 


\tag {10}
$$
we can use precisions as:
$$
1/\sigma_a^2= J_a, 1/\sigma_v^2= J_v 

\tag{11}
$$
Complete the square for the quadratic in **y**:

$$ (J_a + J_v)(y^2 - 2\mu_c y + \mu_c^2) + \left(\frac{m_a^2}{\sigma_a^2} + \frac{m_v^2}{\sigma_v^2} - (J_a + J_v)\mu_c^2\right)

\tag {12}
$$
with:
$$ 
\mu_c = \frac{J_a m_a + J_v m_v}{J_a + J_v} = \frac{m_a\sigma_v^2 + m_v\sigma_a^2}{\sigma_a^2 + \sigma_v^2}, \quad \sigma_c^2 = \frac{1}{J_a + J_v} = \frac{\sigma_a^2 \sigma_v^2}{\sigma_a^2 + \sigma_v^2}

\tag {13}
$$


**Now simplify constant term:**

$$ \frac{m_a^2}{\sigma_a^2} + \frac{m_v^2}{\sigma_v^2} - (J_a + J_v)\mu_c^2 
\tag{14}
$$

from equation 14 substitute $\mu_c$ with 

$$ 
\mu_c = \frac{J_a m_a + J_v m_v}{J_a + J_v} = \frac{m_a/\sigma_a^2+m_v/\sigma_v^2}{J_a+J_v}

\tag {15}$$
so const term becomes:

$$
\frac{m_a^2}{\sigma_a^2} + \frac{m_v^2}{\sigma_v^2} - (J_a + J_v)\cdot \Bigl [\frac{m_a/\sigma_a^2+m_v/\sigma_v^2}{J_a+J_v} \Bigr ]^2=\frac{m_a^2}{\sigma_a^2} + \frac{m_v^2}{\sigma_v^2} - \frac{(m_a/\sigma_a^2+m_v/\sigma_v^2)^2}{J_a+J_v}

\tag {16}
$$
under common denominator $J_a+J_v$
$$
=J_am_a^2+ J_v{\sigma_v^2} - \frac{(m_a/\sigma_a^2+m_v/\sigma_v^2)^2}{J_a+J_v}= 


\frac{(J_a+J_v) (J_am_a^2+J_vm_v^2)- (J_am_a+J_vm_v)^2}{J_a+J_v}

\tag {17}
$$

expand the numerator

$$
[J_a^2m_a^2+ J_a J_v m_v^2+J_a J_v m_a^2+J_v^2m_v^2] - 
[J_a^2m_a^2+2J_a J_v m_a m_v + J_v^2m_v^2]

\tag{18}
$$
Now  $J_a^2m_a^2$  $J_v^2m_v^2$ cancel out to zero $0$ and we are left with:

$$
= J_a J_v m_v^2+J_a J_v m_a^2 -2J_aJ_vm_am_v = J_aJ_v(m_v^2+m_v^2-2m_am_v)=J_aJ_v(m_a-m_v)^2
\tag{19}

$$

Now lets take the $J_aJ_v(m_a-m_v)^2$ and substitute with the numerator in equation 17:

$$
\frac{J_aJ_v(m_a-m_v)^2}{J_a+J_v}=\frac{\frac{1}{\sigma_a^2\sigma_v^2} (m_a-m_v)^2}{\frac{1}{\sigma_a^2}+\frac{1}{\sigma_v^2}}=\frac{\frac{1}{\sigma_a^2\sigma_v^2} (m_a-m_v)^2}{\frac{\sigma_a^2+\sigma_v^2}{\sigma_a^2\sigma_v^2}}
\tag{20}
$$
There we have the final equation for the constant term:
$$
=\frac{(m_a-m_v)^2}{\sigma_a^2+\sigma_v^2} 

\tag{21}
$$

So the quadratic becomes:

$$-\frac{1}{2}\left[\left(J{a}+J_{v}\right)\left(y-\mu_{c}\right)^{2}+\frac{\left(m_{a}-m_{v}\right)^{2}}{\sigma_{a}^{2}+\sigma_{v}^{2}}\right]

\tag{22}$$
So:

$$\mathcal{N}\left(y; m_{a}, \sigma_{a}^{2}\right) \mathcal{N}\left(y; m_{v}, \sigma_{v}^{2}\right)
=\frac{1}{2 \pi \sigma_{a} \sigma_{v}} \exp \left[-\frac{1}{2}\left(J_{a}+J_{v}\right)\left(y-\mu_{c}\right)^{2}\right] \exp \left[-\frac{1}{2} \frac{\left(m_{a}-m_{v}\right)^{2}}{\sigma_{a}^{2}+\sigma_{v}^{2}}\right]

\tag{23}$$
And here the term in y is a Gaussian:

$$\exp \left[-\frac{1}{2}\left(\tau_{a}+\tau_{v}\right)\left(y-\mu_{c}\right)^{2}\right]=\sqrt{2 \pi \sigma_{c}^{2}} \cdot \mathcal{N}\left(y; \mu_{c}, \sigma_{c}^{2}\right)
\tag{24}$$
The second exponential (the constant term) is a **Gaussian in** $m_a-m_v$ with mean 0 and variance $\sigma_a^2+\sigma_v^2$
$$\exp \left[-\frac{1}{2} \frac{\left(m_{a}-m_{v}\right)^{2}}{\sigma_{a}^{2}+\sigma_{v}^{2}}\right]=\sqrt{2 \pi\left(\sigma_{a}^{2}+\sigma_{v}^{2}\right)}\cdot \mathcal{N}\left(m_{a}-m_{v} ; 0, \sigma_{a}^{2}+\sigma_{v}^{2}\right)

\tag{25}$$

When we combine the terms:
$$
= \frac{1}{2 \pi \sigma_{a} \sigma_{v}}\sqrt{2 \pi \sigma_{c}^{2}} \cdot  \sqrt{2 \pi\left(\sigma_{a}^{2}+\sigma_{v}^{2}\right)} \cdot
\mathcal{N}\left(y; \mu_{c}, \sigma_{c}^{2}\right)\cdot \mathcal{N}\left(m_{a}-m_{v} ; 0, \sigma_{a}^{2}+\sigma_{v}^{2}\right)
\tag{26}
$$
Lets handle the  left constant part:
$$
=\frac{1}{2 \pi \sigma_{a} \sigma_{v}}\sqrt{2 \pi \sigma_{c}^{2}} \cdot  \sqrt{2 \pi\left(\sigma_{a}^{2}+\sigma_{v}^{2}\right)} 
= \frac{2 \pi \cdot \sigma_c \cdot \sqrt{(\sigma_a^2+\sigma_v^2)}}{2 \pi \sigma_{a} \sigma_{v}}

\tag{27}
$$
substitute $\sigma_c$ :

$$
= \frac{\frac{\sigma_a\sigma_v}{\sqrt{(\sigma_a^2+\sigma_v^2)}}\cdot \sqrt{(\sigma_a^2+\sigma_v^2)}}{\sigma_{a} \sigma_{v}}=1

\tag{28}
$$
Now we are only left with
$$
=\mathcal{N}(m_a - m_v; 0, \sigma_a^2 + \sigma_v^2) \mathcal{N}(y; \mu_c, \sigma_c^2) 
\tag{29}
$$

So this gives us a new combined product of gaussian with $\mu_c$ and $\sigma_c$ and a constant term with respect to y which is becomes:

$$ \mathcal{N}(y; m_a, \sigma_a^2) \mathcal{N}(y; m_v, \sigma_v^2) = \mathcal{N}(m_a - m_v; 0, \sigma_a^2 + \sigma_v^2) \mathcal{N}(y; \mu_c, \sigma_c^2) 

\tag{30}
$$

$\mathcal{N}(m_a - m_v; 0, \sigma_a^2 + \sigma_v^2)$ is a **constant w.r.t. $y$** that measures how compatible the two measurements are (how close $m_a$ and $m_v$ are relative to their noise).

$\mathcal{N}(y; \mu_c, \sigma_c^2)$ is a **Gaussian in $y$** with the familiar precision-weighted mean and variance.
## 1.3)  Use the factorization inside the integral

So now we can also take the constant term out of integral in the likelihood:

$$
p(m_a, m_v \mid C=1) = \frac{1}{ t_{max}- t_{min}}N(m_a-m_v;0;\sigma_a^2+\sigma_v^2) \int_{\hat t{min}}^{\hat t_{max}} N(y;\mu_c,\sigma_c^2) dy
\tag {31}
$$


## 1.4) Evaluate the Gaussian integral over a finite interval

$$ \int_{t_{\min}}^{t_{\max}} \mathcal{N}(y; \mu_c, \sigma_c^2) dy = \Phi\left(\frac{t_{\max} - \mu_c}{\sigma_c}\right) - \Phi\left(\frac{t_{\min} - \mu_c}{\sigma_c}\right)

\tag {32}
$$
## 1.5) Final formulation for likelihood
$$p(m_a, m_v \mid C = 1) $$
$$ =\frac{1}{t_{\max} - t_{\min}} \mathcal{N}(m_a - m_v; 0, \sigma_a^2 + \sigma_v^2) \left[\Phi\left(\frac{t_{\max} - \mu_c}{\sigma_c}\right) - \Phi\left(\frac{t_{\min} - \mu_c}{\sigma_c}\right)\right]

\tag{33}
$$


----

# 2 - Separate causes C=2

## 2.1) Generative model under C=2

Now since we assume that there are two seperate sources we would have to estimate two latent duration for each ($y_a$ and $y_v$)

These latent durations $y_a$ and $y_v$ are independent of each other but both are bounded with same latent $t_{min}$, $t_{max}$ and they generate measurements from noisy gaussian distribution.
So auditory and visual distributions conditined on y would be:

$$m_a | y_a \sim N(y_a,\sigma_a^2) \tag 1$$
$$m_v | y_v \sim N(y_v,\sigma_v^2) \tag 2$$

Since they are bounded with same max min durations uniform prior densities would be:

$$p(y_a)=p(y_v)=\frac{1}{t_{max}-t_{min}} \tag 3$$ on  $t_{min}$, $t_{max}$ 


The **likelihood** of the two measurements under C=2 are two integrand of joint probabilities:

$$
p(m_a,m_v|C=2 ) = \int \int p(m_a|y_a)p(m_v|y_v)p(y_a)p(y_v)dy_ady_v
\tag 4
$$
## 2.2) Factor the double integral

Because the integrand separates into a function of $y_a$ times a function of $y_v$  and the domains (boundaries) are independent this gives us:


$$
p(m_a,m_v|C=2 ) = \Bigl[\int_{t_{\min}}^{t_{\max}} p(m_a|y_a)p(y_a)dy_a \Bigr]
\Bigl[ \int_{t_{\min}}^{t_{\max}} p(m_v|y_v)p(y_v)dy_v \Bigr]
\tag 5
$$


## 2.3) Evaluate each integrals 

Again we can use the symmetry of gaussians:

$$ \mathcal{N}(m_a; y, \sigma_a^2) = \mathcal{N}(y; m_a, \sigma_a^2) \tag 6 $$
Then:
$$ \int_{t_{\min}}^{t_{\max}} p(m_a \mid y_a) p(y_a) dy_a = \frac{1}{t_{\max} - t_{\min}} \int_{t_{\min}}^{t_{\max}} \mathcal{N}(y_a; m_a, \sigma_a^2) dy_a

\tag 7
$$

So $p(m_a,m_v|C=2 )$ becomes:
$$
p(m_a,m_v|C=2 )=  \frac{1}{t_{\max} - t_{\min}} \int_{t_{\min}}^{t_{\max}} \mathcal{N}(y_a; m_a, \sigma_a^2) dy_a 
\cdot \frac{1}{t_{\max} - t_{\min}} \int_{t_{\min}}^{t_{\max}} \mathcal{N}(y_v; m_v, \sigma_v^2) dy_v
$$

$$
 = (\frac{1}{ t_{max}- t_{min}})^2\Bigr [ \Phi(\frac { t_{max}-m_a}{\sigma_a})-\Phi(\frac { t_{min}-m_a}{\sigma_a})  \Bigl ]
\cdot\Bigr [ \Phi(\frac { t_{max}-m_v}{\sigma_v})-\Phi(\frac { t_{min}-m_v}{\sigma_v})  \Bigl ]


$$
$$
=\prod_{x\in (m_a,m_v)} \frac{1}{ t_{max}- t_{min}}\Bigr [ \Phi(\frac { t_{max}-x}{\sigma_x})-\Phi(\frac { t_{min}-x}{\sigma_x})  \Bigl ]

\tag 8
$$


# 3) Posterior probability of common causes

$$
p(C=1 | m_a,m_v) = \frac{p(m_a,m_v|C=1)p_c}{p(m_a,m_v|C=1)p_c+(1-p_c)p(m_a,m_v|C=2)}
$$

where $p_c$ is the p common

## 4) Estimate
### 4.1) Estimate of model averaging
$$

\hat S = p(C=1 | m_a,m_v).\hat S_{c=1}+(1-p(C=1 | m_a,m_v)).m_a

\tag 1
$$

Where $\hat S_{c=1}$ is the fused estimate:
$$
\hat S_{c=1} = \frac{\sigma_v^2}{\sigma_a^2+\sigma_v^2}.m_a+\frac{\sigma_a^2}{\sigma_a^2+\sigma_v^2}.m_v=\frac{J_a}{J_a+J_v}.m_a+\frac{J_v}{J_a+J_v}.m_v
\tag 2
$$

Where $J_a,J_v$ are the precisions

### 4.2) Estimate of probability matching
$$
\hat S = p(C=1 | m_a,m_v).\hat S_{c=1}+(1-p(C=1 | m_a,m_v)).\hat S_{c=2}
\tag 3

$$

## Where $\hat S_{c=2}$ is the separate estimate (auditory only in our case)
$$
\hat S_{c=2} = m_a
\tag 4

### 4.3) Estimate of model selection
$$
\hat S = \begin{cases}
\hat S_{c=1}, & \text{if } p(C=1 | m_a,m_v) > 0.5 \\
\hat S_{c=2}, & \text{otherwise}
\end{cases}
\tag 5

"""
