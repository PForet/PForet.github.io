---
title: Convex optimization for machine learning
layout: single
header:
  overlay_image: /assets/images/head/head12.png
permalink: /unreacheable-3/
author_profile: True
comments: true
share: False
related: False
use_math: true
sitemap: false
toc: true
toc_label: "Content"
toc_icon: "angle-double-down"
---

# Convex optimization for machine learning

## Convex functions

### General definition of a convex function 

Several ways of characterizing a convex function:

Lets consider the function $$f : R^n \to R$$. The function is said convex if and only if for any $$x$$ and $$y$$ in $$R^n$$ and any $$\theta \in [0,1]$$, we have:

$$ f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)$$

which means that the graph of $$f$$ is below the line segment between the points $$(x,f(x))$$ and $$(y,f(y))$$. 

On the other hand, a function $$g$$ is said to be concave if $$-g$$ is convex. Easy!

It is worth noting that a convex function does not have to be defined on $$R^n$$. In the previous definition, one can substitute $$R^n$$ by any convex set $$C$$, meaning any set that verifies the following condition: 

$$
\forall \,x, y \in C, \forall \,\, \theta \in [0,1],\, \theta x + (1-\theta)y \in C
$$

In most of machine learning applications, $$C$$ would represent all the possible values that the parameters can take. This is generally an interval of $$R^n$$, so this condition is not an issue in most of the cases.

### For functions that are differentiable once

Now, if our function $$f$$ is differentiable (it has a gradient $$\nabla f(x)$$ everywhere), then it is convex if and only if:

$$f(y) \geq f(x) + \nabla f(x)^T(y-x)$$ 

[insert graph here]


### For functions that are differentiable twice

If $$f$$ can be differentiate twice (meaning its Hessian $$\nabla^2 f(x)$$ exists everywhere), then the convexity condition boils down to:

$$\nabla^2 f(x) \geq 0$$

where the inequality should be understood component wise. 

[insert graph here with f, f' and f'']

### Some examples of convex and concave function:

A lot of commonly used functions are convex or concave. For instance:

- $$f(x) = e^{ax}$$ for any real $$a$$ is convex
- $$f(x) = x^a$$ defined on $$R_{+}^*$$ is convex if $$a\geq 1$$ or if $$a\leq 0$$  
- $$f(x) = x^a$$ defined on $$R_{+}^*$$ is concave if $$x \in [0,1]$$
- $$f(x) = \mid x\mid^p$$ is convex on $$R$$ for $$p\geq 1$$
- $$f(x) = log(x)$$ is concave on $$R^*_+$$ (thus $$f(x) = -log(x)$$ is convex)
- Any norm is convex
- Max function: $$f(x_1, x_2, ... x_n) = \max_{i} x_i$$ is convex in $$R^n$$. 
- $$f(x,y) = x^2/y$$ is convex for $$x \in R$$ and $$y>0$$

The following functions are very important in machine learning as we will see:

- Entropy: $$f(x) = -x\ln x$$ is concave on $$R_+*$$
- Log sum exp: $$f(x_1, x_2, ... x_n) = \ln(e^{x_1} + e^{x_2} + ... + e^{x_n})$$ is convex on $$R^n$$ 

It is a good idea to try to prove the convexity of these functions using the conditions stated above, to get better at recoginizing convex/concave functions. Some of the proofs of the convexity of these functions are available at the end of this post, and a lot more can be found page 73 of [the Boyd](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf). 

## Composition of convex and concave functions

Some rules allows us to compose convex functions to get more complex, but still convex, functions. This can be a little bit tricky: for instance, the composition of two convex functions are not necessary convex! However these rules are very helpful and are good to keep in mind. They are as follows:

Consider two functions $$h: R^m \to R$$ and  $$g: R^n \to R^m$$. Define $$f = h \circ g$$, meaning that for $$x \in R^n$$, we have 

$$f(x) = h(t_1, t_2, ..., t_k, ..., t_m)$$

with

$$\forall k \in [1,...,m],\, t_k = g_k(x_1, x_2, ..., x_i, ..., x_n)$$

Then $$f$$ is convex if $$h$$ is convex AND for all $$k \in [1,...,m]$$, one of the following is true:
- $$g_k$$ is affine.
- OR: $$g_k$$ is convex and $$h$$ is non-decreasing in argument $$k$$. 
- OR: $$g_k$$ is concave and $$h$$ is non-increasing in argument $$k$$. 

On the other hand, $$f$$ is concave if $$h$$ is concave AND for all $$k \in [1,...,m]$$, one of the following holds:
-  $$g_k$$ is affine.
- OR: $$g_k$$ is concave and $$h$$ is non-decreasing in argument $$k$$. 
- OR: $$g_k$$ is convex and $$h$$ is non-increasing in argument $$k$$. 

A function that is _non-decreasing in argument $$k$$_ simply means that the function is non-decreasing when the $$k^{th}$$ argument changes, for any possible values of the other argument. For instance, $$f(x_1, x_2) = e^{x_1+x_2}$$ is non-decreasing is both arguments: for any fixed value of $$x_1$$, the function $$t\to f(x_1, t)$$ is increasing, and vice-versa. 

### Example 1

Consider the following function:

$$h(z_1, ..., z_n) = \left(\sum_{i=1}^n max(0,z_i)^p \right)^{1/p}$$

for some $$p\geq 1$$. We know that the p-norm $$x\to (\sum_{i=1}^n  \mid x \mid^p)^{1/p}$$ is convex and non-decreasing on $$R^n_+$$ in each of its arguments. The function $$x \to max(0,x)$$ is convex, thus $$h$$ is convex

### Example 2

Consider a linear regression that use the mean squared error as a loss. To calibrate the model, we want to find the weights $$W$$ and bias $$b$$ that minimize:

$$f(W,b) = \|WX - b\|^2$$ 

We know that the squared norm $$x\to \mid x \mid ^2$$ is convex. The function $$X,b \to WX - b$$ is linear, thus $$f$$ is convex. As a result, we can fit our linear regression on the dataset by solving this convex optimization problem. 

### When the rules are not enough

Sometime these rules are not enough to show that a function is convex. For instance, consider the very useful log-sum-exp function:

$$f(x_1, x_2, ... x_n) = \ln(e^{x_1} + e^{x_2} + ... + e^{x_n})$$

The exponentials are all convex, the sum of convex functions is convex, but the logarithm is concave: we cannot apply the rules we just see. However, the log-sum-exp function is definitely convex: proving it just requires a little more work. 


## Other useful operations that preserve convexity

The following operations are very useful too to prove that a more complicated function is convex: we just have to show that we can 'build' it using convex functions, the rules that we just saw or these operations:

### Weighted sum

If $$h_1, h_2, ..., h_k$$ are some convex functions, then the weighted sum

$$f(x) = \sum_{i=1}^k w_if_i(x)$$ 

is convex for any positive weights $$(w_i)_{i=1...k}$$

### Pointwise maximum

If $$h_1, h_2, ..., h_k$$ are some convex functions, then their maximum for any point:

$$f(x) = \max_i f_i(x)$$

is convex too. 

### Partial minimization

If $$h(x,y)$$ is convex in $$x$$ and $$y$$, then the solution to the minimization over $$y$$ is convex in $$x$$. In other words:

$$f(x) = \min_y f(x,y)$$

is convex. 


## Positive matrix and convexity 

[TODO]

## Convex optimization problems

Now that we have detailed how to recognize convex functions, let's explain what we are supposed to do with them. Convex optimization deals with the following kind of problems:

$$\begin{array}{cl} & \min_x f(x) 
\\ s.t.  & \left\{ \begin{array}{l}
 h_i(x) = 0, i=1,...,N_e \\ 
 g_i(x) \leq 0, i=1,...,N_i \end{array} \right. \end{array}$$

Where $$f$$ is called the objective function, $$h_i$$ are the equality contraints and $$g_i$$ are the inequality constraints. The functions $$f$$ and $$g_i$$ must be convex, and $$h_i$$ must be linear. In that case, the optimization problem is said to be convex and exhibit a lot of very interesting properties, as we will see. 

Sometime, the equality constraints are replaced by inequality constraints (as $$h_i = 0 \iff h_i \leq 0$$ and $$-h_i \leq 0$$, and if $$h_i$$ is linear, then both $$h_i$$ and $$-h_i$$ are linear thus convex). However, in this articles and the follow up, we will keep the formulation with the equality contraint. It is also possible to write the constraints in vector form, for instance $$Ax \leq b$$, with $$A \in R^{n\times m}, x\in R^m, b \in R^n$$. In that case, each coordinate of the vector $$Ax$$ should be less than $$b$$. 

For instance, the optimization problem for a Support Vector Machine is, in the simplest case (separable datapoints, hard-margin):

$$ \begin{array}{cl} & \min_{w,b} \|w\| \\
 s.t. & \,\, -y_i \times (w^Tx_i-b) \leq -1 \, ,\,  i=1,...,N \end{array}$$ 


Lets detail now some special cases of convex optimization problems, for which some efficient solutions exists:

### Linear program (LP)

The simplest case, for which very efficient algorithms exists:

$$\begin{array}{cl} & \min_x c^Tx \\
 s.t. & \left\{ \begin{array}{l}
 Ax = b \\ 
 Gx \leq h\end{array} \right. \end{array}$$


[exemple with label robustness ?]

### Quadratic programming (QP)

$$\begin{array}{cl} & \min_x \frac{1}{2} x^TQx + c^Tx \\
 s.t. & \left\{ \begin{array}{l}
 Ax = b \\ 
 Gx \leq h\end{array} \right. \end{array}$$

[example with least square]
[example with markov]

### Second order cone programming (SOCP)

$$\begin{array}{cl} & \min_x c^Tx \\
 s.t.  & \left\{ \begin{array}{l}
 \|A_ix + b_i\|_2 \leq c_i^T + d_i, \, i=1,...,m \\ 
 Gx = h\end{array} \right. \end{array} $$

 [demander à laurent pour un exemple]


## Losses and maximum-likelihood

### Exemple 1: Linear regression

Let's begin with one of the simplest model possible: the linear regression. The distribution that we assume is the following:

$$y_i | X_i \sim N(\beta^TX_i, \sigma^2)$$ 

meaning the output of our model is a linear combination of the features ($$\beta^TX_i$$) plus a Gaussian noise of fixed variance. Another way to write it would be:

$$\begin{array}{rl} y_i = \beta_1 x_{i,1} + ... + \beta_m x_{i,m} + \sigma^2 \epsilon & \textrm{with} & \epsilon \sim N(0,1) \end{array}$$

Now suppose that we have a dataset $$D = \left\{ (X_1, y_1), (X_2, y_2), ..., (X_N, y_N)\right\}$$ which consists of $$N$$ observations, with one observation consisting of a vector of features $$X_i$$ and a output value $$y_i$$. A key assumption is that all of these observations are independent and identically distributed. The fist step is to express the likelihood of our observations using our statistical model:

$$\begin{array}{lr} P(D\mid\beta, \sigma) = P\left(\cap_i (X_i, y_i) \mid\beta, \sigma\right) = \prod_i P(X_i,y_i\mid\beta, \sigma) & \textrm{by independance} \end{array}$$

Now for any datapoint $$(X_i,y_i)$$, we have:

$$P(X_i, y_i \mid \beta, \sigma) = P(y_i\mid X_i, \beta, \sigma)P(X_i) = \frac{1}{\sqrt{2\pi \sigma^2}}e^{-(\beta^TX_i - y_i)^2/2\sigma^2}\times P(X_i)$$

Here, we used another very useful hypothesis: $$P(X_i \mid \beta, \sigma) = P(X_i)$$. In plain english, this means that the probability of the features are independent of the parameters. The likelihood function associate the parameters $$(\beta, \sigma)$$ to the probability of the observations:

$$L(\beta, \sigma | D) = \prod_i \left( \frac{1}{\sqrt{2\pi \sigma^2}}e^{-(\beta^TX_i - y_i)^2/2\sigma^2}\times P(X_i) \right)$$

Applying the negative logarithm to the likelihood function allows us to derivate the loss. 

$$l(\beta, \sigma) = -\ln L(\beta, \sigma | D) = \sum_i \left( \frac{1}{2\sigma^2}(\beta^TX_i - y_i)^2 - \ln P(X_i) + \frac{1}{2}\ln(2\pi\sigma^2) \right)$$

A lot of simplifications happen here, because we are only interested in the terms that depend on the parameters $$(\beta, \sigma)$$. Everything else can be considered as a constant for the optimization problem and can be disregarded.

$$ \begin{array}{ll}\textrm{arg}\min_{\beta, \sigma} l(\beta, \sigma) & = 
\textrm{arg}\min_{\beta, \sigma} \sum_i \left( \frac{1}{2\sigma^2}(\beta^TX_i - y_i)^2 - \ln P(X_i) + \frac{1}{2}\ln(2\pi\sigma^2) \right) \\
& = \textrm{arg}\min_{\beta, \sigma} \sum_i \left( \frac{1}{2\sigma^2}(\beta^TX_i - y_i)^2 + \frac{1}{2}\ln(2\pi\sigma^2) \right)\end{array}$$ 

We observe that for any fixed $$\sigma$$, the optimization problem in $$\beta$$ does not depend on $$\sigma$$: 

$$
\textrm{arg}\min_{\beta} \sum_i \left( \frac{1}{2\sigma^2}(\beta^TX_i - y_i)^2 + \frac{1}{2}\ln(2\pi\sigma^2) \right) =  \textrm{arg}\min_{\beta} \sum_i (\beta^TX_i - y_i)^2
$$ 

Thus, if we are only interested in estimating $$\beta$$ and not $$\sigma$$, we must solve a simple convex (indeed, quadratic) optimization problem. To use a more 'machine learning' jargon, we fit the model by minimizing the loss, which is in our case the Mean Squared Error (MSE).

### Exemple 2: Linear regression with prior on the weights:

In the previous example, we made the implicite assumption that before seeing any data, all values of $$\beta$$ are equally likely. This happened when we used $$P(D\mid \beta, \sigma)$$ to compute the likelihood instead of $$P(D, \beta, \sigma)$$. In the case when we have no prior on the parameters (meaning all values of those parameters would be equally likely), this doesn't change the optimization problem:

$$\textrm{arg}\max_{\beta, \sigma}  P(D, \beta, \sigma) = 
\textrm{arg}\max_{\beta, \sigma}  P(D\mid \beta, \sigma)P(\beta, \sigma) = \textrm{arg}\max_{\beta, \sigma}  P(D\mid \beta, \sigma)$$ 

if we consider that $$P(\beta, \sigma)$$ is just a constant. However, it is sometime interesting to consider the case when we have some prior distribution for the parameter. For instance, to avoid overfitting (larges values of $$\beta$$ here), we can consider that each coordinate of beta follows a Gaussian distribution: $$\beta_p \sim N(0, 1/\lambda)$$ for $$p=1,..., K$$. As a result, very large values of $$\beta$$ would be very unlikely, especially if $$1/\lambda$$ is small. In this example, we take the same problem as before, focusing on $$\beta$$ (we don't care about $$\sigma$$), while having a Gaussian prior on $$\beta$$, meaning $$\beta_p \sim N(0, 1/\lambda)$$. The parameter $$\lambda$$ is considered as given: we will see that we can find it by cross-validation. Let's follow the same receipe as before: 

$$\begin{array}{ll}\textrm{arg}\max_{\beta}  P(D, \beta) & = \textrm{arg}\max_{\beta}  P(D\mid \beta)P(\beta) \\
& = \textrm{arg}\max_{\beta}  \prod_{i=1}^N \left( \frac{1}{\sqrt{2\pi \sigma^2}}e^{-(\beta^TX_i - y_i)^2/2\sigma^2}\times P(X_i) \right) \times \prod_{p=1}^K \frac{\lambda}{\sqrt{2\pi}}e^{-\lambda\beta_p^2/2} \\
& = \textrm{arg}\min_{\beta} \sum_{i=1}^N (\beta^TX_i - y_i)^2 + \sum_{p=1}^K \frac{\lambda}{2}\beta_p^2 \\
& = \textrm{arg}\min_{\beta} \sum_{i=1}^N (\beta^TX_i - y_i)^2 + \frac{\lambda}{2}\| \beta \|_2^2 \end{array}$$

We recognize the loss of a ridge regression! This shows a very important parallel between optimization and statistical modelling: adding a $$l^2$$ penalty on the coefficient in the optimization problem is equivalent to supposing a Gaussian prior on the coefficients in the description of the statistical model. In the same way, if you chose a Laplacian distribution as your prior on $$\beta$$ (with a density function $$x\to \frac{1}{2b}e^{-\mid x \mid / b}$$), you end up with a $$l^1$$ penalty in the loss function, which is now:

$$\textrm{arg}\min_{\beta} \sum_{i=1}^N (\beta^TX_i - y_i)^2 + \lambda\| \beta \|_1$$

We recognize the optimization problem of a lasso regression. All of these problems are convex (the objective function is a sum of convex functions)

### Example 3: Multinomial logistic regression 

Now let's move to classification problems. We assume the same setting as before, except that now $$y_i$$ is a label between $$1$$ and $$M$$. For each observation $$(X_i, y_i)$$, we define the 'score' of the $$j^{th}$$ class as $$\beta_j^TX_i$$. Thus, given some features $$X_i$$ and a set of parameters $$(\beta_1, ..., \beta_M)$$, we compute a score vector $$S_i = (\beta_1^TX_i, ..., \beta_M^TX_i)$$. Finally, we apply the softmax function on this score vector, to transform the score vector into a probability distribution: 

$$P(y_i = j | X_i, \beta) = \frac{\exp(\beta_j^TX_i)}{\sum_{k=1}^M \exp(\beta_k^TX_i)}$$

We can easily check that this defines a probability distribution, as $$\sum_{k=1}^M P(y_i = j \mid X_i, \beta) = 1$$ and $$P(y_i = j \mid X_i, \beta) \geq 0$$ for each coordinate. This is basically all we need to apply the receipe we just saw:

$$\begin{array}{ll} \textrm{arg}\max_{\beta_1, ..., \beta_M} P(D\mid \beta_1, ..., \beta_M) 
& = \textrm{arg}\max_{\beta_1, ..., \beta_M} \prod_{i=1}^N P(X_i, y_i\mid \beta_1, ..., \beta_M) \\
& = \textrm{arg}\max_{\beta_1, ..., \beta_M} \prod_{i=1}^N \prod_{k=1}^M P(y_i = k | X_i, \beta)^{1(y_i = k)} \\
& = \textrm{arg}\min_{\beta_1, ..., \beta_M} -\sum_{i=1}^N \sum_{k=1}^M 1(y_i = k) \ln P(y_i = k | X_i, \beta) \end{array}$$ 

Let's decompose this somewhat complicated notation. $$1(y_i = k)$$ is equal to $$1$$ if the $$i^{th}$$ observation is in the $$k^{th}$$ class and $$0$$ else, thus the second sum has only one non-zero element. We just sum over all the observations the negative logarithm of the predicted probability for the observation's true class. You are probably familiar with this loss, called the multiclass cross-entropy or the negative log likelihood. 

By plugging the predicted probabilities that we derived from the score functions, the optimization problem becomes: 

$$\begin{array}{ll} &\textrm{arg}\min_{\beta_1, ..., \beta_M} -\sum_{i=1}^N \sum_{j=1}^M 1(y_i = j) \ln \left(\frac{\exp(\beta_j^TX_i)}{\sum_{k=1}^M \exp(\beta_k^TX_i)}\right)  \\
= &\textrm{arg}\min_{\beta_1, ..., \beta_M} \sum_{i=1}^N \left[ \ln\left(\sum_{j=1}^M\exp(\beta_j^TX_i)\right) - \left(\sum_{j=1}^M 1(y_i = j)\times \beta_j^TX_i \right)\right] \end{array}$$

This is a convex optimization problem too (recognize that the log-sum-exp is convex and that the second term is simply linear). 


## Applied examples

TODO but what ? 