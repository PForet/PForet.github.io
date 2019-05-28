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

Ever been puzzled by the "dual" argument in scikit learn, or the dual gradient descent in some paper you were reading? Well even if you haven't been, having some knowledge of convex optimization is very valuable for a machine learning enthusiast. However, it usually overlooked in the online machine learning courses. But fear not, we have you covered! 

This article is the first of a two-part series on convex optimization applied to machine learning. This first post introduce the basics of convex optimization, only requiring some knowledge of algebra. The second one focus on duality (conjugates, lagrangian, dual problems).

 This series aims at covering the basic theory and can be seen as an introductory course that will give you all the tools needed to understand the algorithms that relie on convex optimization or duality. The focus on examples, illustrations and applications will help you to build the intuition necessary to put these new skills to good use.

**How to approach this** This is a long post! Don't hesitate to visit it several times, to take the time to redo the examples or to ask questions in the comments. You will benefit from it the most if you don't rush through. Convex optimization is a very powerful tool in machine learning, even for deep learning. Take the time to build an intuition of how it works and to recognize which problem would benefit from this approach.
{: .notice--warning}

<div class="notice--info">
<b> For later: </b> More elaborate applications will come later, for instance:
<ul>
  <li>Robustness to adversarial examples</li>
  <li>Training a model when the labels are noisy</li>
  <li>Integration of convex optimization problems in deep neural networks</li>
</ul>
This series will give you the tools to understand it all.
</div>

**Resources** If you are interested in convex optimization, the [Boyd](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) is probably one of the best books on the subject. It is in open access, and I'll reference it often during this series, so why not check it out?
{: .notice--info}

## Convex functions

### General definition of a convex function 

Let's start with the beginning and define what is a convex function. 

There are several ways of characterizing a convex function, but let's begin with the definition:

Lets consider the function $$f : R^n \to R$$. The function is said convex if and only if for any $$x$$ and $$y$$ in $$R^n$$ and any $$\theta \in [0,1]$$, we have:

$$ f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)$$

which means that the graph of $$f$$ is below the line segment between the points $$(x,f(x))$$ and $$(y,f(y))$$. 

On the other hand, a function $$g$$ is said to be concave if $$-g$$ is convex. Easy!

It is worth noting that a convex function does not have to be defined on $$R^n$$. In the previous definition, one can substitute $$R^n$$ by any convex set $$C$$, meaning any set that verifies the following condition: 

$$
\forall \,x, y \in C, \forall \,\, \theta \in [0,1],\, \theta x + (1-\theta)y \in C
$$

In most of machine learning applications, $$C$$ would represent all the possible values that the parameters can take. This is generally an interval of $$R^n$$, so this condition is not an issue in most of the cases.

### Rules on derivatives

**Functions that are differentiable once**

Now, if our function $$f$$ is differentiable (it has a gradient $$\nabla f(x)$$ everywhere), then it is convex if and only if:

$$f(y) \geq f(x) + \nabla f(x)^T(y-x)$$ 

[insert graph here]


**For functions that are differentiable twice**

If $$f$$ can be differentiate twice (meaning its Hessian $$\nabla^2 f(x)$$ exists everywhere), then the convexity condition boils down to:

$$\nabla^2 f(x) \geq 0$$

where the inequality should be understood component wise. 

[insert graph here with f, f' and f'']

### Some examples of convex and concave function

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

## Composing convex functions

### Composition rules

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

**Example 1**

Consider the following function:

$$h(z_1, ..., z_n) = \left(\sum_{i=1}^n max(0,z_i)^p \right)^{1/p}$$

for some $$p\geq 1$$. We know that the p-norm $$x\to (\sum_{i=1}^n  \mid x \mid^p)^{1/p}$$ is convex and non-decreasing on $$R^n_+$$ in each of its arguments. The function $$x \to max(0,x)$$ is convex, thus $$h$$ is convex

**Example 2**

Consider a linear regression that use the mean squared error as a loss. To calibrate the model, we want to find the weights $$W$$ and bias $$b$$ that minimize:

$$f(W,b) = \|WX - b\|^2$$ 

We know that the squared norm $$x\to \mid x \mid ^2$$ is convex. The function $$X,b \to WX - b$$ is linear, thus $$f$$ is convex. As a result, we can fit our linear regression on the dataset by solving this convex optimization problem. 


<div class="notice--warning">
Sometime these rules are not enough to show that a function is convex. For instance, consider the very useful log-sum-exp function:

$$f(x_1, x_2, ... x_n) = \ln(e^{x_1} + e^{x_2} + ... + e^{x_n})$$

The exponentials are all convex, the sum of convex functions is convex, but the logarithm is concave: we cannot apply the rules we just see. However, the log-sum-exp function is definitely convex: proving it just requires a little more work. 
</div>

### Other useful operations that preserve convexity

The following operations are very useful too to prove that a more complicated function is convex: we just have to show that we can 'build' it using convex functions, the rules that we just saw or these operations:

**Weighted sum**

If $$h_1, h_2, ..., h_k$$ are some convex functions, then the weighted sum

$$f(x) = \sum_{i=1}^k w_if_i(x)$$ 

is convex for any positive weights $$(w_i)_{i=1...k}$$

**Pointwise maximum**

If $$h_1, h_2, ..., h_k$$ are some convex functions, then their maximum for any point:

$$f(x) = \max_i f_i(x)$$

is convex too. 

**Partial minimization**

If $$h(x,y)$$ is convex in $$x$$ and $$y$$, then the solution to the minimization over $$y$$ is convex in $$x$$. In other words:

$$f(x) = \min_y f(x,y)$$

is convex. 


## Positive matrix and convexity 

Positive definite matrices plays a very important role in convex optimization and statistics. Let's start by defining what they are:

### Definition and convexity

Take a symmetric matrix $$M \in R^{n\times n}$$. We say that $$M$$ is positive definite ($$M\in S^n_{++}$$)if for any vector $$z$$, we have $$z^TMz > 0$$. We say that the matrix is only semi-definite ($$M\in S^n_{+}$$) if we have instead $$z^TMz \geq 0$$.  

These matrices are important for us, as the function:

$$x\to x^TQx$$

is convex if $$Q$$ is semi-positive definite (strictly convex if positive definite). Because the covariance matrix of a multivariate distribution is positive semi-definite, these matrices plays a key role in statistics.

### Eigenvalues

Another way to characterize these matrices is to say that a matrix is positive definite if and only if all of its eigenvalues are stricty positive (i.e $$\forall x, Qx = \lambda x \implies \lambda > 0$$). 
This is a very useful characterization of positive matrices. It also means that the log-determinant is well defined for positive definite matrices: it is the sum of the logarithms of the eigenvalues of the matrix:

$$ - \ln \det (X) = - \ln \prod_i \lambda_i = - \sum \ln \lambda_i$$

It is also written $$\ln \det (X^{-1})$$ sometimes. This function is convex on $$S^n_{++}$$, and very useful. It appears in the negative log-likelihood of a multivariate Gaussian distribution for instance.

**Proof of the equivalence:** Observe that if $$z$$ is an eigenvector of $$M$$, then $$z^TMz = z^T(\lambda z) = (z^Tz) \lambda > 0$$ with $$(z^Tz) > 0$$ so the eigenvalue $$\lambda$$ must be positive. On the other hand, we can write $$Q = \sum_i \lambda_i q_i^Tq_i$$ so that $$z^TQz = \sum_i \lambda_i (q_iz)^T(q_iz)$$ and if all the eigenvalues of $$Q$$ are positive, then $$z^TQz$$ is positive too.
{: .notice--info}

### Covariance matrices

Another useful fact is that for any matrix $$A \in R^{n\times m}$$, the matrix $$A^TA$$ is semi-definite positive (definite if $$A$$ is of rank $$m$$). This can be checked easily by noting that for any vector $$z$$ we have $$z^T(A^TA)z = (Az)^T (Az) = \|Az\|_2^2 \geq 0$$. As a result, the empirical covariance matrix $$X^TX$$ is semi-definite positive.

One quick example in finance: the function that associate a portfolio $$w$$ with its variance $$w^TX^TXw$$ is convex. 




**Are semi-definite matrix an issue?** Well, generally yes. If some matrix $$Q$$ is only semi-definite, then the function $$z\to z^TQz$$ is still convex, but not strictly convex. This means that the they is are several values of $$z$$ that minimize this function, which is bad if $$z$$ is some parameters of our model that we want to estimate! Moreover, these semi-definite matrices are often encountered when you dont have enough data. For instance, if you have more dimensions than observations in your data matrix $$X$$, the empirical covariance $$X^TX$$ will only be semi-definite. This happens quite a lot in finance, if you have 500 stocks but only 100 days of returns for instance. Definitely a problem if you want to optimize a portfolio!
{: .notice--warning}

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


Let's detail now some of the most common cases of convex optimization problems. For each type of problem, we provide a typical usecase that you might encounter "in real life".

### Linear program (LP)

The simplest case, where both the objective function and the constraints are linear:

$$\begin{array}{cl} & \min_x c^Tx \\
 s.t. & \left\{ \begin{array}{l}
 Ax = b \\ 
 Gx \leq h\end{array} \right. \end{array}$$

These problems can be solved very efficiently, and even a laptop can cruch them for thousands of variables and constraints easily. They have applications in operations research, signal processing, macro-economy, and in machine learning when $$l_1$$ norms are used. 

**Example: Linear regression with Mean Absolute Error loss**
Also called Least absolute deviations (LAD), is is in essence similar to a standard linear regression, but we use the $$l_1$$ norm (absolute error) instead of the $$l_2$$ norm (squared error) as our loss function. As a result, the optimization problem is written:

$$\min_w \| Xw - Y \|_1 = \min_w \sum_i |w^Tx_i - y_i|$$

Because we used the $$l_1$$ norm, we can express this problem as a linear program. It is easy to see that the solution to the problem above is also the solution of:

$$\begin{array}{cl} & \min_{w,\xi} \sum_i \xi_i \\
 s.t. & \begin{array}{lr}
|w^Tx_i - y_i| \leq \xi_i & i=1,...,N 
\end{array} \end{array}$$

And the constraints on the absolute value can be replaced by linear constraints: 

$$\begin{array}{cl} & \min_{w,\xi} \sum_i \xi_i \\
 s.t. & \left\{ \begin{array}{l}
 w^Tx_i - y_i \leq \xi_i \\ 
 -(w^Tx_i - y_i) \leq \xi_i \\
 \end{array} \right. \end{array}$$

### Quadratic programming (QP)

This type of problem deals with quadratic objective functions (of the form $$x\to x^TQx$$ with $$Q\in S^n_{++}$$) and linear constraints. Their standard form is:

$$\begin{array}{cl} & \min_x \frac{1}{2} x^TQx + c^Tx \\
 s.t. & \left\{ \begin{array}{l}
 Ax = b \\ 
 Gx \leq h\end{array} \right. \end{array}$$

These problems can be solved pretty efficiently too, although not as efficiently as linear programs.

**Example 1: Elastic net**

The elastic net is a linear regression with both an $$l_1$$ and a $$l_2$$ penalization. The loss is in this case the mean squared error. As a result, the training problem is:

$$\min_w \|Xw - Y\|_2^2 + \lambda_1 \|w\|_1 + \lambda_2 \|w\|_2^2$$

We can develop the first norm to make it clearer that we indeed have a quadratic objective function, and we do the same trick as above for the $$l_1$$ penalty. Thus the program becomes:

$$\begin{array}{cl} & \min_{w,\xi} w^T X^TX w - 2Y^TXw + Y^TY + \lambda_2\sum_i w_i^2 + \lambda_1\sum_i \xi_i \\
 s.t. & \left\{ \begin{array}{l}
 w \leq \xi \\ 
 -w \leq \xi \end{array} \right. \end{array}$$

Remember that the empirical covariance matrix $$X^TX$$ is positive definite, so $$w^T X^TX w$$ is quadratic. $$\sum_i w_i^2$$ is obviously quadratic too (can be written $$w^Tw$$), and the rest is linear. As a result, the elastic net training problem can be written as a quadratic programming problem.

**Example 2: Markowitz portfolio** 

You want to invest in a mix of $$n$$ assets, and you want the expected return of your investments to be greater than a certain level $$r$$. However you are not a gambler, and you want the variance of your returns to be as low as possible. If you have access to the expected returns $$\mu$$ of the stocks and their covariance matrix $$\Sigma$$, you then want to invest in the portfolio that solves:

$$\begin{array}{cll} & \min_w w^T\Sigma w & \textrm{(minimize the variance)}\\
 s.t. &  \mu^Tw \geq r & \textrm{(for a given minimum return)}\end{array}$$


**Example 3: Dual problem of a Support Vector Machine** 

The dual problem for training a support vector machine is also a quadratic program:

$$\begin{array}{cl} & \min_{\mu} \frac{1}{2} \|\sum_i\mu_i y_ix_i\|_2^2-\sum_i \mu_i \\
 s.t. & \left\{\begin{array}{lr}  
0 \leq \mu_i \leq C & \forall i \in [1,...,N]\\
\sum_i \mu_i y_i = 0 \\
\end{array}\right. \end{array}$$

But more on that later...

We will explain how to transform the original Hinge loss into this dual problem in the second part of this series
{: .notice--info}

### Second order cone programming (SOCP)

Second order cone programs might seem a little bit more esoteric at first glance, so let's analyze them step by step. Here, we focus on a linear objective function, and we can have some linear equality or inequality constraints. In fact, the only difference between SOCP and linear programs is that we constrain $$x$$ to lies in a $$l_2$$ norm cone (thus the name "second order cone"). To understand what this mean, let's define what a norm cone is:

A norm cone is a subset of $$R^{n+1}$$ of the form  $$\left\{(x,t)\mid \|x\| \leq t\right\}$$ for any norm $$\|.\|$$ that you can think of. Some common ones are the first order cone $$\left\{(x,t)\mid \sum_i \mid x_i \mid \leq t\right\}$$ for the $$l_1$$ norm and the second order cone $$\left\{(x,t)\mid \sum_i x_i^2 \leq t\right\}$$ for the $$l_2$$ norm. All norm cones are convex, and thus the second-order cone $$\left\{x\mid\|A_ix + b_i\|_2 \leq c_i^Tx + d_i\right\}$$ is convex too. Second order cone programs deals with this kind of feasible sets:

$$\begin{array}{cl} & \min_x c^Tx \\
 s.t.  & \left\{ \begin{array}{l}
 \|A_ix + b_i\|_2 \leq c_i^Tx + d_i, \, i=1,...,m \\ 
 Gx = h\end{array} \right. \end{array} $$

Notice that you can add some linear inequality constraints by taking $$A_i=0$$ and $$b_i=0$$ (thus ending up with $$0 \leq c_i^Tx + d_i$$) or some quadratic constraints by taking $$c_i=0$$ (and ending up with $$\|A_ix + b_i\|_2^2 \leq d_i^2$$).

**Example: Portfolio with controled risk**

Let's take again the portfolio example that we saw in the previous section. Maybe the concept of volatility seems a little bit too abstract, and you would prefer to control the risk of big losses instead. To do so, a solution is to optimize a portfolio to maximize the returns, while having a probability smaller than $$\alpha$$ of loosing more than a certain amount $$\eta$$. For instance, you might want to portfolio that earn as much as possible, while having less than 5% of chances of loosing 20% or more of your money. SOCP allows you to do so:

First, let's assume that the stock returns follows a (multivariate) normal distribution: $$r \sim N(\mu, \Sigma)$$ so that the returns of the whole portfolio is normal too: $$r^Tw \sim N(\mu^Tw, w^T\Sigma w)$$. The constraint "having a probability smaller than $$\alpha$$ of loosing more than a certain amount $$\mu$$" is translated into:

$$\begin{array}{lr} P\left(r^Tw \leq \eta\right) \leq \alpha & \textrm{(1)} \end{array}$$

We can express this probability using the cumulative density function of the normal distribution, $$\Phi: z\to \frac{1}{\sqrt{2\pi}}\int_{-\infty}^z e^{-x^2/2}dx$$:

$$\begin{array}{rl} \textrm{(1)} \iff & P\left(\frac{r^Tw - \mu^Tw}{\sqrt{w^T\Sigma w}} \leq \frac{\eta- \mu^Tw}{\sqrt{w^T\Sigma w}}\right) \leq \alpha \\
\iff & \Phi\left( \frac{\eta- \mu^Tw}{\sqrt{w^T\Sigma w}} \right) \leq \alpha \\
\iff & \eta \leq \mu^Tw  + \Phi^{-1}(\alpha)\sqrt{w^T\Sigma w}  \\
\iff & \eta \leq \mu^Tw  + \Phi^{-1}(\alpha)\|\Sigma^{1/2} w\|_2 \end{array}$$

For $$\alpha \leq \frac{1}{2}$$, we have $$\Phi^{-1}(\alpha)\leq 0$$ thus (1) is equivalent to a second order cone constraint! Our final optimization problem would look like:

$$\begin{array}{cl} & \max_x \mu^Tw \\
 s.t.  & \left\{ \begin{array}{l}
 \eta \leq \mu^Tw  + \Phi^{-1}(\alpha)\|\Sigma^{1/2} w\|_2 \\ 
\sum_i w_i = 1 \end{array} \right. \end{array}$$

**Square root of a matrix** The last ingredient in the previous reasoning was the fact that $$\sqrt{w^T\Sigma w} = \|\Sigma^{1/2} w\|_2$$. Indeed we have $$\|\Sigma^{1/2} w\|_2^2 = (\Sigma^{1/2} w)^T(\Sigma^{1/2} w) = w^T\Sigma w$$. $$\Sigma^{1/2}$$ exists because  $$\Sigma$$ is semi-definite positive: with the decomposition $$\Sigma = A^T \Lambda A$$ with $$\Lambda$$ being a diagonal matrix with positive coefficient, we have $$\Sigma^{1/2} = \sqrt{\Lambda} A$$, where the square root should be understood component-wise. 
{: .notice--info}

## Losses and maximum-likelihood


### The receipe

Now that we have a good understanding of convex optimization problems, it's time to show how to use them to solve machine learning problems. One standard receipe is as follows:
- Make some hypothesis on the distribution of what you want to predict. These distributions should be conditional on the inputs $$x_i$$ and some parameters $$\beta$$ (for instance, we can suppose that $$y\mid x, \beta \sim N(\beta^T x, \sigma^2)$$ for a linear regression with Gaussian noise).
- For a dataset $$D = \left\{ (X_1, y_1), (X_2, y_2), ..., (X_N, y_N)\right\}$$ which consists of $$N$$ independant observations, compute the joint probability of your dataset and your parameters :

$$ \begin{array}{r,l,l} 
P(D, \beta) & = P(D\mid\beta)P(\beta) & \textrm{Condition on the parameters} \\
& = P(y_1,...y_n \mid X_1,...X_n, \beta) P(\beta) \\
& = \left(\prod_{i=1}^N P(y_i \mid X_i, \beta)\right) P(\beta) & \textrm{By independance} \end{array} $$
 
- We want to find the parameters that maximize this probability. Equivalently, we want to minimize the negative logarithm of it:

$$\min_{\beta} \left\{\sum_{i=1}^N -\ln P(y_i \mid X_i, \beta) -\ln P(\beta)\right\}$$

This last expression is called the "loss function" that is to be minimized during training. Hopefully it might be convex and can be solved using some convex optimization algorithms. In this formulation, $$-\ln P(y_i \mid X_i, \beta)$$ is the loss of a training example, and $$-\ln P(\beta)$$ is some kind of regularization. 

Going from the probabilistic model to the optimization problem (by taking the negative log) or the other way around (by taking the negative exponential) is often very fruitful and can give a lot of insights about a model. Now let's move to some classic example to make it easier to grasp.

**Limits:** This method works great for a lot of simple models (linear, Poisson regressions, etc...), but it has some limits. For instance, maybe the optimization problem ends up not being convex. On the other and, some convex optimization problems like Support Vector Machines don't have such a probabilistic interpretation. 
{: .notice--warning}

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