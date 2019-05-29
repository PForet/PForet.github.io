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

Having some knowledge of convex optimization is very valuable for a machine learning enthusiast. However, it usually overlooked in the online machine learning courses. But fear not, we have you covered! 

This article is the first of a two-part series on convex optimization applied to machine learning. This first post introduces the basics of convex optimization, only requiring some knowledge of algebra. The second one focuses on duality (conjugates, Lagrangian, dual problems).

 This series aims at covering the basic theory and can be seen as an introductory course that will give you all the tools needed to understand the algorithms that rely on convex optimization or duality. The focus on examples, illustrations and applications will help you to build the intuition necessary to put these new skills to good use.

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

<video muted autoplay loop>
  <source src="/assets/videos/convex/convex_1.mp4" type="video/mp4" width="320" height="240">
  Video unavailable on your browser.
</video>

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

where the inequality should be understood component-wise. 

[insert graph here with f, f' and f'']

### Some examples of convex and concave functions

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

It is a good idea to try to prove the convexity of these functions using the conditions stated above, to get better at recognizing convex/concave functions. Some of the proofs of the convexity of these functions are available at the end of this post, and a lot more can be found page 73 of [the Boyd](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf). 

## Composing convex functions

### Composition rules

Some rules allow us to compose convex functions to get more complex, but still convex, functions. This can be a little bit tricky: for instance, the composition of two convex functions are not necessary convex! However, these rules are very helpful and are good to keep in mind. They are as follows:

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

Consider a linear regression that uses the mean squared error as a loss. To calibrate the model, we want to find the weights $$W$$ and bias $$b$$ that minimizes:

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

Positive definite matrices play a very important role in convex optimization and statistics. Let's start by defining what they are:

### Definition and convexity

Take a symmetric matrix $$M \in R^{n\times n}$$. We say that $$M$$ is positive definite ($$M\in S^n_{++}$$)if for any vector $$z$$, we have $$z^TMz > 0$$. We say that the matrix is only semi-definite ($$M\in S^n_{+}$$) if we have instead $$z^TMz \geq 0$$.  

These matrices are important for us, as the function:

$$x\to x^TQx$$

is convex if $$Q$$ is semi-positive definite (strictly convex if positive definite). Because the covariance matrix of a multivariate distribution is positive semi-definite, these matrices play a key role in statistics.

### Eigenvalues

Another way to characterize these matrices is to say that a matrix is positive definite if and only if all of its eigenvalues are strictly positive (i.e $$\forall x, Qx = \lambda x \implies \lambda > 0$$). 
This is a very useful characterization of positive matrices. It also means that the log-determinant is well defined for positive definite matrices: it is the sum of the logarithms of the eigenvalues of the matrix:

$$ - \ln \det (X) = - \ln \prod_i \lambda_i = - \sum \ln \lambda_i$$

It is also written $$\ln \det (X^{-1})$$ sometimes. This function is convex on $$S^n_{++}$$, and very useful. It appears in the negative log-likelihood of a multivariate Gaussian distribution for instance.

**Proof of the equivalence:** Observe that if $$z$$ is an eigenvector of $$M$$, then $$z^TMz = z^T(\lambda z) = (z^Tz) \lambda > 0$$ with $$(z^Tz) > 0$$ so the eigenvalue $$\lambda$$ must be positive. On the other hand, we can write $$Q = \sum_i \lambda_i q_i^Tq_i$$ so that $$z^TQz = \sum_i \lambda_i (q_iz)^T(q_iz)$$ and if all the eigenvalues of $$Q$$ are positive, then $$z^TQz$$ is positive too.
{: .notice--info}

### Covariance matrices

Another useful fact is that for any matrix $$A \in R^{n\times m}$$, the matrix $$A^TA$$ is semi-definite positive (definite if $$A$$ is of rank $$m$$). This can be checked easily by noting that for any vector $$z$$ we have $$z^T(A^TA)z = (Az)^T (Az) = \|Az\|_2^2 \geq 0$$. As a result, the empirical covariance matrix $$X^TX$$ is semi-definite positive.

One quick example in finance: the function that associate a portfolio $$w$$ with its variance $$w^TX^TXw$$ is convex. 




**Are semi-definite matrix an issue?** Well, generally yes. If some matrix $$Q$$ is only semi-definite, then the function $$z\to z^TQz$$ is still convex, but not strictly convex. This means that there are several values of $$z$$ that minimize this function, which is bad if $$z$$ is some parameters of our model that we want to estimate! Moreover, these semi-definite matrices are often encountered when you don't have enough data. For instance, if you have more dimensions than observations in your data matrix $$X$$, the empirical covariance $$X^TX$$ will only be semi-definite. This happens quite a lot in finance: if you have 500 stocks but only 100 days of returns for instance. Definitely a problem if you want to optimize a portfolio!
{: .notice--warning}

## Convex optimization problems

Now that we have detailed how to recognize convex functions, let's explain what we are supposed to do with them. Convex optimization deals with the following kind of problems:

$$\begin{array}{cl} & \min_x f(x) 
\\ s.t.  & \left\{ \begin{array}{l}
 h_i(x) = 0, i=1,...,N_e \\ 
 g_i(x) \leq 0, i=1,...,N_i \end{array} \right. \end{array}$$

Where $$f$$ is called the objective function, $$h_i$$ are the equality constraints and $$g_i$$ are the inequality constraints. The functions $$f$$ and $$g_i$$ must be convex, and $$h_i$$ must be linear. In that case, the optimization problem is said to be convex and exhibit a lot of very interesting properties, as we will see. 

Sometime, the equality constraints are replaced by inequality constraints (as $$h_i = 0 \iff h_i \leq 0$$ and $$-h_i \leq 0$$, and if $$h_i$$ is linear, then both $$h_i$$ and $$-h_i$$ are linear thus convex). However, in this article and the follow-up, we will keep the formulation with the equality constraint. It is also possible to write the constraints in vector form, for instance $$Ax \leq b$$, with $$A \in R^{n\times m}, x\in R^m, b \in R^n$$. In that case, each coordinate of the vector $$Ax$$ should be less than $$b$$. 

For instance, the optimization problem for a Support Vector Machine is, in the simplest case (separable datapoints, hard-margin):

$$ \begin{array}{cl} & \min_{w,b} \|w\| \\
 s.t. & \,\, -y_i \times (w^Tx_i-b) \leq -1 \, ,\,  i=1,...,N \end{array}$$ 


Let's detail now some of the most common cases of convex optimization problems. For each type of problem, we provide a typical use case that you might encounter "in real life".

### Linear program (LP)

The simplest case, where both the objective function and the constraints are linear:

$$\begin{array}{cl} & \min_x c^Tx \\
 s.t. & \left\{ \begin{array}{l}
 Ax = b \\ 
 Gx \leq h\end{array} \right. \end{array}$$

These problems can be solved very efficiently, and even a laptop can crunch them for thousands of variables and constraints easily. They have applications in operations research, signal processing, macro-economy, and in machine learning when $$l_1$$ norms are used. 

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

A norm cone is a subset of $$R^{n+1}$$ of the form  $$\left\{(x,t)\mid \|x\| \leq t\right\}$$ for any norm $$\|.\|$$ that you can think of. Some common ones are the first order cone $$\left\{(x,t)\mid \sum_i \mid x_i \mid \leq t\right\}$$ for the $$l_1$$ norm and the second order cone $$\left\{(x,t)\mid \sqrt{\sum_i x_i^2} \leq t\right\}$$ for the $$l_2$$ norm. 

| First order cone | Second order cone |
|:--:|:--:|
|$$\left\{(x1,x2,t)\mid \, \mid x_1 \mid + \mid x_2 \mid \leq t\right\}$$ | $$\left\{(x1,x2,t) \mid \sqrt{x_1^2+x_2^2} \leq t\right\}$$ | 
|![l1]({{ "/assets/videos/convex/cone_l1.png" | absolute_url }}) | ![l2]({{ "/assets/videos/convex/cone_l2.png" | absolute_url }})|

All norm cones are convex, and thus the second-order cone $$\left\{x\mid\|A_ix + b_i\|_2 \leq c_i^Tx + d_i\right\}$$ is convex too. Second order cone programs deals with this kind of feasible sets:

$$\begin{array}{cl} & \min_x c^Tx \\
 s.t.  & \left\{ \begin{array}{l}
 \|A_ix + b_i\|_2 \leq c_i^Tx + d_i, \, i=1,...,m \\ 
 Gx = h\end{array} \right. \end{array} $$

Notice that you can add some linear inequality constraints by taking $$A_i=0$$ and $$b_i=0$$ (thus ending up with $$0 \leq c_i^Tx + d_i$$) or some quadratic constraints by taking $$c_i=0$$ (and ending up with $$\|A_ix + b_i\|_2^2 \leq d_i^2$$).

**Example: Portfolio with controlled risk**

Let's take again the portfolio example that we saw in the previous section. Maybe the concept of volatility seems a little bit too abstract, and you would prefer to control the risk of big losses instead. To do so, a solution is to optimize a portfolio to maximize the returns, while having a probability smaller than $$\alpha$$ of loosing more than a certain amount of $$\eta$$. For instance, you might want to create a portfolio that earns as much as possible, while having less than 5% of chances of losing 20% or more of your money. SOCP allows you to do so:

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

## Implementation

Now that we have described some convex optimization problem, let's see how to solve them using Python. There are several ways to do so, and each method has its advantages and its drawbacks. Some of them are flexible and easy to implement but don't scale very well, and others offer top performance but require the user to write the problem in a particular form. 

A good thing to keep in mind is that solving these problems relies on two different components:

- The python API you use to specify your problem and get your results (for instance, `cvxpy`)
- The solver which is called by the module to actually solve the problem (for instance, `ECOS`)

All solvers are not created equal. Some works for only a particular type of problem (LP solvers, QP solvers, SOCP solvers, etc...). Some are open sources, other only available under commercial licenses. Most of the time, the python API will try to find the solver the most adapted to the problem you are trying to solve, but for top performances, you might have to google your way around to find which solver is the most adapted to your problem. 

### CVXPY

[CVXPY](https://www.cvxpy.org/) is perfect for prototyping and small experiments. It is very user-friendly, and you can define your functions with a "numpy-like" syntax. For instance,  to fit a LASSO regression, we just have to define some losses [(code from cvxpy tutorial) ](https://www.cvxpy.org/examples/machine_learning/lasso_regression.html):

{% highlight python %}
import cvxpy as cp
import numpy as np

def squared_error(X, Y, beta):
    """We define the squared error here"""
    return cp.norm2(cp.matmul(X, beta) - Y)**2

def regularizer(beta):
    """For regularization, we add a l1 penalty"""
    return cp.norm1(beta)

def objective_fn(X, Y, beta, lambd):
    """The loss we want to minimize is the squared error + the penalty"""
    return squared_error(X, Y, beta) + lambd * regularizer(beta)

# Define the variable that will contain the parameters
beta = cp.Variable(n)

# The strength of the penalty
# We specify 'nonneg' because the penalty must be positive
lambd = cp.Parameter(nonneg=True)

# Our optimization problem is the minimization of the optimization function
problem = cp.Problem(cp.Minimize(objective_fn(X_train, Y_train, beta, lambd)))

# Call the solver here
problem.solve()

# To get the values of the parameters that minimize the loss
best_parameters = beta.value
{% endhighlight %}

As you can see, everything is quite intuitive. Constraints can be added in a very simple way too. The solver is chosen automatically and CVXPY will check if your problem is convex before solving it. All of this makes CVXPY a very good tool for fast prototyping. On the other hands, CVXPY doesn't scale very well sometimes, so if performances become an issue you might want to move to another method. But still, a great tool for experimenting!

You can see more machine-learning examples [in the tutorials](https://www.cvxpy.org/examples/index.html#) to get a hand on it. 

### CVXOPT

CVXOPT is more "low level" and requires more work from the user. For instance, solving a SOCP would look like this [code from here](https://cvxopt.org/userguide/coneprog.html#second-order-cone-programming): 


{% highlight python %}
from cvxopt import matrix, solvers
c = matrix([-2., 1., 5.])
G = [ matrix( [[12., 13., 12.], [6., -3., -12.], [-5., -5., 6.]] ) ]
G += [ matrix( [[3., 3., -1., 1.], [-6., -6., -9., 19.], [10., -2., -2., -3.]] ) ]
h = [ matrix( [-12., -3., -2.] ),  matrix( [27., 0., 3., -42.] ) ]
sol = solvers.socp(c, Gq = G, hq = h)
{% endhighlight %}

With CVXOPT, you really need to express the optimization problem in "standard form", meaning you must be able to write your problem in one of the forms that we saw in the "convex optimization problems" section. CVXOPT will then take the matrices that parametrize your problem as inputs. 

As a result, CVXOPT requires you to know exactly what type of problem you are solving (LP, QP, SOCP, etc...). In exchange, you can achieve peak performances!

### To summarize

For beginners and fast prototyping on small datasets, CVXPY is ideal. If you need better performances and you know how to express your problem in standard form, CVXOPT is the way to go! 

## What next?

Congrats for making it to the end! Skipping through definitions like we did can be tiresome, but we will use what we have learned here a lot later. Trust me, the payoff is huge! Next stop: **duality**