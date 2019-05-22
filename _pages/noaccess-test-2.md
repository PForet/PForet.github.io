---
title: Duality applied to machine learning
layout: single
header:
  overlay_image: /assets/images/head/head12.png
permalink: /unreacheable-2/
author: 'No one'
author_profile: False
comments: true
share: False
related: False
use_math: true
sitemap: false
---

## Convex conjugate

Let's take a (non necessary convex function) $$f: E \to R$$ where $$E$$ is some subset of $$R^n$$. The convex conjugate of $$f$$ is defined as

$$f^*(y) = \sup_{x\in E} \left(y^Tx - f(x)\right)$$ 

We call 'domain' of $$f^*$$ the subspace of $$R^n$$ on which $$f^*$$ is finite. 

Let's analyse this thing. The first thing that we notice is that no matter $$f$$, the function $$f^*$$ is a supremum of a set of linear functions (the functions $$y \to y^Tx - f(x)$$ for any given $$x$$). As a result, $$f^*$$ is convex even if $$f$$ is not.

| Dual of the original function | Dual of the dual |
|:--:|:--:|
|![convex conjugate]({{ "/assets/videos/convex/conjugate_clean_convex.mp4" | absolute_url }}) | ![convex conjugate]({{ "/assets/videos/convex/conjugate_clean_convex_dual.mp4" | absolute_url }})|


| Dual of the original function | Dual of the dual |
|:--:|:--:|
|![convex conjugate]({{ "/assets/videos/convex/conjugate_nonconvex_primal.mp4" | absolute_url }}) | ![convex conjugate]({{ "/assets/videos/convex/conjugate_nonconvex_dual.mp4" | absolute_url }})|

| $$f$$ and $$f**$$ for non-convex function | | 
|:--:|:--|
| ![convex conjugate]({{ "/assets/videos/convex/f_and_fpp.jpg" | absolute_url | muted}}) | Some description|

### Useful conjugate of convex functions

The table below gives the conjugate of some of the most useful convex functions:

|Name          | Function      | f*               | Domain  |
|--------------|---------------|------------------|---------|
|Affine       |$$x\to ax + b$$  |$$y\to-b$$          |{$$a$$}    |
|Neg log      |$$x\to -\ln x$$  |$$y\to -\ln(-y)-1$$ | $$R^*_-$$ | 
|Exponential  |$$x\to e^x$$     |$$y\to y\ln(y)-y$$  | $$R_+$$   |
|Negative entropy| $$x\to x\ln(x)$$ | $$y\to e^{y-1}$$| $$R$$     |
|Inverse      |$$x\to 1/x$$     | $$y\to -2\sqrt{-y}$$| $$R_-$$ |
|Log1p exp    |$$x\to \ln(1+e^x)$$| $$y\to y\ln(y)+(1-y)\ln(1-y)$$| $$[0,1]$$|
|Neg Log1p exp |$$x\to -\ln(1-e^x)$$| $$y\to y\ln(y)-(1+y)\ln(1+y)$$| $$R_+$$|
|Root1p square |$$x\to \sqrt{1+x^2}$$|$$y\to -\sqrt{1-y^2}$$|$$[-1,1]$$|
|Log-sum-exp  |$$x\to \ln(\sum_i e^{x_i})$$ | $$y \to \sum_i y_i \ln y_i$$ |$$y \in R^n_+$$ where $$\sum_i y_i=1$$|
|Power | $$x\to \frac{\mid x\mid^p}{p}$$ with $$p > 1$$| $$y\to \frac{\mid y\mid^q}{q}$$ with $$\frac{1}{p}+\frac{1}{q} = 1$$|$$R$$|
|Negative root| $$x\to \frac{-x^p}{p}$$ with $$p < 1$$ | $$y\to \frac{-(-y)^q}{q}$$ with $$\frac{1}{p}+\frac{1}{q} = 1$$|$$R_-$$| 
|p-Norm   |$$x\to\|x\|_p$$| $$y\to 0$$| $$y\in R^n$$ where $$\|y\|_q\leq 1$$ with $$\frac{1}{p}+\frac{1}{q} = 1$$|
|Squared p-Norm   |$$x\to\frac{1}{2}\|x\|_p^2$$| $$y\to\frac{1}{2}\|y\|_q^2$$ where $$\frac{1}{p}+\frac{1}{q} = 1$$| $$R^n$$|
|Quadratic function| $$x\to x^TQx$$ with $$Q \in S_{++}^n$$| $$y\to y^TQ^{-1}y$$| $$R^n$$|
| Log-determinant | $$X\to \ln \det (X^{-1})$$ with $$X\in S_{++}^n$$| $$Y\to \ln\det(-Y)^{-1}-n$$ | $$S_{--}^n$$|

The following results are also very useful to compute the conjugate of a composition of functions. In that case, we consider that we know the conjugate of the function $$h$$. The domain of the conjugate stays the same.

| Function      | f*               |
|---------------|------------------|
|$$x\to h(ax)$$ with $$a\neq 0$$ | $$y\to h^*(\frac{y}{a})$$| 
|$$x\to h(a+x)$$ | $$y\to h^*(y)-y^Ta$$| 
|$$x\to a\times h(x)$$ with $$a > 0$$ | $$y\to a \times h^*(\frac{y}{a})$$| 
|$$x\to \alpha + \beta x + \gamma \times h( \lambda x + \mu)$$ with $$\gamma > 0$$ | $$y\to -\alpha - \mu \frac{y-\beta}{\lambda} + \gamma h^*(\frac{y-\beta}{\gamma \lambda})$$|


## Langrangian and dual 

TODO 

## Applied examples

### Regularized logistic regression

[MAKE A DEMO TO EL GHAOUI EXAMPLE WITH CONJUGATES]

### Dual SVM 

Consider the training problem of a SVM: 

$$ \textrm{arg} \min_{w,b} \frac{1}{2}\|w\|_2^2+C\sum_{i=1}^n \max(1-y_i(w^Tx_i+b), 0)$$

At first we can add some slack variables to get rid of the "max". Let's convince ourselves that the following optimization problem is equivalent to the one we just introduced:

$$ \begin{array}{cl} & \textrm{arg} \min_{w,b} \frac{1}{2}\|w\|_2^2+C\sum_{i=1}^n \xi_i \\
 s.t. & \left\{\begin{array}{l}
\xi_i \geq 0 \\
\xi_i \geq 1-y_i(w^Tx_i+b) \\ 
\end{array}\right. \end{array}$$

The Lagrangian of this optimization problem is:

$$L(w,b,\xi, \lambda, \mu) =  \frac{1}{2}\|w\|_2^2+C\sum_{i=1}^n \xi_i - \sum_i \lambda_i \xi_i + \sum_i \mu_i (1-y_i(w^Tx_i+b) - \xi_i)$$

To find the dual problem, we must solve:

$$G^*(\lambda, \mu) = \min_{w, b, \xi} L(w,b,\xi, \lambda, \mu)$$

First, let's group the terms for each parameters:

$$ \begin{array}{rlr} 
\min_{w, b, \xi} L(w,b,\xi, \lambda, \mu) =
& \min_{w} \frac{1}{2}\|w\|_2^2 - \sum_i\mu_i y_i(w^Tx_i) & (1) \\ 
& + \min_{\xi} \sum_{i=1}^n (C-\mu_i - \lambda_i) \xi_i & (2) \\
& + \min_{b} \sum_i -\mu_i y_i b & (3) \\
& + \sum_i \mu_i & (4) \end{array}$$

Now, because each of these terms only depend one one of the variables of the optimization, we can take care of them one at a time. Don't forget that we are not interested in the domain over which this minimization problem don't have a solution. As a result, we will add some (convex) constraints on the domain of $$G^*$$. 

Let's begin with the simple stuff. 
- (4) Don't require any manipulation, obviously.
- (3) If $$\sum_i -\mu_i y_i \neq 0$$, the minimization problem is unbounded. Thus we add the constraint $$\sum_i \mu_i y_i = 0$$, and the value of the problem is $$0$$.
- (2) If for any $$i$$ we have $$C-\mu_i - \lambda_i \neq 0$$, the problem is unbounded too. As a result, we add the constraints $$\forall i \in [1,...,N], \,C = \mu_i + \lambda_i$$ and the value of the problem is $$0$$. 

Finally, (1) is the only non-trivial part of the problem. Thanksfully, we can use some well-known conjugate to avoid any tedious calculation:

$$\min_{w} \frac{1}{2}\|w\|_2^2 - \sum_i\mu_i y_i(w^Tx_i) = - \sup_{w} \left\{\left(\sum_i\mu_i y_ix_i\right)^Tw - \frac{1}{2}\|w\|_2^2 \right\}$$ 

We know the value of this supremum: is is the conjugate of $$x\to \frac{1}{2}\|w\|_2^2$$, which is $$y\to \frac{1}{2}\|w\|_2^2$$ (self-conjugate!) evaluated at $$y=\sum_i\mu_i y_ix_i$$. Thus we have:

$$\min_{w} \frac{1}{2}\|w\|_2^2 - \sum_i\mu_i y_i(w^Tx_i) = - \frac{1}{2} \|\sum_i\mu_i y_ix_i\|_2^2$$


Plugging all the values and the constraints together, we obtain the dual problem:

$$\begin{array}{cl} & \max_{\lambda, \mu} - \frac{1}{2} \|\sum_i\mu_i y_ix_i\|_2^2+\sum_i \mu_i \\
 s.t. & \left\{\begin{array}{lr}  
C = \mu_i + \lambda_i & \forall i \in [1,...,N]\\
\lambda_i \geq 0 & \forall i \in [1,...,N]\\
\mu_i \geq 0 & \forall i \in [1,...,N]\\
\sum_i \mu_i y_i = 0 \\
\end{array}\right. \end{array}$$

We must remember to add the constraints $$\lambda \geq 0$$ and $$\mu \geq 0$$, as these multipliers are used for inequality constraints. Finally, we can simplify a little bit the constraints by substituting $$\lambda$$, and we can write it as a minimization problem for esthetic purposes:

$$\begin{array} & \min_{\mu} \frac{1}{2} \|\sum_i\mu_i y_ix_i\|_2^2-\sum_i \mu_i \\
 s.t. & \left\{\begin{array}{lr}  
0 \leq \mu_i \leq C & \forall i \in [1,...,N]\\
\sum_i \mu_i y_i = 0 \\
\end{array}\right. \end{array}$$

#### Wait? How de we predict anything now? We lost all the parameters of the decision rule? 

Fear not, enthousiastic reader, for there is a simple solution to this problem. Once we have fitted our SVM on the dataset, we know the optimal coefficients $$\mu^*$$, solution of the dual problem. We can plug it inside the Lagrangian:

$$L(w,b,\xi, \lambda, \mu) = \frac{1}{2}\|w\|_2^2+C\sum_{i=1}^n \xi_i - \sum_i \lambda_i \xi_i + \sum_i \mu_i (1-y_i(w^Tx_i+b) - \xi_i)$$

Recall that we had $$\lambda^* = C - \mu^*$$ at the optimum:

$$L(w,b,\xi,\mu^*) = \frac{1}{2}\|w\|_2^2+\sum_{i=1}^n \mu^*_i\xi_i + \sum_{i=1}^N \mu_i^* (1-y_i(w^Tx_i+b) - \xi_i)$$
$$= \frac{1}{2}\|w\|_2^2 + \sum_{i=1}^N \mu_i^* - \sum_{i=1}^N \mu_i^*y_i(w^Tx_i+b)$$

Also, we saw that at the optimum $$\sum_{i=1}^N \mu_i^*y_i = 0$$, thus the parameter $$b$$ has no influence on the result and we can set $$b=0$$ [wtf? C'est quoi ce bordel?] 

We can recover the optimal parameters $$w^*$$ by solving this very easy convex problem (we get rid of $$\mu^*$$ which is just a constant now): 

$$\min_{w,b} \frac{1}{2}\|w\|_2^2 - \sum_{i=1}^N \mu_i^*y_iw^Tx_i$$

The first order condition gives easily:

$$\nabla_w \left(\frac{1}{2}\|w\|_2^2 - \sum_{i=1}^N \mu_i^*y_iw^Tx_i\right) = 0$$
$$\iff w^* = \sum_{i=1}^N \mu_i^*y_ix_i$$

Finally, we observe that the decision rule depends only on the product $$x_i^Tx_i$$ and not the actual features vector $$x_i$$, thus the kernel trick still works!

$$\textrm{sign}\left(w^Tx_i\right) = \textrm{sign}\left(\sum_{i=1}^N \mu_i^*y_ix_i^Tx_i\right)$$


[TODO sparsity of the solution, kernel trick]

### Kernel trick

The dual problem is not the only one to enjoy the kernel trick: it can also be applied to the original problem. Check it out: the training data $$(x_i)_{i=1,...,N}$$ spans a subspace of $$R^M$$, and we can express any vector of $$R^M$$ as the sum of a vector of this subspace and an orthogonal vector. The parameter $$w$$ is in $$R^M$$, meaning we can write $$w = v+r$$ with $$v = \sum_i \alpha_i x_i$$ ($$v$$ is in the subspace) and $$\forall i, r^Tx_i = 0$$ ($$r$$ is orthogonal). If we plug that inside the original minimization problem: 

$$ \textrm{arg} \min_{r,v,b} \frac{1}{2}\|v+r\|_2^2+C\sum_{i=1}^n \max(1-y_i((v+r)^Tx_i+b), 0)$$

Because $$r$$ and $$v$$ are orthogonals, we have $$\|v+r\|_2^2 = \|r\|_2^2 + \|v\|_2^2$$. Also, $$(v+r)^Tx_i = v^Tx_i$$, thus the problems boils down to: 

$$ \textrm{arg} \min_{r,v,b} \frac{1}{2}\|r\|_2^2 + \frac{1}{2}\|v\|_2^2+C\sum_{i=1}^n \max(1-y_i(v^Tx_i+b), 0)$$

We see that in this problem it is a free-lunch to take $$r=0$$. Finally, if we plug in $$v = \sum_i \alpha_i x_i$$, we have:

$$ \textrm{arg} \min_{\alpha,b} \frac{1}{2}\sum_p\sum_q \alpha_p\alpha_qx_p^Tx_q + C\sum_{i=1}^n \max(1-y_i(\sum_p\alpha_p x_p^Tx_i+b), 0)$$

Only the products $$x_i^Tx_j$$ comes up in this formulation, so the kernel trick works. 
