---
title: Duality applied to machine learning
layout: single
header:
  overlay_image: /assets/images/head/head4.png
permalink: /unreacheable-2/
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

# Duality applied to machine learning

In our [introduction to convex optimization](/unreacheable-3/), we showed how to recognize convex problems, express them in convenient forms and solve them using Python. Today, we go a little bit further an tackle a very important aspect of convex optimization: duality. We will focus on how to calculate quickly the dual of some well known machine learning algorithms, while giving you the keys to do it on your own on new cases if necessary. We will also explain the practical advantages of solving the dual problem. 

After reading this post, you will not be puzzled anymore by the `dual` option in scikit learn, or the _dual formulation of SVM_ that you might have encountered in some papers. Again, don't hesitate to read [the first part of this series](/unreacheable-3/) if you haven't done so yet. But let's start now!

**Resources** Parts of this mini-course are based on the [Boyd](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf), so go check it out if you like what you see here and would want to learn more.
{: .notice--info}


## A brief overview

Before diving in the technical part, let's describe a little bit more what "duality" means in optimization. After all, I don't want to suffer though this whole post, only to discover at the end that you didn't care about any of this.

### What is duality 

In science, duality loosely refers to the existence of two ways of representing the same thing. In particular, duality in optimization means that there exists two ways of expressing the same optimization problem. For instance, if we focus on the training problem of a linear support vector machine:

$$ \begin{array}{lr} \min_{w,b} \frac{1}{2}\|w\|_2^2+C\sum_{i=1}^n \max(1-y_i(w^Tx_i+b), 0) & \boldsymbol{(1)} \end{array}$$

which is sometime written using some slack variables:

$$\begin{array}{lr} \begin{array}{rl} & \min_{w,b} \frac{1}{2}\|w\|_2^2+C\sum_{i=1}^n \xi_i \\
 s.t. & \left\{\begin{array}{lr}  
\xi_i \geq 0 & \forall i \in [1,...,N]\\
\xi_i \geq 1-y_i(w^Tx_i+b) & \forall i \in [1,...,N] \\
\end{array}\right. \end{array}  & \boldsymbol{(2)} \end{array}$$

This problem is called the _primal_ form. It is the optimization problem that arises naturally when describing the loss and the regularization:
- The score of an observation $$x$$ is $$w^Tx$$ for some parameters $$w$$ to learn. We predict the label $$1$$ if the score is positive and the label $$-1$$ else.
- We use the Hinge loss $$t\to \max(0, 1-t \times y)$$ for some score $$t$$ and true label $$y \in \{-1,1\}$$. If the label is $$y=1$$, then, we want the score to be positive, and greater than $$1$$ if possible. On the other hand, if $$y=-1$$, we want the score to be negative and more negative than $$-1$$. 
- We add a $$l_2$$ penalty on the weights for regularization.

However in practice we more often solve the following optimization problem, called _dual problem_:

$$\begin{array}{lr} \begin{array}{rl} & \min_{\mu} \frac{1}{2} \|\sum_i\mu_i y_ix_i\|_2^2-\sum_i \mu_i \\
 s.t. & \left\{\begin{array}{lr}  
0 \leq \mu_i \leq C & \forall i \in [1,...,N]\\
\sum_i \mu_i y_i = 0 \\
\end{array}\right. \end{array} & \boldsymbol{(3)} \end{array}$$

This formulation has some advantages over the primal problem:
- $$\boldsymbol{(3)}$$ is differentiable, whereas $$\boldsymbol{(1)}$$ is only sub-differentiable due to the max function.
- $$\boldsymbol{(3)}$$ has only box constraints (constraints of the form $$0 \leq \mu \leq C$$ for a fixed $$C$$), whereas $$\boldsymbol{(2)}$$ has linear inequality constraints. 

Is it not clear that these two problems are the same, but that's what we will show in this article.

## Langrangian and dual 

### Definitions

First things first, let's start by defining the Lagrangian of some optimization problem. Consider the following constrained problem:

$$\begin{array}{rl} & \min_{x} f(x) \\
 s.t. & \left\{\begin{array}{lr}  
l_i(x) = 0 & \forall i \in [1,...,N_e]\\
h_j(x) \leq 0 & \forall j \in [1,...,N_i]\\
\end{array}\right. 
\end{array}$$

where:
- $$x \in R^{n}$$ is some vector of parameters.
- $$f$$ is a convex objective function.
- $$l_i$$ are some linear functions.
- $$h_j$$ are some convex functions.

The Lagrangian is defined as:

$$\mathcal{L}(x, \lambda, \nu) = f(x) + \sum_i \lambda_i l_i(x) + \sum_j \nu_j h_j(x)$$

with $$\lambda \in R^{N_e}$$ and $$\nu \in R^{N_i}_+$$ (meaning each coordinate $$\nu_j$$ is non-negative). These new variables are called Lagrange multipliers or dual variables. Sometime it is more convenient to write

$$\mathcal{L}(x, \lambda, \nu) = f(x) + \lambda^T l(x) + \nu^T h(x)$$

where the constraints are stacked as vectors:

$$ \begin{array}{lll} l(x) = \begin{pmatrix} l_1(x) \\ l_2(x) \\ \vdots \\ l_{N_e}(x) \end{pmatrix}  &
\textrm{and} & 
h(x) = \begin{pmatrix} h_1(x) \\ h_2(x) \\ \vdots \\ h_{N_i}(x) \end{pmatrix} \end{array}$$

Both formulations can be mixed too, with several vectors $$\lambda_p$$ and $$\nu_q$$. The only important thing is to not forget that the dual variables associated with inequality constraints must be non-negative.

Finally, it's easy to see that the Lagrangian of a convex optimization problem is convex!

### Dual value function

To obtain the dual problem, we define the _dual function value_ $$g(\lambda, \nu)$$ by solving the following unconstrained optimization problem:

$$g(\lambda, \nu) = \min_{x} \mathcal{L}(x, \lambda, \nu)$$

Later in this post we will see some very useful functions, called _convex conjugate_, that are extremely useful for solving these problems and expressing $$g(\lambda, \nu)$$ simply. 

One key observation here is that $$x \to \mathcal{L}(x, \lambda, \nu)$$ is linear for any vector $$x$$ and functions $$f$$, $$l$$ and $$h$$, even if $$f$$, $$l$$ or $$h$$ are non-convex! As a result, $$g$$ is always a concave function, as it is the pointwise minimum of a family of linear functions. 

Finally, the most important thing to notice is that:

$$\forall \, \lambda, \nu, \, \, g(\lambda, \nu) \leq p^*$$

where

$$\begin{array}{rrl} p^*= & & \min_{x} f(x) \\
 & s.t. & \left\{\begin{array}{lr}  
l_i(x) = 0 & \forall i \in [1,...,N_e]\\
h_j(x) \leq 0 & \forall j \in [1,...,N_i]\\
\end{array}\right. 
\end{array}$$

is the optimal value of the original constrained problem. In other words, for any value of the multipliers $$\lambda$$ and $$\mu$$, the dual value is lower than the optimal value of the original problem. 

**Proof:** This is easy to show: on the feasable domain (the subset of values of $$x$$ for which all the constraints are respected), $$\lambda^Th(x)=0$$ as $$l(x) = 0$$, and $$\mu^Th(x) \leq 0$$ as $$h_j(x) \leq 0$$ and $$\nu \geq 0$$, so $$\mathcal{L}(x, \lambda, \nu) \leq f(x)$$. As a result, when taking the minimum on $$x$$, we have $$g(\lambda, \nu) \leq p^*$$
{: .notice--info}

### Dual problem 

We just saw that for any values of the multipliers $$\lambda$$ and $$\mu$$, the dual value function gives a lower bound on the optimal value of the original problem. The natural thing to do is thus to make this lower bound as tight as possible, by maximizing the dual value function on all possible values of $$\lambda$$ and $$\nu$$:

$$\begin{array}{rl} & \max_{\lambda, \nu} g(\lambda, \nu) \\
 s.t. & \left\{\begin{array}{lr}  
\nu \geq 0\\
\end{array}\right. 
\end{array}$$

Et voilà! This is the dual problem we were looking for. By convention the dual problem is often written as a minimization problem:

$$\begin{array}{rl} & \min_{\lambda, \nu} -g(\lambda, \nu) \\
 s.t. & \left\{\begin{array}{lr}  
\nu \geq 0\\
\end{array}\right. 
\end{array}$$

###  Implicit and explicit constraints

> But wait? there is only one constraint here! The dual problem of the SVM had more. Where is the catch?

I hear your concerns but have faith, nothing fishy happened here. The key is to notice that a contraint problem like this:

$$\begin{array}{rl} & \min_{x} f(x) \\
 s.t. & \left\{\begin{array}{l}  
l(x) = 0 \\
h(x) \leq 0 \\
\end{array}\right. 
\end{array}$$

can also be written in an "uncontrained" form:

$$\min_{x} \widetilde{f}(x)$$

where $$\widetilde{f}$$ is defined as:

$$ \widetilde{f} = \left\{ \begin{array}{ll} 
  f(x) & \textrm{ if }  l(x) = 0 \textrm{ and } h(x) \leq 0 \\
  +\infty & \textrm{ else.}  \end{array} \right.$$

In this case, the constraints $$l$$ and $$h$$ are said _implicite_, but it is the exact same problem for sure. 

Well, the same thing happens with the dual problem: for some values of $$\lambda$$ or $$\nu$$, the dual value function take the value $$-\infty$$. Not very useful as a lower bound! As a result, we generally constraint the dual problem to the domain on which $$g(\lambda, \nu) > -\infty$$ by making the implicite constraints explicite.

This will become very clear on the examples later
{: .notice--info}

### Duality gap

So we know that the maximum of the dual problem is a lower bound on the minimum of the original problem. The distance between this minimum and the lower bound is called the duality gap. And the good news is... it is equal to zero if the original problem is convex! 

As a result, solving the dual allows us to get the solution of the original problem: it is truely another way of training our machine learning algorithm. 

**Slater's condition** I lied a little bit. For the duality gap to be equal to zero, another small condition beside convexity must be satisfied. The feasible domain must contain an interior point, or in other words, there must be a $$x^*$$ such that $$h(x^*) \boldsymbol{<} 0$$ (note that the inequality is strict here). I haven't heard of a ML usecase where this condition is not satisfied, so don't sweat it too much.
{: .notice--warning}


## Convex conjugate

Convex conjugates are very useful functions that help us to calculate the dual value function, among other things. Let's take a little bit of time to introduce them.

### Definition

Let's take a (non necessary convex function) $$f: E \to R$$ where $$E$$ is some subset of $$R^n$$. The convex conjugate of $$f$$ is defined as

$$f^*(y) = \sup_{x\in E} \left(y^Tx - f(x)\right)$$ 

We call "domain" of $$f^*$$ the subspace of $$R^n$$ on which $$f^*$$ is finite. 

Let's analyse this thing. The first thing that we notice is that no matter $$f$$, the function $$f^*$$ is a supremum of a set of linear functions (the functions $$y \to y^Tx - f(x)$$ for any given $$x$$). As a result, $$f^*$$ is convex even if $$f$$ is not.

Secondly, there might be some values of $$y$$ for which $$f^*(y) = +\infty$$ (for $$f(x) = a^Tx$$, $$f^*(y) = +\infty$$ everywhere except on $$y = a$$). This is why we restrict $$f^*$$ to a particular domain. Thinking of this in terms of implicite and explicite constraints will helps later.

### Visualization

We can give a nice geometric interpretation of the dual functions, which will help us to visualize it. 
Take the opposite of the conjugate:

$$-f^*(y) = \inf_{x\in E} \left(f(x) - y^Tx\right)$$ 

To make things easy, we will only deal with functions $$f: R \to R$$. In that case, $$xy$$ is the line that goes through the origin and have a slope of $$y$$. For every $$x$$, $$f(x) - yx$$ is the distance between this line and the graph of $$f$$ at $$x$$. As a result, the opposite of the conjugate is minimum distance between this line and the graph of $$f$$.

But that is not all. If you move the line $$\{xy\mid x\in R\}$$ without changing its slope, in order to make it "touch" the graph of $$f$$, you can observe that is is actually "supporting" the graph of $$f$$ (the graph is above this line everywhere).

The following graphs will make it clearer:

| Dual of the original function | Dual of the dual |
|:--:|:--:|
|![convex conjugate]({{ "/assets/videos/convex/conjugate_clean_convex.mp4" | absolute_url }}) | ![convex conjugate]({{ "/assets/videos/convex/conjugate_clean_convex_dual.mp4" | absolute_url }})|

Let's explain what happens here, by focusing on the left-side figure:
- The dark blue curve is the graph $$\{f(x)\mid x\in R\}$$  of some convex function $$f$$. 
- $$y$$ is changing over time.
- The blue straight line represent $$\{xy\mid x\in R\}$$. You can see that it goes through the origin and through the point $$(1,y).$$
- The gray line also have a slope of $$y$$, just as the blue one. It "supports" the graph of $$f$$.
- The red arrow is the distance between $$\{xy\mid x\in R\}$$ (blue line) and the supporting line that touch the graph of $$f$$ on $$(x, f(x))$$ (gray line). It is also the value of $$\inf_{x\in E} \left(f(x) - y^Tx\right)$$!
- As a result, $$-f^*(y)$$ is equal to the length of the arrow.
- To plot the graph of $$f^*$$ by making $$y$$ change over time and keeping the length of the resulting arrows.

On the right side, we do the same thing but for the dual of $$f$$. See anything special? 

### Biconjugate

> I'm not going to make a conjecture from a single observation...

That is very wise of you, almost too wise for this website. Indeed it seems that the conjugate of the conjugate is the function itself. It is a little bit more complicated that that though. 

To see that, let's consider the case of a non-convex function:

| Dual of the original function | Dual of the dual |
|:--:|:--:|
|![convex conjugate]({{ "/assets/videos/convex/conjugate_nonconvex_primal.mp4" | absolute_url }}) | ![convex conjugate]({{ "/assets/videos/convex/conjugate_nonconvex_dual.mp4" | absolute_url }})|

The good news is that the convex conjugate is convex even if $$f$$ is not. Of course this means that the biconjugate $$f^{**}$$ (the conjugate of the conjugate) is convex too. In fact, $$f^{**}$$ is the convex function that is the closest to $$f$$ (we say that it is the _convex hull_ of the the graph of $$f$$). 

**Takeaway** Just remember that $$f^{**} = f$$ if $$f$$ is convex and continuous, and that $$f^{**}$$ is the best convex approximation of $$f$$ that we can get. 
{: .notice--info}

### Calculating the conjugate

When the function $$f$$ is convex and differentiable, the conjugate can be calculated in a relatively simple way. First, the function $$x\to y^Tx - f(x)$$ is concave, thus a necessary and sufficient condition for $$x^*$$ to be optimal is:

$$\nabla \left(x\to y^Tx - f(x)\right) = 0$$

thus

$$y = \nabla f(x)$$

If we can find $$z$$ such that

$$\nabla f(z) = y$$

(this is often done by inverting the function $$x \to \nabla f(x)$$), then we have 

$$f^*(x) = y^Tz - f(z)$$

If $$\nabla \left(x\to y^Tx - f(x)\right) = 0$$ don't have a solution, this means that the problem is unbounded from below and that $$y$$ is not in the domain of $$f^*$$.

### Bestiary

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

### Relations with duality

Consider the following optimization problem with linear constraints:

$$ \begin{array}{cl} & \min_{x} f(x) \\
 s.t. & \left\{\begin{array}{l}
Ax = b \\
Gx \leq h\\ 
\end{array}\right. \end{array}$$

The Lagrangian of this problem is: 

$$\mathcal{L}(x, \lambda, \nu) = f(x) + \lambda^T(Ax-b) + \nu^T(Gx-h)$$

Thus the dual value function is:

$$\begin{array}{rl}
g(\lambda, \nu) &= \min_x f(x) + \lambda^T(Ax-b) + \nu^T(Gx-h) \\
&= \min_x f(x) + (A^T\lambda+G^T\nu)^Tx - \lambda^Tb -\nu^Th \\
&= -\sup_x \left\{ (-A^T\lambda-G^T\nu)^Tx - f(x)\right\} - \lambda^Tb -\nu^Th \\
&= -f^*(-A^T\lambda-G^T\nu) - \lambda^Tb -\nu^Th \\
\end{array}$$

And the dual problem becomes:

$$ \begin{array}{cl} & \min_{\lambda, \mu} f^*(-A^T\lambda-G^T\nu) + \lambda^Tb + \nu^Th \\
 s.t. & \left\{\begin{array}{l}
\nu \geq 0 \\
-A^T\lambda-G^T\nu \in \textrm{Domain}({f^*})
\end{array}\right. \end{array}$$

## Applied examples: SVM

### Finding the dual problem

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

$$\begin{array}{rl} & \min_{\mu} \frac{1}{2} \|\sum_i\mu_i y_ix_i\|_2^2-\sum_i \mu_i \\
 s.t. & \left\{\begin{array}{lr}  
0 \leq \mu_i \leq C & \forall i \in [1,...,N]\\
\sum_i \mu_i y_i = 0 \\
\end{array}\right. \end{array}$$

### Recovering the decision function

> Wait? How de we predict anything now? We lost all the parameters of the decision rule? 

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

### Kernel trick and dual

Sometime I read online that the dual problem of the SVM allows us to use the kernel trick. This is in fact not true: the dual problem is not the only one to enjoy the kernel trick, as it can also be applied to the original problem. 

Check it out: the training data $$(x_i)_{i=1,...,N}$$ spans a subspace of $$R^M$$, and we can express any vector of $$R^M$$ as the sum of a vector of this subspace and an orthogonal vector. The parameter $$w$$ is in $$R^M$$, meaning we can write $$w = v+r$$ with $$v = \sum_i \alpha_i x_i$$ ($$v$$ is in the subspace) and $$\forall i, r^Tx_i = 0$$ ($$r$$ is orthogonal). If we plug that inside the original minimization problem: 

$$ \textrm{arg} \min_{r,v,b} \frac{1}{2}\|v+r\|_2^2+C\sum_{i=1}^n \max(1-y_i((v+r)^Tx_i+b), 0)$$

Because $$r$$ and $$v$$ are orthogonals, we have $$\|v+r\|_2^2 = \|r\|_2^2 + \|v\|_2^2$$. Also, $$(v+r)^Tx_i = v^Tx_i$$, thus the problems boils down to: 

$$ \textrm{arg} \min_{r,v,b} \frac{1}{2}\|r\|_2^2 + \frac{1}{2}\|v\|_2^2+C\sum_{i=1}^n \max(1-y_i(v^Tx_i+b), 0)$$

We see that in this problem it is a free-lunch to take $$r=0$$. Finally, if we plug in $$v = \sum_i \alpha_i x_i$$, we have:

$$ \textrm{arg} \min_{\alpha,b} \frac{1}{2}\sum_p\sum_q \alpha_p\alpha_qx_p^Tx_q + C\sum_{i=1}^n \max(1-y_i(\sum_p\alpha_p x_p^Tx_i+b), 0)$$

Only the products $$x_i^Tx_j$$ comes up in this formulation, so the kernel trick works. 
