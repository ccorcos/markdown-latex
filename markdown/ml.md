Conventions
===========

All vectors are assumed to be vertical vectors and are denoted with
bold-face, $\boldsymbol{v}$.\
Matrices are capitalized and boldface, ${\boldsymbol{M}}$.

$p(\cdot)$ is the probability function.\
$f(\cdot)$ is the prediction function, predicting some output, $y$ given
input parameters, ${\boldsymbol{x}}$. There are two types of prediction
functions, continuous (regression) and classification.

$\mathcal{D} = \{ ({\boldsymbol{x}}_1,y_1), ({\boldsymbol{x}}_2, y_2), \hdots, ({\boldsymbol{x}}_N,y_N)\}$
is a dataset of $N$ samples drawn from a joint distribution,
$p({\boldsymbol{x}}, y)$, with known inputs, ${\boldsymbol{x}}_i$ and
corresponding correct output, $y_i$.
${\boldsymbol{x}} \sim {\mathbb{R}}^{D \times 1}$ corresponding to D
dimensions or features of each sample. For predicting house prices, a
feature may be size of the house, lot size, number of bathrooms, or
number of bedrooms.

Often, data will be represented in matrix form where
${\boldsymbol{X}} \sim {\mathbb{R}}^{N \times D}$ and
${\boldsymbol{y}} \sim {\mathbb{R}}^{N \times 1}$. Sometimes this data
may be preprocessed (zero mean, unit variance), and often, one of the
dimensions will be a column of 1’s representing a bias or offset term.

Probabilistic Machine Learning Terminology
==========================================

Loss Function
-------------

$L(f({\boldsymbol{x}}),y)$, is some measure of prediction error. There
are many types of loss functions.

- L1-norm: $L(f({\boldsymbol{x}}),y) = \|f({\boldsymbol{x}}) - y\|_1$         – city block distance

- L2-norm: $L(f({\boldsymbol{x}}),y) = \|f({\boldsymbol{x}}) - y\|_2$         – just the euclidian

- p-norm: $L(f({\boldsymbol{x}}),y) = \|f({\boldsymbol{x}}) - y\|_p$          – more generally speaking

- square-error: $L(f({\boldsymbol{x}}),y) = \|f({\boldsymbol{x}}) - y\|_2^2$        – used in linear regression and logistic regression
- hinge-loss: $L(f({\boldsymbol{x}}),y) = \text{max}(0,1-yf({\boldsymbol{x}}))$ – used in SVMs for classification.
- exponential-loss:   $L(f({\boldsymbol{x}}),y) = e^{-yf({\boldsymbol{x}})}$              – used in adaboost for classification

The loss function is equal to the negated conditional log-likelihood

$$\begin{aligned}
L(f({\boldsymbol{x}}),y) &= - \log p(\mathcal{D}|{\boldsymbol{w}})\\
&= - \sum_n \log p(y_n|{\boldsymbol{x}}_n,{\boldsymbol{w}})\end{aligned}$$

Expected Conditional Risk
-------------------------

$$\begin{aligned}
R(f,{\boldsymbol{x}}) &= {\mathbb{E}}_{y \sim p(y|{\boldsymbol{x}})}\;L(f({\boldsymbol{x}}),y)\\
 &= \int L(f({\boldsymbol{x}}),y) \;p(y|{\boldsymbol{x}})  \;dy\end{aligned}$$

says, given $y$ follows some distribution conditioned on
${\boldsymbol{x}}$, what can we expect the loss to be if we marginalize
over $y$. This means, what sort of loss (risk) can we expect from this
predictor, $f(\cdot)$, given (conditioned) on the data we are given
(${\boldsymbol{x}}$).

Expected Risk
-------------

$$\begin{aligned}
R(f) &= {\mathbb{E}}_{{\boldsymbol{x}}} \;R(f,{\boldsymbol{x}})\\
&= {\mathbb{E}}_{{\boldsymbol{x}} \sim p({\boldsymbol{x}})}{\mathbb{E}}_{y \sim p(y|{\boldsymbol{x}})}\;L(f({\boldsymbol{x}}),y) \\
&= \int \int L(f({\boldsymbol{x}}),y) \;p(y|{\boldsymbol{x}}) p({\boldsymbol{x}})  \;dy \;d{\boldsymbol{x}} \\
&= \int \int L(f({\boldsymbol{x}}),y) \;p({\boldsymbol{x}},y) \;dy \;d{\boldsymbol{x}}\end{aligned}$$

marginalizes the expected conditional risk over the input data,
${\boldsymbol{x}}$, leaving us with the overall expected risk of some
prediction function.

Empirical Risk
--------------

Given some data, we can approximate the expected risk with the empirical
risk given by

$$\begin{aligned}
R_{\mathcal{D}}(f) &= \frac{1}{N} \sum_n L(f({\boldsymbol{x}}_n),y_n)\end{aligned}$$

Also, with infinite data, empirical risk is the expected risk

$$\begin{aligned}
\lim_{N \to \infty} R_{\mathcal{D}}(f) &= R(f)\end{aligned}$$

Emperical risk is by definition the average of the loss function over
all data. This is also known as cross-entropy $\mathcal{E}(\cdot)$.

Bayes’ Optimal Binary Classifier
--------------------------------

This is a theoretical probabilistically optimal classifier. Assume some
$\eta({\boldsymbol{x}}) = p(y=1|{\boldsymbol{x}})$. Then the Bayes’
Optimal Binary Classifier is

$$\begin{aligned}
f^*({\boldsymbol{x}}) =  \begin{cases} 1 & \text{if } \eta({\boldsymbol{x}}) \ge 1/2 \\
0 & \text{if } \eta({\boldsymbol{x}}) < 1/2 \end{cases} \end{aligned}$$

This is very useful in proving the performance of your classifier.

Bayes’ Theorem
--------------

You are probably familiar with Bayes’ Theorem but I want to go over it
just to clear up some terms.

$$\begin{aligned}
p({\boldsymbol{w}}|\mathcal{D}) &= \frac{p({\boldsymbol{w}}) \times p(\mathcal{D}|{\boldsymbol{w}})}{p(\mathcal{D})}\\
\text{posterior} &= \frac{\text{prior} \times \text{likelihood}}{\text{evidence}}\end{aligned}$$

These terms will come up. Also pertinent to this topic is the concept of
conjugate priors which defines the type of posterior distribution given
the type of likelihood and prior distributions.

In plain english, this theorem describes that the probability of some
parametric weights fitting some data is equal to any prior knowledge of
what the weights should be times the likelihood that data is described
by those weights divided by the probability of seeing that data.

When doing maximum likelihood estimation, we just maximize (typically
the log) conditional likelihood. When we want to find the maximum a
posterior solution, we typically leave out the evidence term because it
is a constant that is nearly impossible to know and does not effect the
optimization.

Optimization Background
=======================

Constrained Optimization – Lagrange Multipliers
-----------------------------------------------

Suppose we want to minimize some function subject to some constrains.

$$\begin{aligned}
\text{min }& f(x)\\
\text{s.t. }& g(x) = 0\end{aligned}$$

To solve this, we construct the <span>*Lagrangian*</span>.

$$\begin{aligned}
L(x,\lambda) = f(x) + \lambda g(x)\end{aligned}$$

Here, $\lambda$ is called a <span>*lagrange multiplier*</span>. To solve
our constrained minimization, take the derivative with respect to each
of the <span>*primal*</span> variables, $x$ and $\lambda$, and solve
equal to zero. $\frac{d\;L(x,\lambda)}{dx}=0$ and
$\frac{d\;L(x,\lambda)}{d\lambda }=0$. Substitute and solve.

If there is more than one constraint, just create another lagrange
multiplier and add it to the lagrangian.

Lagrange Duality {#sec:lagrange-duality}
----------------

Lagrange duality refers to the difference between the dual and primal
formulations of an optimization problem.

Given the constrained optimization problem, the <span>*primal*</span>
formulation of the problem is given by

$$\begin{aligned}
\min_{x} \;\;& f(x)\\
\text{s.t.} \;\; & g_i(x) \le 0 \; \;\forall \; i\end{aligned}$$

To get the <span>*dual*</span> formulation, we derive the lagrangian as
follows, where $\{\lambda_i\}$ are the set of lagrange multipliers

$$\begin{aligned}
L(x,\{\lambda_i\}) = f(x) + \sum_i \lambda_i g_i(x)\end{aligned}$$

Note that

$$\begin{aligned}
L(x,\{\lambda_i\}) & \le f(x)\\
\min_{x} \;\max_{\{\lambda_i\}} \;L(x,\{\lambda_i\}) & = \min_{x} \;f(x) \text{ s.t. } g_i(x) \le 0 \; \;\forall \; i\end{aligned}$$

Performing this min-max is difficult. This gives rise to the
<span>*dual*</span> formulation which is a lower bound on the optimal
solution.

$$\begin{aligned}
g(\{\lambda_i\}) &= \min_{x} \;L(x,\{\lambda_i\})\\
\max_{\{\lambda_i\}} \;g(\{\lambda_i\}) &= \max_{\{\lambda_i\}} \;\min_{x} \;L(x,\{\lambda_i\})\end{aligned}$$

From before, we can clearly see that

$$\begin{aligned}
g(\{\lambda_i\}) \le f(x)\end{aligned}$$

Thus the solution to the dual

$$\begin{aligned}
\max_{\{\lambda_i\}} \;g(\{\lambda_i\}) \le \min_{x} \; f(x)\end{aligned}$$

The difference between the primal and dual solutions is called the
duality gap.

The dual formulation is properly given by

$$\begin{aligned}
\max_{\{\lambda_i\}} \;\;& g(\{\lambda_i\})\\
s.t. \;\; & \lambda_i \ge 0 \; \;\forall \; i\end{aligned}$$

Nearest Neighbor Classifier (NNC)
=================================

This is a very basic classifier, but it has a very strong theoretical
proof. We define the <span>*nearest neighbor*</span> as the sample in
the training set with the smallest distance to some sample in question.

$$\begin{aligned}
nn({\boldsymbol{x}}) = \text{arg min}_n \;\; \text{distance}({\boldsymbol{x}}, {\boldsymbol{x}}_n)\end{aligned}$$

This distance function is typically the squared euclidian distance

$$\begin{aligned}
nn({\boldsymbol{x}}) = \text{arg min}_n \| {\boldsymbol{x}} - {\boldsymbol{x}}_n \|_2^2\end{aligned}$$

We then classify based on the classification of the nearest neighbor

$$\begin{aligned}
f({\boldsymbol{x}}) = y_{nn({\boldsymbol{x}})}\end{aligned}$$

This is called a non-parametric model because it depends only on our
training data. We will have to carry our training data around with us in
order to classify new samples.

K-Nearest Neighbors Classifier (KNN)
------------------------------------

This simply extends nearest neighbor to classify based on the classes of
the k nearest neighbors

$$\begin{aligned}
knn({\boldsymbol{x}}) = \{ nn_1({\boldsymbol{x}}), nn_2({\boldsymbol{x}}), \hdots, nn_k({\boldsymbol{x}}) \}\end{aligned}$$

We then classify based on the majority class $c \in \{C\}$ (reads: some
class $c$ that is an element of the set of all classes, $\{C\}$).

$$\begin{aligned}
f({\boldsymbol{x}}) = \text{arg max}_c \sum_{n \in knn({\boldsymbol{x}})} {\mathbb{I}}(y_n == c)\end{aligned}$$

Here, we utilize the identity function, ${\mathbb{I}}(\cdot)$ which
returns 1 if input is true and 0 if the input is false.

Theoretical Guarantees
----------------------

We have a simple loss function

$$\begin{aligned}
L(f({\boldsymbol{x}}),y) = \begin{cases} 0 & \text{if } f({\boldsymbol{x}}) = y \\ 1  & \text{if } f({\boldsymbol{x}}) \ne y \end{cases}\end{aligned}$$

We can compute the expected conditional risk as follows. There are two
possible ways of making a mistake and we can compute their probabilities
(where $\eta({\boldsymbol{x}}) = p(y=1|{\boldsymbol{x}})$)\
1)
$p(f({\boldsymbol{x}}) = 1 | y = 0) = \eta(nn({\boldsymbol{x}}))(1- \eta({\boldsymbol{x}}))$\
2)
$p(f({\boldsymbol{x}}) = 0 | y = 1) = (1- \eta(nn({\boldsymbol{x}})))\eta({\boldsymbol{x}})$\

Writing out the expected conditional risk

$$\begin{aligned}
R(f,{\boldsymbol{x}}) = \eta(nn({\boldsymbol{x}}))(1- \eta({\boldsymbol{x}})) +  (1- \eta(nn({\boldsymbol{x}}))){\boldsymbol{x}})\end{aligned}$$

It can be shown that

$$\begin{aligned}
\lim_{N \to \infty} \eta(nn({\boldsymbol{x}})) = \eta({\boldsymbol{x}})\end{aligned}$$

Thus for $N \to \infty$

$$\begin{aligned}
R(f,{\boldsymbol{x}}) = 2\eta({\boldsymbol{x}})(1-\eta({\boldsymbol{x}}))\end{aligned}$$

When compared to the Bayes’ optimal binary classifier

$$\begin{aligned}
R(f^*,{\boldsymbol{x}}) = \text{min}\{\eta({\boldsymbol{x}}), 1-\eta({\boldsymbol{x}}) \}\end{aligned}$$

We see that

$$\begin{aligned}
R(f,{\boldsymbol{x}}) = 2R(f^*,{\boldsymbol{x}}) (1 - R(f^*,{\boldsymbol{x}}) )\end{aligned}$$

We can then proceed to show an upper bound on the expected risk

$$\begin{aligned}
R(f) &= {\mathbb{E}}_{x}R(f,{\boldsymbol{x}})\\
&=  {\mathbb{E}}_{x}2R(f^*,{\boldsymbol{x}})(1- R(f^*,{\boldsymbol{x}}))\\
&=  2{\mathbb{E}}_{x}R(f^*,{\boldsymbol{x}}) - 2{\mathbb{E}}_{x}R(f^*,{\boldsymbol{x}})^2\\
&=  2R(f^*) - [2R(f^*)^2 + \text{var}(R(f^*,x))]\\
&=  2R(f^*)(1 - R(f^*)) - \text{var}(R(f^*,x))\\
&\le  2R(f^*)(1 - R(f^*))\end{aligned}$$

In the last step, we drop off the variance term and set the = to $\le$
because the variance must be positive.

We have thus shown a very strong theoretical guarantee for NNC and that
is

$$\begin{aligned}
R(f^*) \le R(f^{NNC}) \le 2R(f^*,{\boldsymbol{x}}) (1 - R(f^*,{\boldsymbol{x}}) )\end{aligned}$$

Linear Regression
=================

Linear regression is the problem of fitting a line to a set of points
given a square-error loss function and a linear parametric model defined
by

$$\begin{aligned}
f({\boldsymbol{x}}) = {\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}\end{aligned}$$

where ${\boldsymbol{w}}$ are a set of parametric weights for each
dimension.

Note the difference between parametric models (linear regression) and
non-parametric models (KNN or NNC) is that non-parametric models use
only the data to make predictions while parametric models make
predictions based on some parameters, in this case ${\boldsymbol{w}}$.
These parameters are often referred to as weights. This is because the
prediction is based on a weighted sum of sample features (dimensions).

This gives us an empirical risk (or cross-entropy) of

$$\begin{aligned}
R_{\mathcal{D}}(f) = \frac{1}{N} \sum_n \|y_n - f({\boldsymbol{x}})\|_2^2\end{aligned}$$

In this case, empirical risk is often referred to as the
residual-sum-of-squares (RSS) or mean-square-error (MSE).

The proper probabilistic way of deriving the solution to the weights is
by maximizing the conditional likelihood or probability of seeing the
data. By doing , we are creating a <span>*discriminative*</span> model
because we optimizing the conditional probability. Later on with
Bayesian linear regression, we will create a <span>*generative*</span>
model by maximizing the complete likelihood (joint probability).

$$\begin{aligned}
\text{arg max}_{{\boldsymbol{w}}} p(\mathcal{D}|{\boldsymbol{w}}) = \text{arg max}_{{\boldsymbol{w}}} \prod_n p(y_n|x_n,{\boldsymbol{w}})\end{aligned}$$

To clean up the notion, we will leave out the weights in the probability
and assume it is implied

$$\begin{aligned}
\text{arg max}_{{\boldsymbol{w}}} p(\mathcal{D}) = \text{arg max}_{{\boldsymbol{w}}} \prod_n p(y_n|x_n)\end{aligned}$$

We assume a noisy observation model such that

$$\begin{aligned}
y = {\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}} + \eta\end{aligned}$$

where the noise, $\eta \sim N(0,\sigma^2)$ is zero mean gaussian noise.
Thus for one training sample

$$\begin{aligned}
p(y_n|x_n) &= N(y_n - {\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}, \sigma^2)\\
&= \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(y_n - {\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}})^2}{2\sigma^2}}\end{aligned}$$

Now we can maximize the conditional likelihood. A common trick is to
equivalently maximize log-likelihood. This turns products into sums
which makes things easier.

$$\begin{aligned}
\log p(\mathcal{D}) &=  \log \prod_n p(y_n|x_n)\\
&=  \sum_n \log p(y_n|x_n)\\
&=  \sum_n -\frac{(y_n - {\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}})^2}{2\sigma^2} - \log\sigma\sqrt{2\pi}\\\end{aligned}$$

Now, to find the optimal weights, we take the derivative set to zero

$$\begin{aligned}
\frac{\partial}{\partial {\boldsymbol{w}}} \log p(\mathcal{D}) &\propto \frac{\partial}{\partial {\boldsymbol{w}}} \sum_n (y_n - {\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}})^2 = 0\end{aligned}$$

This means that
$\text{arg max}_{{\boldsymbol{w}}} \log p(\mathcal{D}) = \text{arg min}_{{\boldsymbol{w}}} R_{\mathcal{D}}(f)$!

$$\begin{aligned}
{\boldsymbol{w}} &= \text{arg min}_{{\boldsymbol{w}}} R_{\mathcal{D}}(f)\\
&= \text{arg min}_{{\boldsymbol{w}}} \frac{1}{N} \sum_n (y_n - {\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n){^{\textrm T}}(y_n - {\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n)\\
&= \text{arg min}_{{\boldsymbol{w}}} \frac{1}{N} \sum_n {\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n{\boldsymbol{x}}_n{^{\textrm T}}{\boldsymbol{w}} - 2y_n{\boldsymbol{x}}_n{^{\textrm T}}{\boldsymbol{w}} + y_n^2\\
&= \text{arg min}_{{\boldsymbol{w}}} \frac{1}{N} \left[ {\boldsymbol{w}}{^{\textrm T}}\left(\sum_n {\boldsymbol{x}}_n{\boldsymbol{x}}_n{^{\textrm T}}\right){\boldsymbol{w}} - 2\left(\sum_n y_n{\boldsymbol{x}}_n{^{\textrm T}}\right){\boldsymbol{w}} \right]+\text{constant}\\
&= \text{arg min}_{{\boldsymbol{w}}} \frac{1}{N} \left[ {\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{X}}{\boldsymbol{w}} - 2\left({\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{y}}\right){\boldsymbol{w}} \right]+\text{constant}\\
0 &= \frac{\partial}{\partial {\boldsymbol{w}}} \frac{1}{N} \left[ {\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{X}}{\boldsymbol{w}} - 2\left({\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{y}}\right){\boldsymbol{w}} \right]\\
{\boldsymbol{w}}^{MLE} &= \left({\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{X}}\right)^{-1}{\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{y}} \end{aligned}$$

The solution derived above is also known as the maximum likelihood
estimation (MLE) solution or the least-mean-squares (LMS) solution.

Note that all of the different dimensions of the weights are
independent. Also, the solution is independent of the noise, $\eta$.

Also note that we could use numerical optimization tools. What if $D$ is
large? That matrix inversion may be intractable. Thankfully, the
(empirical risk) objective function is convex

$$\begin{aligned}
\frac{\partial R_{\mathcal{D}}(f)}{\partial {\boldsymbol{w}}} &\propto {\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{X}}{\boldsymbol{w}} - 2{\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{y}}\\
\frac{\partial^2 R_{\mathcal{D}}(f)}{\partial {\boldsymbol{w}}{\boldsymbol{w}}{^{\textrm T}}} &= {\boldsymbol{H}}\\
&\propto {\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{X}}\end{aligned}$$

This is positive semidefinite, thus proving convexity.

Linear regression can be extended to nonlinear regression using a basis
function, ${\boldsymbol{\phi}}({\boldsymbol{x}})$, which transforms the
data into a nonlinear basis. For example, given a dataset where $D=2$,
we can use a nonlinear basis function to transform the data into a
quadratic space

$$\begin{aligned}
{\boldsymbol{\phi}}({\boldsymbol{x}}) \to \left[ \begin{array}{c} x_1\\x_2\\x_1^2\\x_1\cdot x_2\\ x_2^2 \end{array}\right]\end{aligned}$$

This allows us to fit a quadratic to our data using linear regression
rather than simply a line. In this way, we are increasing the
dimensionality of our data. However, this will be prone to overfitting.
See section [sec:bias-variance] about the trade-off between model
complexity and overfitting.

Ridge Linear Regression
-----------------------

Also known as regularized linear regression.

This can be thought of from two difference perspectives.

1\) Given the MLE solution, what if
${\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{X}}$ is not invertible? This
could happen if $N<D$. Intuitively, this would mean there is not enough
data to estimate all the parameters. This could also happen is the
columns of ${\boldsymbol{X}}$ are not linearly independent. This would
happen if any features are perfectly correlated.

An easy solution to this is to add a diagonal matrix to the solution

$$\begin{aligned}
{\boldsymbol{w}} &= \left( {\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{X}} + \lambda {\boldsymbol{I}}\right)^{-1}{\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{y}} \end{aligned}$$

This makes linear regression more numerically stable. The matrix is now
guaranteed to be invertible.

$\lambda>0$ is called the regularization term. It is considered a
<span>*hyperparameter*</span> of the model because it will need to be
optimized separately from the optimization of the parametric weights of
the model. For more on tuning hyperparameters, see section
[sec:hyperparameters].

2\) Suppose our model is susceptible to overfitting. Thus we want to our
model to be more simple. We can do this by introducing a
<span>*prior*</span> belief

$$\begin{aligned}
p({\boldsymbol{w}}) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{{\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{w}}}{2\sigma^2}}\end{aligned}$$

namely, that ${\boldsymbol{w}}$ is around zero resulting in a simple
model. This line of thinking is to regard ${\boldsymbol{w}}$ as a random
variable and we will use the observed data to update our <span>*a
prior*</span> belief on ${\boldsymbol{w}}$.

Now, rather than maximizing the conditional likelihood, we want to
maximize the joint probability

$$\begin{aligned}
p(\mathcal{D},{\boldsymbol{w}}) &= p(\mathcal{D}|{\boldsymbol{w}})p({\boldsymbol{w}})\\
\log p(\mathcal{D},{\boldsymbol{w}}) &= \sum_n \log p(y_n|{\boldsymbol{x}}_n,{\boldsymbol{w}}) + \log p({\boldsymbol{w}})\\
&= - \frac{1}{2\sigma_0^2} \sum_n ({\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n - y_n)^2 -  \frac{1}{2\sigma^2} \|{\boldsymbol{w}}\|_2^2 + \text{constant}\\
\mathcal{E}({\boldsymbol{w}}) &= \frac{1}{2\sigma_0^2} \sum_n ({\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n - y_n)^2 +  \frac{1}{2\sigma^2} \|{\boldsymbol{w}}\|_2^2\end{aligned}$$

Where $\sigma_0$ is the variance of the noise. Now, we seek to maximize
the <span>*a posterior*</span> (MAP) solution to the weights

$$\begin{aligned}
{\boldsymbol{w}}^{MAP} = \text{arg max}_{{\boldsymbol{w}}} \log p({\boldsymbol{w}}|\mathcal{D})\end{aligned}$$

and we know that

$$\begin{aligned}
p({\boldsymbol{w}},\mathcal{D}) &= p({\boldsymbol{w}}|\mathcal{D})p(\mathcal{D})\\
\log p({\boldsymbol{w}},\mathcal{D}) &= \log p({\boldsymbol{w}}|\mathcal{D})+ \log p(\mathcal{D})\\
\log p({\boldsymbol{w}},\mathcal{D}) &= \log p({\boldsymbol{w}}|\mathcal{D})+ \text{constant}\\\end{aligned}$$

so maximizing the a posterior is analogous to maximizing the the joint
likelihood. Thus,

$$\begin{aligned}
{\boldsymbol{w}}^{MAP} &= \text{arg max}_{{\boldsymbol{w}}} \log p({\boldsymbol{w}}|\mathcal{D}) \\
&= \text{arg max}_{{\boldsymbol{w}}} \log p({\boldsymbol{w}},\mathcal{D}) \\
&= \text{arg min}_{{\boldsymbol{w}}} \mathcal{E}({\boldsymbol{w}}) \\
&= \text{arg min}_{{\boldsymbol{w}}} \sum_n ({\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n - y_n)^2 +  \frac{\sigma_0^2}{\sigma^2} \|{\boldsymbol{w}}\|_2^2\\
&= \left( {\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{X}} + \frac{\sigma_0^2}{\sigma^2} {\boldsymbol{I}}\right)^{-1}{\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{y}}\end{aligned}$$

where the regularization term $\lambda = \frac{\sigma_0^2}{\sigma^2}$,
the ratio of the model noise variance $\sigma_0$, to the variance of the
weights $\sigma$. A smaller $\sigma$ indications a stronger prior for a
simpler model, thus more regularization.

Again, regularization must be treated as a hyperparameter of the model.
For more on tuning hyperparameters, see section [sec:hyperparameters].

Bayesian Linear Regression
--------------------------

This is the full blown Bayesian treatment of linear regression. Bayesian
methods can be applied all over the place. Conjugate priors make life a
bit easier though its not necessary. However, for linear regression, all
distributions are Gaussian which makes things nice and easy.

First we define the likelihood as a normal distribution with some
precision/variance due to noise

$$\begin{aligned}
p(y|{\boldsymbol{x}}) &= N({\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}, \beta^{-1})\end{aligned}$$

$\beta$ is known as precision and is the inverse of the variance of the
noise (remember $\eta \sim N(0,\sigma^2)$). We also define a prior that
tries urge a simpler model.

$$\begin{aligned}
p({\boldsymbol{w}}) = N({\boldsymbol{0}},\alpha^{-1}{\boldsymbol{I}})\end{aligned}$$

Our prior suggests that ${\boldsymbol{w}}$ is around zero resulting in a
simple model. $\alpha$ is the precision that tells us how confident we
think we are about where ${\boldsymbol{w}}$ is centered. Given this
prior (the weights centered at zero), $\alpha$ defines how simple out
model ought to be.

Now to derive the maximum a posterior solution

$$\begin{aligned}
p({\boldsymbol{w}}|\mathcal{D}) &\propto p({\boldsymbol{w}})  p(y|{\boldsymbol{x}}, {\boldsymbol{w}})\\
&\propto N({\boldsymbol{w}}|{\boldsymbol{0}}, \alpha^{-1}) \times \prod_n N(y_n|{\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n, \beta^{-1})\\
&\propto \exp{- \frac{\alpha}{2}{\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{w}} - \frac{\beta}{2}\sum_n(y_n - {\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n)^2} \\
&\propto \exp{ - \frac{1}{2}({\boldsymbol{w}} - {\boldsymbol{\mu}}){^{\textrm T}}{\boldsymbol{\Sigma}}^{-1}({\boldsymbol{w}} - {\boldsymbol{\mu}})}\\
&\propto N({\boldsymbol{w}}|{\boldsymbol{\mu}}, {\boldsymbol{\Sigma}})\end{aligned}$$

The last step involves completing the square.

$$\begin{aligned}
{\boldsymbol{\Sigma}} &= \left(\alpha {\boldsymbol{I}} + \beta {\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{X}}\right)^{-1}\\
{\boldsymbol{\mu}} &= \beta {\boldsymbol{\Sigma}} {\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{y}}\end{aligned}$$

Note that we derived what the MAP solution is proportional to. That is
because the evidence is just a constant. If we were to maximize the log
posterior probability, we would be left with
${\boldsymbol{w}} = {\boldsymbol{\mu}}$. This is the same solution as we
saw before for ridge regression only this time we also have our
confidence in that solution.

Next, we derive the predictive distribution for a new data point

$$\begin{aligned}
p(y|{\boldsymbol{x}}, \mathcal{D}) &= \int \text{likelihood} \times \text{posterior } d{\boldsymbol{w}}\\
&= \int p(y|{\boldsymbol{x}}, {\boldsymbol{w}}) p({\boldsymbol{w}}|\mathcal{D})d{\boldsymbol{w}}\\
&\propto \int \exp{-\frac{\beta}{2}(y - {\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}})^2 - \frac{1}{2}({\boldsymbol{w}} - {\boldsymbol{\mu}}){^{\textrm T}}{\boldsymbol{\Sigma}}^{-1}({\boldsymbol{w}} - {\boldsymbol{\mu}})}d{\boldsymbol{w}}\\
 &\propto N(y|{\boldsymbol{\mu}}{^{\textrm T}}{\boldsymbol{x}}, \beta^{-1} + {\boldsymbol{x}}{^{\textrm T}}{\boldsymbol{\Sigma}}{\boldsymbol{x}}) \\
 &\propto N(y|{\boldsymbol{\mu}}{^{\textrm T}}{\boldsymbol{x}},\sigma^2({\boldsymbol{x}})
 \end{aligned}$$

What we are left with is same prediction as before,
${\boldsymbol{\mu}}{^{\textrm T}}{\boldsymbol{x}}$, but now we also have
a confidence for each data point, $\sigma^2({\boldsymbol{x}})$! This
allows you to plot your regression curve along with error bounds
indicating confidence in any region of the curve.

So how do we choose $\alpha$ and $\beta$? There is no analytical
solution. There are some iterative procedures involving
eigendecomposition, but in practice, they are tuned as hyperparameters
of the model using cross-validation.

A huge benefit of Bayesian linear regression is that we can update the
model with new data without recalculating computing all of the training
data. When we first train the model on the training data and get a
posterior, we can use that as the prior to compute a new posterior with
new data!

Another benefit is that we treat every prediction as an independent
gaussian process and thus we can compute the confidence in each
prediction given by $\sigma^2({\boldsymbol{x}})$.

Logistic Regression
===================

This is very analogous to linear regression except it is used for
classification. It is a non-linear model with no analytical solution,
but it is often referred to as a linear model because the decision
boundary must be a hyperplane. The model returns a probability of some
sample belonging to a specific class or not, defined by the conditional
likelihood

$$\begin{aligned}
p(y_n=c|{\boldsymbol{x}}_n) &= \frac{1}{1+e^{-{\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n}}\\
&= \sigma({\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n)\end{aligned}$$

where $\sigma(\cdot)$ is the sigmoid function. Note that this is binary
classification. Either $y=c$ or $y\ne c$. We will alter the notation to
be $y=1$ and $y=0$ respectively. This allows us to rewrite this nicely

$$\begin{aligned}
p(y_n|{\boldsymbol{x}}_n) &= \sigma({\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n)^{y_n}(1-\sigma({\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n))^{1-y_n}\end{aligned}$$

We then proceed to determine the optimal weights using maximum
likelihood estimation. But first, we take the log to make things easier

$$\begin{aligned}
\log p(\mathcal{D}) &= \sum_n y_n \log \sigma({\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n) + (1-y_n) \log (1-\sigma({\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n))\end{aligned}$$

It is convenient to work with the negation of the log likelihood known
as the cross-entropy error function

$$\begin{aligned}
\mathcal{E}({\boldsymbol{w}}) &= \sum_n -y_n \log \sigma({\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n) - (1-y_n) \log (1-\sigma({\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n))\end{aligned}$$

We find the minimum of the cross-entropy error using the stationary
point condition

$$\begin{aligned}
\frac{\partial \mathcal{E}({\boldsymbol{w}})}{\partial {\boldsymbol{w}}} &= \sum_n -y_n[1- \sigma({\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n)]{\boldsymbol{x}}_n - (1-y_n)\sigma({\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n){\boldsymbol{x}}_n\\
0 &= \sum_n [\sigma({\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n) - y_n]{\boldsymbol{x}}_n\end{aligned}$$

Note that there is no closed-form analytical solution to identify
${\boldsymbol{w}}$ from the stationary point condition. Thus we use
numerical optimization. Gradient decent works, but second-order Newton’s
method works swimmingly. However inverting the hessian makes Newton’s
method unscalable. We can show that the cross-entropy function is
convex, thus a numerical optimization can guarantee a global optima. We
start by computing the hessian matrix

$$\begin{aligned}
{\boldsymbol{H}} &= \frac{\partial^2 \mathcal{E}({\boldsymbol{w}})}{\partial {\boldsymbol{w}}{\boldsymbol{w}}{^{\textrm T}}}\\
&= \sum_n \sigma({\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n)[1-\sigma({\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n)]{\boldsymbol{x}}_n{\boldsymbol{x}}_n{^{\textrm T}}\\
&= \sum_n (\alpha_n {\boldsymbol{x}}_n)(\alpha_n {\boldsymbol{x}}_n){^{\textrm T}}\end{aligned}$$

where
$\alpha_n = \sqrt{\sigma({\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n)[1-\sigma({\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n)]}$.

For any vector, ${\boldsymbol{v}}$,

$$\begin{aligned}
{\boldsymbol{v}}{^{\textrm T}}{\boldsymbol{H}}{\boldsymbol{v}} &=  \sum_n {\boldsymbol{v}} {^{\textrm T}}(\alpha_n {\boldsymbol{x}}_n)(\alpha_n {\boldsymbol{x}}_n){^{\textrm T}}{\boldsymbol{v}} \\
&= \sum_n [\alpha_n{\boldsymbol{v}}{^{\textrm T}}{\boldsymbol{x}}_n]^2 \ge 0\end{aligned}$$

and thus, this optimization is convex which guarantees our solution is a
global optima.

Multi-class Classification with Binary Logistic Regression
----------------------------------------------------------

You have two choices for using binary classifiers for multi-class
classification\
1) “one vs rest”: Train K classifiers, one for each class to
discriminate between that class and the rest of the classes. On a new
sample, predict with all K classifiers and choose the one with the
highest probability. This method is beneficial if you have many classes.

2\) “one vs one”: Train “K choose 2” classifiers, one for each pair of
classes to discriminate between them. On a new sample, predict with all
classifiers and choose the class that was predicted most often. This
method is beneficial if you have lot of data, because you are training
on a subset of the data – only the data for the two classes.

Multinomial Logistic Regression
-------------------------------

Multinomial logistic regression is used for multi-class classification
and is a simple extension of logistic regression. The conditional
likelihood is given by

$$\begin{aligned}
p(y_n = c_k|{\boldsymbol{x}}_n) &= \frac{e^{{\boldsymbol{w}}_k{^{\textrm T}}{\boldsymbol{x}}_n}}{\sum_{k'} e^{{\boldsymbol{w}}_{k'}{^{\textrm T}}{\boldsymbol{x}}_n}}\end{aligned}$$

This is called the <span>*softmax*</span> function.

Also, since we are no longer doing binary classification, we need to use
one-hot encoding for the target vector,
${\boldsymbol{y}}_n \sim {\mathbb{R}}^{K \times 1}$ such that

$$\begin{aligned}
y_{nk} = \begin{cases} 1 & \text{if } y_n = k \\ 0 & \text{otherwise} \end{cases}\end{aligned}$$

And now the conditional log likelihood

$$\begin{aligned}
\log P(\mathcal{D}) &= \sum_n \log P({\boldsymbol{y}}_n|{\boldsymbol{x}}_n)\\
&= \sum_n \log \prod_k P(y_{nk}=1|{\boldsymbol{x}}_n)^{y_{nk}}\\
&= \sum_n \sum_k y_{nk} \log P(y_{nk}=1|{\boldsymbol{x}}_n)\\
\mathcal{E}({\boldsymbol{w}}_1, {\boldsymbol{w}}_2, \hdots, {\boldsymbol{w}}_K) &= -\sum_n \sum_k y_{nk} \log P(y_{nk}=1|{\boldsymbol{x}}_n)\\\end{aligned}$$

The cross-entropy function is convex and can therefore has a unique
global optimum. Optimization requires numerical techniques analogous to
those used for binary logistic regression, but large-scale
implementation of multinomial logistic regression is non-trivial both
for the number of classes and the number of training samples.

Gaussian Discriminant Analysis (GDA)
====================================

Gaussian discriminant analysis is a generative classification model
(funny, its called “discriminant” analysis). The primary benefit of GDA
is it is a parametric classification model that has a closed form
analytical solution, unlike logistic regression.

Linear Discriminant Analysis (LDA)
----------------------------------

Linear discriminant analysis is very similar to logistic regression.
Given a dataset with two classes (a binary classification problem), we
great a generative model by fitting gaussians to the data. However, for
LDA we must assume the same variance. Thus, we must must maximize the
log likelihood to find the parameters ${\boldsymbol{\mu}}_1$,
${\boldsymbol{\mu}}_2$ and $\sigma$.

$$\begin{aligned}
\log P(\mathcal{D}|{\boldsymbol{\mu}}_1, {\boldsymbol{\mu}}_2,\sigma) = \sum_{n:y_n=1} \log \left(p_1 \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{({\boldsymbol{x}}_n - {\boldsymbol{\mu}}_1)^2}{2\sigma^2}}\right) + \sum_{n:y_n=2} \log \left(p_1 \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{({\boldsymbol{x}}_n - {\boldsymbol{\mu}}_2)^2}{2\sigma^2}}\right)\end{aligned}$$

Where $p_i = 1/N \sum_n {\mathbb{I}}(y_n == c_i) = N_i/N$. LDA is
analogous to logistic regression because we will have a linear decision
boundary given by

$$\begin{aligned}
-\frac{({\boldsymbol{x}} - {\boldsymbol{\mu}}_1)^2}{2\sigma^2} + \log p_1 &= -\frac{({\boldsymbol{x}} - {\boldsymbol{\mu}}_2)^2}{2\sigma^2} + \log p_2\\
\frac{{\boldsymbol{\mu}}_2 - {\boldsymbol{\mu}}_1}{\sigma^2}{\boldsymbol{x}} + \left( \frac{{\boldsymbol{\mu}}_1^2 - {\boldsymbol{\mu}}_2^2}{2\sigma^2}  - \log p_1 + \log p_2\right) &= 0\\
{\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}} + {\boldsymbol{b}} &= 0\end{aligned}$$

If we decouple the bias term, ${\boldsymbol{b}}$, from
${\boldsymbol{w}}$ in logistic regression (remove the leading 1 from
${\boldsymbol{x}}$), such that
$p(y_n=c|{\boldsymbol{x}}_n) =  \sigma({\boldsymbol{b}} + {\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n)$
for logistic regression, then we can compute the equivalent logistic
regression parameters

$$\begin{aligned}
{\boldsymbol{w}} &= {\boldsymbol{\Sigma}}^{-1}({\boldsymbol{\mu}}_2 - {\boldsymbol{\mu}}_1)\\
{\boldsymbol{b}} &= \frac{1}{2}({\boldsymbol{\mu}}_1{^{\textrm T}}{\boldsymbol{\Sigma}}^{-1}{\boldsymbol{\mu}}_1 - {\boldsymbol{\mu}}_2{^{\textrm T}}{\boldsymbol{\Sigma}}^{-1}{\boldsymbol{\mu}}_2) - \log \frac{p_2}{p_1}\end{aligned}$$

The benefit of LDA over logistic regression is that the parameters can
be estimated analytically in closed-form. However, the closed-form LDA
is the optimal solution. In fact, LDA is rarely ever used in practice
because constraining the variances to be the same is in fact a harder
problem than QDA. LDA is explored simply to contrast with logistic
regression. In practice, GDA usually refers to QDA.

Quadratic Discriminant Analysis (QDA)
-------------------------------------

Quadratic discriminant analysis is a generative model in the same way as
LDA except we do not assume the covariances to be the same. This allows
us to directly compute the parameters of the gaussians for each class.
Different covariance matrices gives rise to a quadratic decision
boundary. If we do not assume the same covariances, the log likelihood
decomposes into independent optimizations for generating a gaussian for
each class!

In practice, for both LDA and QDA, you will never care to compute the
decision boundary. You will approximate the data with gaussians, then
compute the probability that new sample belongs to each gaussian
(class), and choose the the class with the highest probability.

GDA vs Logistic Regression
--------------------------

GDA makes strong modeling assumptions and is more efficient both in
amount of training data necessary and computation. If the modeling
assumptions are correct (the data is gaussian) GDA is most certainly
better. However logistic regression makes weaker assumptions and
significantly more robust to deviations from modeling assumptions – it
finds the optimal linear boundary. When data is non-gaussian, logistic
regression will almost always be better. If the data is poisson,
logistic regression is very good.

Kernel Methods
==============

Kernel methods involve using a nonlinear basis function in your model
and leveraging the <span>*kernel trick*</span>.

A kernel function $k(\cdot, \cdot)$ is the inner product between two
nonlinear basis functions.

$$\begin{aligned}
k({\boldsymbol{x}}_1, {\boldsymbol{x}}_2) &= {\boldsymbol{\phi}}({\boldsymbol{x}}_1){^{\textrm T}}{\boldsymbol{\phi}}({\boldsymbol{x}}_2) \\
&= {\boldsymbol{\phi}}({\boldsymbol{x}}_2){^{\textrm T}}{\boldsymbol{\phi}}({\boldsymbol{x}}_1) \end{aligned}$$

The kernel matrix is a matrix of kernel functions defined by

$$\begin{aligned}
{\boldsymbol{K}} &= {\boldsymbol{\Phi}}{\boldsymbol{\Phi}}{^{\textrm T}}\\
&= \left[ \begin{array}{c c c c} k({\boldsymbol{x}}_1, {\boldsymbol{x}}_1) & k({\boldsymbol{x}}_1, {\boldsymbol{x}}_2) & \hdots & k({\boldsymbol{x}}_1, {\boldsymbol{x}}_N) \\ k({\boldsymbol{x}}_2, {\boldsymbol{x}}_1) & k({\boldsymbol{x}}_2, {\boldsymbol{x}}_2) & \hdots & k({\boldsymbol{x}}_2, {\boldsymbol{x}}_N) \\ \hdots & \hdots & \hdots & \hdots \\ k({\boldsymbol{x}}_N, {\boldsymbol{x}}_1) & k({\boldsymbol{x}}_N, {\boldsymbol{x}}_2) & \hdots & k({\boldsymbol{x}}_N, {\boldsymbol{x}}_N) \end{array} \right]\end{aligned}$$

The kernel matrix ${\boldsymbol{K}}$ is positive semi-definite and
symmetric and ${\boldsymbol{K}} \in {\mathbb{R}}^{N \times N}$ and
doesn’t depend on the dimension of the basis function!

Some other notation

$$\begin{aligned}
{\boldsymbol{\Phi}}{^{\textrm T}}&= ({\boldsymbol{\phi}}({\boldsymbol{x}}_1), {\boldsymbol{\phi}}({\boldsymbol{x}}_2), \hdots ,{\boldsymbol{\phi}}({\boldsymbol{x}}_N))\end{aligned}$$

${\boldsymbol{\Phi}} \in {\mathbb{R}}^{N \times M}$ where $M$ is the
dimensionality of the basis function,
${\boldsymbol{\phi}}({\boldsymbol{x}})$.

$$\begin{aligned}
{\boldsymbol{\Phi}}{\boldsymbol{x}} &= \left[ \begin{array}{c} k({\boldsymbol{x}}_1, {\boldsymbol{x}}) \\ k({\boldsymbol{x}}_2, {\boldsymbol{x}}) \\ \vdots \\ k({\boldsymbol{x}}_N, {\boldsymbol{x}}) \end{array} \right]\\
&= {\boldsymbol{k}}_x\end{aligned}$$

What most compelling about the kernel trick is that since we do not need
to compute the basis function to compute the kernel, we our kernel
function can have an infinite dimensional basis function! Here is a
simple example illustrating the point. Suppose we have the following
mapping

$$\begin{aligned}
{\boldsymbol{\psi}}_\theta({\boldsymbol{x}}) &= \left[ \begin{array}{c} \cos(\theta x_1) \\ \sin(\theta x_1) \\ \cos(\theta x_2) \\ \sin(\theta x_2) \end{array} \right]\end{aligned}$$

Now lets consider the basis function

$$\begin{aligned}
{\boldsymbol{\phi}}_L({\boldsymbol{x}}) &= \left[ \begin{array}{c}  {\boldsymbol{\psi}}_0({\boldsymbol{x}}) \\ {\boldsymbol{\psi}}_{\frac{2\pi}{L}}({\boldsymbol{x}}) \\ {\boldsymbol{\psi}}_{2\frac{2\pi}{L}}({\boldsymbol{x}}) \\ \vdots \\ {\boldsymbol{\psi}}_{L\frac{2\pi}{L}}({\boldsymbol{x}})  \end{array} \right]\end{aligned}$$

where $L \in [0,2\pi]$. Can we compute the inner product as
$L \to \infty$?

$$\begin{aligned}
{\boldsymbol{\phi}}_\infty({\boldsymbol{x}}){^{\textrm T}}{\boldsymbol{\phi}}_\infty({\boldsymbol{x}}) &= \lim_{L \to \infty} {\boldsymbol{\phi}}_L({\boldsymbol{x}}){^{\textrm T}}{\boldsymbol{\phi}}_L({\boldsymbol{x}})\\
&= \int_0^{2\pi} \cos(\theta(x_{m1} - x_{n1})) + \cos(\theta(x_{m2} - x_{n2}))\; d\theta \\
&= 1 - \frac{\sin (2\pi(x_{m1} - x_{n1}))}{x_{m1} - x_{n1}} + 1 - \frac{\sin (2\pi(x_{m2} - x_{n2}))}{x_{m2} - x_{n2}}\end{aligned}$$

This inner product of an infinite-dimensional feature space is finite
and thus computable!

In practice, it is very difficult, if not impossible, to compute a basis
function from a kernel function. A very popular kernel function is known
as the Gaussian kernel, Radial Basis Function (RBF) kernel, or Gaussian
RBF kernel

$$\begin{aligned}
k({\boldsymbol{x}}_m, {\boldsymbol{x}}_n) &= e^{- \|{\boldsymbol{x}}_m - {\boldsymbol{x}}_n\|_2^2 / 2\sigma^2}\end{aligned}$$

Composing new kernels is somewhat of a “black art” since we cannot back
out the basis function. What is so beneficial about the kernel trick is
it allows our model infinite complexity / dimensionality. The
regularization term therefore regulates overfitting. The kernel trick is
not possible without regularization.

Kernelized Ridge Regression
---------------------------

The cross-entropy error function for ridge regression with a nonlinear
basis function is given by

$$\begin{aligned}
\mathcal{E}({\boldsymbol{w}}) &= \frac{1}{2}\sum_n(y_n - {\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{\phi}}({\boldsymbol{x}}_n))^2 + \frac{\lambda}{2}\|{\boldsymbol{w}}\|_2^2\end{aligned}$$

And the ${\boldsymbol{w}}^{MAP}$ solution is given by

$$\begin{aligned}
\frac{\partial \mathcal{E}({\boldsymbol{w}})}{\partial {\boldsymbol{w}}} &= -\sum_n(y_n - {\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{\phi}}({\boldsymbol{x}}_n)){\boldsymbol{\phi}}({\boldsymbol{x}}_n) + \lambda{\boldsymbol{w}}\\
{\boldsymbol{w}}^{MAP} &= \sum_n \frac{1}{\lambda}(y_n - {\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{\phi}}({\boldsymbol{x}}_n)){\boldsymbol{\phi}}({\boldsymbol{x}}_n)\\
&= \sum_n \alpha_n {\boldsymbol{\phi}}({\boldsymbol{x}}_n) \\
&= {\boldsymbol{\Phi}}{^{\textrm T}}{\boldsymbol{\alpha}}\end{aligned}$$

$\alpha_n =  \frac{1}{\lambda}(y_n - {\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{\phi}}({\boldsymbol{x}}_n))$
but we do not know the vector of all $\alpha_n$’s,
${\boldsymbol{\alpha}}$.

Next, we substitute this solution
${\boldsymbol{w}}^{MAP} = {\boldsymbol{\Phi}}{^{\textrm T}}{\boldsymbol{\alpha}}$
back into $\mathcal{E}({\boldsymbol{w}})$.

$$\begin{aligned}
\mathcal{E}({\boldsymbol{w}}) &= \frac{1}{2}\sum_n(y_n - {\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{\phi}}({\boldsymbol{x}}_n))^2 + \frac{\lambda}{2}\|{\boldsymbol{w}}\|_2^2\\
 &= \frac{1}{2} \|{\boldsymbol{y}} - {\boldsymbol{\Phi}}{\boldsymbol{w}}\|_2^2 + \frac{\lambda}{2}\|{\boldsymbol{w}}\|_2^2\\
 &= \frac{1}{2} \|{\boldsymbol{y}} - {\boldsymbol{\Phi}}{\boldsymbol{\Phi}}{^{\textrm T}}{\boldsymbol{\alpha}}\|_2^2 + \frac{\lambda}{2}\|{\boldsymbol{\Phi}}{^{\textrm T}}{\boldsymbol{\alpha}}\|_2^2\\
\mathcal{E}({\boldsymbol{\alpha}}) &= \frac{1}{2} {\boldsymbol{\alpha}}{^{\textrm T}}{\boldsymbol{\Phi}}{\boldsymbol{\Phi}}{^{\textrm T}}{\boldsymbol{\Phi}}{\boldsymbol{\Phi}}{^{\textrm T}}{\boldsymbol{\alpha}} - ({\boldsymbol{\Phi}}{\boldsymbol{\Phi}}{^{\textrm T}}{\boldsymbol{y}}){^{\textrm T}}{\boldsymbol{\alpha}} + \frac{\lambda}{2}{\boldsymbol{\alpha}}{^{\textrm T}}{\boldsymbol{\Phi}}{\boldsymbol{\Phi}}{^{\textrm T}}{\boldsymbol{\alpha}}\\
&= \frac{1}{2} {\boldsymbol{\alpha}}{^{\textrm T}}{\boldsymbol{K}}^2{\boldsymbol{\alpha}} - ({\boldsymbol{K}}{\boldsymbol{y}}){^{\textrm T}}{\boldsymbol{\alpha}} + \frac{\lambda}{2}{\boldsymbol{\alpha}}{^{\textrm T}}{\boldsymbol{K}}{\boldsymbol{\alpha}}\end{aligned}$$

Note that we drop the ${\boldsymbol{y}}{^{\textrm T}}{\boldsymbol{y}}$
because it is a constant and will not effect the cross-entropy
minimization. Now lets derive the optimal ${\boldsymbol{\alpha}}$ from
the cross-entropy error function using the stationary point condition

$$\begin{aligned}
\frac{ \partial \mathcal{E}({\boldsymbol{\alpha}})}{\partial {\boldsymbol{\alpha}}} &=  {\boldsymbol{K}}^2{\boldsymbol{\alpha}} - {\boldsymbol{K}}{\boldsymbol{y}} + \lambda{\boldsymbol{K}}{\boldsymbol{\alpha}} = 0\\
{\boldsymbol{\alpha}} &= ({\boldsymbol{K}} + \lambda {\boldsymbol{I}})^{-1}{\boldsymbol{y}}\end{aligned}$$

Note that the solution to ${\boldsymbol{\alpha}}$ depends on
${\boldsymbol{K}}$ and not directly on
${\boldsymbol{\phi}}({\boldsymbol{x}})$! So long as you know how to
compute the kernel function, you don’t even need to know the basis
function. More to come on this, but for now, we want back out the
${\boldsymbol{w}}^{MAP}$ solution

$$\begin{aligned}
{\boldsymbol{w}}^{MAP} &= {\boldsymbol{\Phi}}{^{\textrm T}}({\boldsymbol{K}} + \lambda {\boldsymbol{I}})^{-1}{\boldsymbol{y}}\end{aligned}$$

Then, for prediction

$$\begin{aligned}
{\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{\phi}}({\boldsymbol{x}}) &={\boldsymbol{y}}{^{\textrm T}}({\boldsymbol{K}} + \lambda {\boldsymbol{I}})^{-1}  {\boldsymbol{\Phi}} {\boldsymbol{x}} \\
 &={\boldsymbol{y}}{^{\textrm T}}({\boldsymbol{K}} + \lambda {\boldsymbol{I}})^{-1} {\boldsymbol{k}}_x \\\end{aligned}$$

Note that to make a prediction, once again, we only need to know the
kernel function!

To summarize, first we must come up with a kernel function that
satisfies

$$\begin{aligned}
k({\boldsymbol{x}}_1, {\boldsymbol{x}}_2) &= k({\boldsymbol{x}}_2, {\boldsymbol{x}}_1)\end{aligned}$$

Then we can calculate ${\boldsymbol{\alpha}}$

$$\begin{aligned}
{\boldsymbol{\alpha}} &= ({\boldsymbol{K}} + \lambda {\boldsymbol{I}})^{-1}{\boldsymbol{y}}\end{aligned}$$

And then we can make predictions

$$\begin{aligned}
f({\boldsymbol{x}}) &={\boldsymbol{y}}{^{\textrm T}}({\boldsymbol{K}} + \lambda {\boldsymbol{I}})^{-1} {\boldsymbol{k}}_x\end{aligned}$$

Kernelized Nearest Neighbor Classifier
--------------------------------------

Is
$d({\boldsymbol{x}}_m, {\boldsymbol{x}}_n) = \|{\boldsymbol{x}}_n - {\boldsymbol{x}}_m\|_2^2$
a kernel function?

$$\begin{aligned}
d({\boldsymbol{x}}_m, {\boldsymbol{x}}_n) &= \|{\boldsymbol{x}}_n - {\boldsymbol{x}}_m\|_2^2\\
&= {\boldsymbol{x}}_m{^{\textrm T}}{\boldsymbol{x}}_m + {\boldsymbol{x}}_m{^{\textrm T}}{\boldsymbol{x}}_m - 2{\boldsymbol{x}}_m{^{\textrm T}}{\boldsymbol{x}}_n\\
&= k({\boldsymbol{x}}_m,{\boldsymbol{x}}_m) + k({\boldsymbol{x}}_n, {\boldsymbol{x}}_n) - 2k({\boldsymbol{x}}_m,{\boldsymbol{x}}_n)\\\end{aligned}$$

The summation of kernel functions and the product of kernel functions
are also kernel functions. Thus, this distance is a kernel function! We
have thus derived the kerneled nearest neighbor classifier as

$$\begin{aligned}
y_n = \text{arg min}_n k({\boldsymbol{x}}, {\boldsymbol{x}}_n)\end{aligned}$$

And now you can use different kernel functions as well to transform the
data into a different basis, perhaps an infinite basis! This can be
easily extended for kernelized K-nearest neighbor classifier.

Gaussian Process
----------------

Gaussian process is the term given to kernelized Bayesian linear
regression. We start by assuming the same prior as before

$$\begin{aligned}
p({\boldsymbol{w}}) &\sim N({\boldsymbol{0}}, \alpha^{-1}{\boldsymbol{I}})\end{aligned}$$

This is equivalent to putting a prior on an infinite set of functions,
$f_{{\boldsymbol{w}}}({\boldsymbol{x}})$.

We use this same idea to derive the gaussian process,
$\mathcal{GP}(\cdot)$.

$$\begin{aligned}
f({\boldsymbol{x}}) &\sim \mathcal{GP}(\cdot)\end{aligned}$$

Here, were are saying that this function is a gaussian random process.
This implies a joint distribution for every sample in the dataset

$$\begin{aligned}
p(f({\boldsymbol{x}}_1), f({\boldsymbol{x}}_2), \hdots, f({\boldsymbol{x}}_N)) &\sim N({\boldsymbol{\mu}}({\boldsymbol{x}}), {\boldsymbol{C}}({\boldsymbol{x}}))\end{aligned}$$

Same as with Bayesian linear regression.

For simplicity, we choose
${\boldsymbol{\mu}}({\boldsymbol{x}}) = {\boldsymbol{0}}$. Intuitively,
as in Bayesian linear regression, the expected value of the prior’s
prediction is zero. ${\boldsymbol{C}}({\boldsymbol{x}})$ is a covariance
matrix. Namely, a kernel matrix!

Lacking a good derivation, here is the predictive distribution

$$\begin{aligned}
p(y|{\boldsymbol{x}}, \mathcal{D}) &= N({\boldsymbol{k_x}}{^{\textrm T}}{\boldsymbol{K}}^{-1}{\boldsymbol{y}}, k({\boldsymbol{x}}, {\boldsymbol{x}}) - {\boldsymbol{k_x}}{^{\textrm T}}{\boldsymbol{K}}^{-1}{\boldsymbol{k_x}})\end{aligned}$$

Support Vector Machines
=======================

Support vector machines are a kernelized method used for classification.
Before discussing support vector machines, it helps to understand the
perceptron.

Perceptron
----------

The perceptron algorithm does binary classification. Suppose the two
classes are $y_n \in \{-1, +1\}$ and we have a linear discriminant
predictive function

$$\begin{aligned}
f({\boldsymbol{x}}_n) &= \text{sign}({\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n)\end{aligned}$$

Our goal is to reduce the cross-entropy error function

$$\begin{aligned}
\mathcal{E}({\boldsymbol{w}}) &= \sum_n {\mathbb{I}}(y_n \ne \text{sign}({\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n))\end{aligned}$$

The solution is to iterate through the data and

$$\begin{aligned}
\text{if } y_n = \text{sign}({\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n) & \;\;\;\text{  do nothing}\\
\text{if } y_n \ne \text{sign}({\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{x}}_n) &\;\;\;\; {\boldsymbol{w}} \gets {\boldsymbol{w}} + y_n{\boldsymbol{x}}_n\end{aligned}$$

If the training data is linearly separable, the algorithm stops in a
finite amount of time. Also, the parameter vector is always a linear
combination of the training samples.

The problem with the perceptron is that it is not a high margin
classifier. Once the algorithm converges, the decision boundary may be
able separate the data perfectly, but that boundary is arbitrary and may
come very close to misclassifying some sample by finding a sub-optimal
solution. The solution to this is to not use a 1-0 loss function, rather
a hinge-loss function.

Hinge-Loss
----------

The hinge loss function ensures a margin for the linear discriminant
(decision boundary).

$$\begin{aligned}
L(f({\boldsymbol{x}}),y) &= \begin{cases} 0 & \text{if } y\;f({\boldsymbol{x}}) \ge 1 \\ 1 - y\;f({\boldsymbol{x}}) & \text{otherwise} \end{cases}\\
&= \text{max}(0, 1-y\;f({\boldsymbol{x}}))\end{aligned}$$

This is known as a high margin loss function and makes for a high margin
classifier. This is because the hinge-loss function still penalizes the
weights for samples that classified correctly but are close to the
decision boundary.

SVM Derivation
--------------

SVMs do classification based on a hinge loss of a nonlinear basis
function discriminant. The cross-entropy error function is defined by

$$\begin{aligned}
\mathcal{E}({\boldsymbol{w}}) &= \sum_n \text{max}(0, 1- y_n[{\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{\phi}}({\boldsymbol{x}}_n) + b]) + \frac{\lambda}{2}\|{\boldsymbol{w}}\|_2^2\\
 &= C\sum_n \text{max}(0, 1- y_n[{\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{\phi}}({\boldsymbol{x}}_n) + b]) + \frac{1}{2}\|{\boldsymbol{w}}\|_2^2
 \end{aligned}$$

It is common to use $C = 1/\lambda$. We also rewrite with slack
variables

$$\begin{aligned}
 \xi_n &= \text{max}(0, 1- y_n[{\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{\phi}}({\boldsymbol{x}}_n) + b])
 \end{aligned}$$

Using slack variables allows us to formulate a constrained
differentiable convex optimization problem.

Now we are left with the <span>*primal formulation*</span> (note the
simple change in notation – $f(\cdot)$ is the primal objective
function).

$$\begin{aligned}
\min_{{\boldsymbol{w}}, b, \{\xi_{n}\}} f(\{{\boldsymbol{x}}_{n}\}) =
\min_{{\boldsymbol{w}}, b, \{\xi_{n}\}} & C \sum_{n} \xi_{n} + \frac{1}{2}\|{\boldsymbol{w}}\|_{2}^{2}\\
s.t. \;\; &1 - y_{n}[{\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{\phi}}({\boldsymbol{x}}_{n}) + b] \le \xi_{n}, \; \;\forall n \\
&\xi_{n} \ge 0\end{aligned}$$

To solve this constrained optimization problem, we introduce the
Lagrangian. $\{\alpha_{n}\}$ and $\{\lambda_{n}\}$ are Lagrange
multipliers ensuring the constraints.

$$\begin{aligned}
&L(\{{\boldsymbol{x}}_{n}\}, {\boldsymbol{w}}, b, \{\xi_{n}\}, \{\alpha_{n}\}, \{\lambda_{n}\}) \\&= C \sum_{n} \xi_{n} + \frac{1}{2}\|{\boldsymbol{w}}\|_{2}^{2} - \sum_{n} \lambda_{n} \xi_{n} + \sum_{n} \alpha_{n} \left(1 - y_{n}[{\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{\phi}}({\boldsymbol{x}}_{n}) + b] - \xi_{n} \right)\\
&= C \sum_{n} \xi_{n} + \frac{1}{2}\|{\boldsymbol{w}}\|_{2}^{2} - \sum_{n} \lambda_{n} \xi_{n} + \sum_{n} \alpha_{n}  - \sum_{n} \alpha_{n} y_{n}{\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{\phi}}({\boldsymbol{x}}_{n}) - \sum_{n} \alpha_{n} y_{n} b - \sum_{n} \alpha_{n} \xi_{n}\\
&= \sum_{n} (C - \lambda_{n} - \alpha_{n}) \xi_{n} + \frac{1}{2}\|{\boldsymbol{w}}\|_{2}^{2}  + \sum_{n} \alpha_{n}  - \sum_{n} \alpha_{n} y_{n}{\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{\phi}}({\boldsymbol{x}}_{n}) - b\sum_{n} \alpha_{n} y_{n}\end{aligned}$$

To solve the primal formulation we need to perform a $max$, then a $min$
on the Lagrangian. This is difficult.

$$\begin{aligned}
L(\{{\boldsymbol{x}}_{n}\},{\boldsymbol{w}}, b, \{\xi_{n}\}, \{\alpha_{n}\}, \{\lambda_{n}\}) &\le f(\{{\boldsymbol{x}}_{n}\})\\
\min_{{\boldsymbol{w}}, b, \{\xi_{n}\}}  \;\;\max_{ \{\alpha_{n}\} , \{\lambda_{n}\}} L(\{{\boldsymbol{x}}_{n}\},{\boldsymbol{w}}, b, \{\xi_{n}\}, \{\alpha_{n}\}, \{\lambda_{n}\}) &= \min_{{\boldsymbol{w}}, b, \{\xi_{n}\}}f(\{{\boldsymbol{x}}_{n}\})\end{aligned}$$

Flipping $min$ and $max$ gives rise to the dual formulation which gives
yields (in most cases) a different result which is a lower bound on the
optimal primal result (for more on lagrange duality, see section
[sec:lagrange-duality]).

$$\begin{aligned}
g( \{\alpha_{n}\}, \{\lambda_{n}\}) &= \min_{{\boldsymbol{w}}, b, \{\xi_{n}\}}  L(\{{\boldsymbol{x}}_{n}\},{\boldsymbol{w}}, b, \{\xi_{n}\}, \{\alpha_{n}\}, \{\lambda_{n}\})\\
 \max_{ \{\alpha_{n}\} , \{\lambda_{n}\}}g( \{\alpha_{n}\}, \{\lambda_{n}\}) &=  \max_{ \{\alpha_{n}\} , \{\lambda_{n}\}} \min_{{\boldsymbol{w}}, b, \{\xi_{n}\}}  L(\{{\boldsymbol{x}}_{n}\},{\boldsymbol{w}}, b, \{\xi_{n}\}, \{\alpha_{n}\}, \{\lambda_{n}\})\end{aligned}$$

Note that

$$\begin{aligned}
 \max_{ \{\alpha_{n}\} , \{\lambda_{n}\}}g( \{\alpha_{n}\}, \{\lambda_{n}\}) &\le  \min_{{\boldsymbol{w}}, b, \{\xi_{n}\}} f(\{{\boldsymbol{x}}_{n}\})\end{aligned}$$

Where $g( \{\alpha_{n}\}, \{\lambda_{n}\})$ is the dual objective
function and $f(\{{\boldsymbol{x}}_{n}\})$ is the primal objective
function. This discrepancy is called the duality gap.

Continuing to derive the dual formulation, we minimize $L$ using the
stationary point condition with respect to the primal variables

$$\begin{aligned}
\frac{dL}{d{\boldsymbol{w}}} &= {\boldsymbol{w}} - \sum_{n} y_{n} \alpha_{n} {\boldsymbol{\phi}}({\boldsymbol{x}}_{n}) = 0 \\
\frac{dL}{db} &= \sum_{n}  \alpha_{n} y_{n} = 0 \\
\frac{dL}{d\xi_{n}} &= C - \alpha_{n} - \lambda_{n} = 0 \end{aligned}$$

Note here that one of the constraints solves for the primal weight
variable. Substitute the back to find the dual objective function:

$$\begin{aligned}
g( \{\alpha_{n}\}, \{\lambda_{n}\}) &= \min_{{\boldsymbol{w}}, b, \{\xi_{n}\}} L({\boldsymbol{w}}, b, \{\xi_{n}\}, \{\alpha_{n}\}, \{\lambda_{n}\})\\
 &= \cancel{\sum_{n} (C - \lambda_{n} - \alpha_{n}) \xi_{n}} + \frac{1}{2}\|{\boldsymbol{w}}\|_{2}^{2}  + \sum_{n} \alpha_{n}  - \sum_{n} \alpha_{n} y_{n}{\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{\phi}}({\boldsymbol{x}}_{n}) - \cancel{b\sum_{n} \alpha_{n} y_{n}}\\
 &= \frac{1}{2}\|\sum_{n} y_{n} \alpha_{n} {\boldsymbol{\phi}}({\boldsymbol{x}}_{n})\|_{2}^{2}  + \sum_{n} \alpha_{n}  - \sum_{n} \alpha_{n} y_{n}\left(\sum_{m} y_{m} \alpha_{m} {\boldsymbol{\phi}}({\boldsymbol{x}}_{m}) \right){^{\textrm T}}{\boldsymbol{\phi}}({\boldsymbol{x}}_{n}) \\
  &=  \sum_{n} \alpha_{n}  - \frac{1}{2}\sum_{mn} \alpha_{n} \alpha_{m}y_{n}y_{m}  {\boldsymbol{\phi}}({\boldsymbol{x}}_{m}){^{\textrm T}}{\boldsymbol{\phi}}({\boldsymbol{x}}_{n}) \end{aligned}$$

Now we have our dual formulation subject to the Lagrangian constraints
we previously derived:

$$\begin{aligned}
 \max_{ \{\alpha_{n}\} \in \mathbb{R}^{+}, \{\lambda_{n}\}\in \mathbb{R}^{+}} &g( \{\alpha_{n}\}, \{\lambda_{n}\}) =  \sum_{n} \alpha_{n}  - \frac{1}{2}\sum_{mn} \alpha_{n} \alpha_{m}y_{n}y_{m}  {\boldsymbol{\phi}}({\boldsymbol{x}}_{m}){^{\textrm T}}{\boldsymbol{\phi}}({\boldsymbol{x}}_{n})\\
 s.t. \;\;\;& \alpha_{n} \ge 0 \;\; \forall n\\
& \lambda_{n} \ge 0 \;\; \forall n\\
 & \sum_{n}  \alpha_{n} y_{n} = 0 \\
 & C - \alpha_{n} - \lambda_{n} = 0 \;\; \forall n\end{aligned}$$

We can clean this up a little bit

$$\begin{aligned}
\lambda_{n} &\ge 0\\
C - \alpha_{n} - \lambda_{n} &= 0\\
\lambda_{n}& = C - \alpha_n\\
\implies \alpha_{n} &\le C\\\end{aligned}$$

The final form of the the dual formulation is

$$\begin{aligned}
 \max_{ \{\alpha_{n}\}, \{\lambda_{n}\}} & \sum_{n} \alpha_{n}  - \frac{1}{2}\sum_{mn} \alpha_{n} \alpha_{m}y_{n}y_{m}  k({\boldsymbol{x}}_{m},{\boldsymbol{x}}_{n})\\
 s.t. \;\;\;& 0 \le \alpha_{n} \le C \;\; \forall\; n\\
 & \sum_{n}  \alpha_{n} y_{n} = 0\end{aligned}$$

One obvious benefit of the dual formulation is the use of a kernel
function! This is a quadratic programming problem and can be solved
using MATLAB’s <span>*quadprog()*</span> function.

Analysis
--------

One of the primal variables has already been derived while solving the
lagrangian, that is

$$\begin{aligned}
{\boldsymbol{w}} = \sum_{n} y_{n} \alpha_{n} {\boldsymbol{\phi}}({\boldsymbol{x}}_{n})\end{aligned}$$

To solve for $b$, we look at complementary slackness. Complementary
slackness says the lagrange multipliers times the constrains in the
lagrangian must equal zero at the optimal solution to both primal and
dual

$$\begin{aligned}
&\lambda_{n} \xi_{n} = 0\\
 &\alpha_{n} \left(1 - y_{n}[{\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{\phi}}({\boldsymbol{x}}_{n}) + b] - \xi_{n} \right) = 0\\\end{aligned}$$

From the first condition, $\alpha_n < C$,

$$\begin{aligned}
&\lambda_n = C - \alpha_n > 0\\
&\implies \xi_n = 0\end{aligned}$$

And if we assume that $\alpha_n \ne 0$, then

$$\begin{aligned}
&1 - y_{n}[{\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{\phi}}({\boldsymbol{x}}_{n}) + b] - \xi_{n} = 0\\
&\implies b = y_n - {\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{\phi}}({\boldsymbol{x}}_{n})\end{aligned}$$

for $y \in \{-1, +1\}$.

From this analysis, we can deduce from
$ {\boldsymbol{w}} = \sum_{n} y_{n} \alpha_{n} {\boldsymbol{\phi}}({\boldsymbol{x}}_{n})$
that the solution is dependent only those samples whose corresponding
$\alpha_n > 0$. These samples are called <span>*support vectors*</span>.
This allows us to throw out all of the training data for those
$\alpha_n = 0$, shrinking the amount of data we need to carry around.

From complementary slackness we see that for those support vectors
($\alpha_n >0$):\
- if $\xi_n = 0$, then
$y_{n}[{\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{\phi}}({\boldsymbol{x}}_{n}) + b] = 1$.
These are points are correctly classified and are right on the hinge of
the loss function, or $1/\|{\boldsymbol{w}}\|_2$ away from the decision
boundary.\
- if $\xi_n < 1$, then
$y_{n}[{\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{\phi}}({\boldsymbol{x}}_{n}) + b] > 0$.
These points are classified correctly, but are not outside the margin of
the hinge-loss function.\
- if $\xi_n > 1$, then
$y_{n}[{\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{\phi}}({\boldsymbol{x}}_{n}) + b] < 0$.
These points are misclassified.

As you can see, the support vectors are only those samples that are
non-zero in the hinge-loss function. This means samples that are
misclassified or within the margin of the decision boundary.

To predict the classification of a new sample,

$$\begin{aligned}
y &= \text{sign}({\boldsymbol{w}}{^{\textrm T}}{\boldsymbol{\phi}}({\boldsymbol{x}}) + b)\\
&= \text{sign}\left(\left[ \sum_{n} y_{n} \alpha_{n} {\boldsymbol{\phi}}({\boldsymbol{x}}_{n})\right]{^{\textrm T}}{\boldsymbol{\phi}}({\boldsymbol{x}}) + b\right)\\
&= \text{sign}\left( \sum_{n} y_{n} \alpha_{n} k({\boldsymbol{x}}_{n},{\boldsymbol{x}}) + b\right)\\\end{aligned}$$

Adaboost
========

Adaboost stands for adaptive boosting. Boosting combines a lot of
classifiers in a greedy way to construct a more powerful classifier and
more complex decision boundaries. Since boosting creates a cascade of
classifiers, it is best to use weak classifiers that are quick and easy
to solve.

Our boosting algorithm prediction function will be

$$\begin{aligned}
h({\boldsymbol{x}}) = \text{sign}\left(\sum_t \beta_t h_t({\boldsymbol{x}})\right)\end{aligned}$$

where $\beta_t$ a weight on each weak classifier
$h_t({\boldsymbol{x}})$.

Here’s how it works. Every data sample has a weighted exponential-loss.
Initially, all of their weights are the same, $w_1(n) = 1/N$. Train a
weak classifier, $h_t({\boldsymbol{x}})$ by minimizing the weighted
classification error

$$\begin{aligned}
\epsilon_t &= \sum_n w_t(n){\mathbb{I}}(y_n \ne h_t({\boldsymbol{x}}_n))\end{aligned}$$

Calculate the weight of this classifier

$$\begin{aligned}
\beta_t &= \frac{1}{2}\log \frac{1-\epsilon_t}{\epsilon_t}\end{aligned}$$

Update the weights on each data sample based on the weighted
exponential-loss

$$\begin{aligned}
w_{t+1}(n) \gets w_t(n)e^{-\beta_t y_n h_t({\boldsymbol{x}}_n)}\end{aligned}$$

Then normalize the them

$$\begin{aligned}
w_{t+1}(n) \gets \frac{w_{t+1}(n)}{\sum_n w_{t+1}(n)} \end{aligned}$$

Now, the sample weights are exponentially weighted so that misclassified
samples are weighted more. This is called a greedy algorithm. Loop back
to training a new weak classifier, $h_{t+1}({\boldsymbol{x}})$. Continue
until convergence.

K-Means Clustering
==================

K-means is an unsupervised learning algorithm that attempts to cluster
data into K clusters. The objective function we wish to minimize is

$$\begin{aligned}
J &= \sum_n \sum_k r_{nk} \|{\boldsymbol{x}}_n - {\boldsymbol{\mu}}_k \|_2^2\end{aligned}$$

where ${\boldsymbol{\mu}}_k$ is the centroid of each cluster and
$r_{nk} \in \{0,1\}$ is an indicator variable where $r_{nk} = 1$ if and
only if
$y_n = k = \text{arg min}_k \|{\boldsymbol{x}}_n - {\boldsymbol{\mu}}_k \|_2$,
sample $n$ is closest to centroid $k$.

This algorithm works by initializing the ${\boldsymbol{\mu}}_k$’s
randomly. Determine the classification of all samples, $r_{nk}$. Then
update ${\boldsymbol{\mu}}_k$ to be the centroid of all the samples
belonging to cluster $k$. Loop back to re-classifying all of the samples
again. Iterate until convergence.

Note that K-means does not guarantee the procedure terminates at a
global optimum. In practice, K-means is run multiple times and the best
solution is used.

Also, determining K is non-trivial. It is a hyperparameter, but it does
not suffice to do cross-validation because the optimal $K = N$.

Gaussian Mixture Model (GMM)
============================

GMM is a generative unsupervised clustering algorithm. Suppose we have a
dataset made of up of multiple gaussian distributions, but they overlap.
K-means does not return a probability of belonging to each cluster, only
a classification. GMMs try to solve this problem by estimating the
gaussians and returning probabilities of belonging to each cluster.

A GMM has the following density function

$$\begin{aligned}
p({\boldsymbol{x}}) = \sum_k w_k N({\boldsymbol{x}}|{\boldsymbol{\mu}}_k, {\boldsymbol{\Sigma}}_k)\end{aligned}$$

Each gaussian has a mean, covariance and weight associated with that
cluster, $w_k$. Because $p({\boldsymbol{x}})$ is a probability,
$w_k \ge 0 \; \forall \; k$ and $\sum_k w_k = 1$.

This optimization is going to be tricky because of the constraints on
the parameters. Namely, that $\sum_k w_k = 1$ and
${\boldsymbol{\Sigma}}_k$ must be positive semi-definite.

So, lets suppose we know the classification of each. Let $z_n$ denote
the classification for each sample. Thus
$w_k = p(z=k) = p(z_n=k) \; \forall \; n$. Now consider the joint
distribution

$$\begin{aligned}
p({\boldsymbol{x}}, z) &= p(z)p({\boldsymbol{x}}|z)\end{aligned}$$

Then the conditional distribution is

$$\begin{aligned}
p({\boldsymbol{x}}|z=k) &= N({\boldsymbol{x}}|{\boldsymbol{\mu}}_k, {\boldsymbol{\Sigma}}_k)\end{aligned}$$

and the marginal distribution is

$$\begin{aligned}
p({\boldsymbol{x}}) = \sum_k w_k N({\boldsymbol{x}}|{\boldsymbol{\mu}}_k, {\boldsymbol{\Sigma}}_k)\end{aligned}$$

The <span>*complete*</span> likelihood is

$$\begin{aligned}
\sum_n \log p({\boldsymbol{x}}_n, z_n) &= \sum_n \log p(z_n)p({\boldsymbol{x}}_n|z_n) \\
&= \sum_k \sum_{n:z_n=k}\log p(z_n)p({\boldsymbol{x}}_n|z_n)\end{aligned}$$

Let is define $\gamma_{nk} \in \{0,1\}$ to indicate whether $z_n=k$.
Then we can write

$$\begin{aligned}
\sum_n \log p({\boldsymbol{x}}_n, z_n) &= \sum_k \sum_{n} \gamma_{nk}[\log w_k + \log N({\boldsymbol{x}}_n|{\boldsymbol{\mu}}_k, {\boldsymbol{\Sigma}}_k)]\\
&= \sum_k \sum_{n} \gamma_{nk}\log w_k + \sum_k \left[ \sum_{n} \gamma_{nk}\log N({\boldsymbol{x}}_n|{\boldsymbol{\mu}}_k, {\boldsymbol{\Sigma}}_k)\right]\end{aligned}$$

After taking the derivative and solving the maximum likelihood
estimation, we come to a rather intuitive solution

$$\begin{aligned}
w_k &= \frac{\sum_n\gamma_{nk}}{\sum_k \sum_n \gamma_{nk}}\\
{\boldsymbol{\mu}}_k &= \frac{1}{\sum_n \gamma_{nk}} \sum_n \gamma_{nk}{\boldsymbol{x}}_n\\
{\boldsymbol{\Sigma}}_k &= \frac{1}{\sum_n \gamma_{nk}} \sum_n \gamma_{nk}({\boldsymbol{x}}_n - {\boldsymbol{\mu}}_k)({\boldsymbol{x}}_n - {\boldsymbol{\mu}}_k){^{\textrm T}}\end{aligned}$$

This is nice and all, but we aren’t given $z_n$, so we can compute them
given the posterior probability

$$\begin{aligned}
p(z_n=k|{\boldsymbol{x}}_n) &= \frac{p({\boldsymbol{x}}_n|z_n=k)p(z_n=k)}{p({\boldsymbol{x}}_n)}\\
&= \frac{p({\boldsymbol{x}}_n|z_n=k)p(z_n=k)}{\sum_{k'} p({\boldsymbol{x}}_n|z_n=k')p(z_n=k')}\end{aligned}$$

Note that we need to know all the parameters to be able to calculate the
posterior $p(z_n=k|{\boldsymbol{x}}_n)$.

To get around this, first we are going to assume a <span>*soft*</span>
$\gamma_{nk}$ meaning that rather than $\gamma_{nk}$ being binary, it is
the posterior probability

$$\begin{aligned}
\gamma_{nk} &= p(z_n=k|{\boldsymbol{x}}_n)\end{aligned}$$

Now we can come up with a simple iterative algorithm to find a solution.
First initialize random parameters
$\theta = \{ \{ {\boldsymbol{\mu}}_k \}, \{ {\boldsymbol{\Sigma}}_k \}, \{ w_k \} \}$.
Then we compute the $\gamma_{nk}$’s. Then we update $\theta$ based on
the new $\gamma_{nk}$’s and loop back over. Note that this optimization
is not convex. This solution can be derived from the expectation
maximization algorithm.

Expectation Maximization (EM) Algorithm
---------------------------------------

In general, EM is used to estimate parameters for probabilistic models
with hidden/latent variables

$$\begin{aligned}
p(x|\theta) &= \sum_z p(x,z|\theta)\end{aligned}$$

where $x$ is observed, $\theta$ are the model parameters, and $z$ is
hidden. To obtain the maximum likelihood estimate of $\theta$

$$\begin{aligned}
\theta &= \text{arg max}_\theta \; l(\theta)\\
 &= \text{arg max}_\theta \sum_n \log p(x_n|\theta)\\
&= \text{arg max}_\theta \sum_n \log \sum_{z_n} p(x_n, z_n|\theta)\end{aligned}$$

$l(\theta)$ is called the <span>*incomplete*</span> log-likelihood. The
difficulty with the incomplete log-likelihood is that it needs to sum
over all possible hidden variables and then take the logarithm. This
log-sum format makes computation intractable. Thus the EM algorithm
leverages a clever trick to change this into sum-log by changing this
into the expected (<span>*complete*</span>) log-likelihood

$$\begin{aligned}
Q_q(\theta) &= \sum_n {\mathbb{E}}_{z_n \sim q(z_n)} \log p(x_n, z_n | \theta) \\
&= \sum_n \sum_{z_n} q(z_n)\log p(x_n, z_n | \theta)\end{aligned}$$

Now if we choose the distribution of $z$ to be the posterior
distribution, $q(z) = p(z|x,\theta)$, then we define

$$\begin{aligned}
Q(\theta) &= Q_{z \sim p(z|x, \theta)}(\theta)\\
&= \sum_n \sum_{z_n} p(z|x, \theta) \log p(x_n, z_n | \theta)\\
&= \sum_n \sum_{z_n} p(z|x, \theta)[ \log p(x_n | \theta) + \log p(z_n| x_n, \theta)]\\
&= \sum_n \sum_{z_n} p(z|x, \theta) \log p(x_n | \theta) + \sum_n \sum_{z_n} p(z|x, \theta)\log p(z_n| x_n, \theta)\\
&= \sum_n  \log p(x_n | \theta) \sum_{z_n}p(z|x, \theta)+ \sum_n{\mathbb{H}}[p(z|x, \theta)]\\
&= \sum_n  \log p(x_n | \theta)+ \sum_n{\mathbb{H}}[p(z|x, \theta)]\\
&=  l(\theta)+ \sum_n{\mathbb{H}}[p(z|x, \theta)]\end{aligned}$$

Where ${\mathbb{H}}[p(x)] = - \int p(x) \log p(x) dx$ is known as the
entropy of the probabilistic distribution, $p(x)$. As before, we need to
know the parameters, $\theta$, to compute the posterior probability
$p(z|x, \theta)$. Thus we will use a known $\theta^{OLD}$ to compute the
expected likelihood.

$$\begin{aligned}
Q(\theta,\theta^{OLD} )&= \sum_n \sum_{z_n} p(z|x, \theta^{OLD}) \log p(x_n, z_n | \theta)\end{aligned}$$

It can be shown that

$$\begin{aligned}
l(\theta) &\ge Q(\theta,\theta^{OLD} )+  \sum_n{\mathbb{H}}[p(z|x, \theta^{OLD})]\\
&\ge A(\theta,\theta^{OLD} )\end{aligned}$$

and thus we have a lower lower bound on the log-likelihood defined by
the <span>*auxiliary function*</span>, $A(\theta,\theta^{OLD} )$. An
important property of the auxiliary function is that
$A(\theta,\theta ) = l(\theta)$.

Thus we want to maximize the auxiliary function

$$\begin{aligned}
\theta^{NEW} = \text{arg max}_\theta A(\theta,\theta^{OLD} )\end{aligned}$$

and repeat this process such that

$$\begin{aligned}
\theta^{(t+1)} \gets \text{arg max}_\theta A(\theta,\theta^{(t)} )\end{aligned}$$

However, this maximization step does not depend on the entropy term, so

$$\begin{aligned}
\theta^{(t+1)} \gets \text{arg max}_\theta Q(\theta,\theta^{(t)} )\end{aligned}$$

Thus, the EM algorithm procedure iteratively predicts the posterior
probability (the E-step),

$$\begin{aligned}
p(z|x, \theta^{OLD})\end{aligned}$$

and then updated the parameters in by the maximization (the M-step)

$$\begin{aligned}
\theta^{(t+1)} \gets \text{arg max}_\theta Q(\theta,\theta^{(t)} )\end{aligned}$$

The EM algorithm converges, but only to a local optima – a global optima
is not guaranteed to be found.

EM for GMM
----------

For the E-step, we simply compute

$$\begin{aligned}
\gamma_{nk} &= p(z_n=k|{\boldsymbol{x}}_n, {\boldsymbol{\theta}})\\
&= \frac{p({\boldsymbol{x}}_n|z_n=k, {\boldsymbol{\theta}})p(z_n=k)}{\sum_{k'} p({\boldsymbol{x}}_n|z_n=k', {\boldsymbol{\theta}})p(z_n=k')}\\
&= \frac{ w_k N({\boldsymbol{x}}|{\boldsymbol{\mu}}_k, {\boldsymbol{\Sigma}}_k)}{\sum_{k'}  w_{k'} N({\boldsymbol{x}}|{\boldsymbol{\mu}}_{k'}, {\boldsymbol{\Sigma}}_{k'})}\end{aligned}$$

Then for the M-step

$$\begin{aligned}
Q({\boldsymbol{\theta}},{\boldsymbol{\theta}}^{old}) &= \sum_n \sum_k p(z_n=k|{\boldsymbol{x}}_n, {\boldsymbol{\theta}}^{OLD}) \log  p({\boldsymbol{x}}_n , z_n = k| {\boldsymbol{\theta}})\\
&= \sum_n \sum_k \gamma_{nk}  \log p({\boldsymbol{x}}_n, z_n = k | {\boldsymbol{\theta}})\\
&= \sum_n \sum_k \gamma_{nk}  \log p({\boldsymbol{x}}_n | z_n = k, {\boldsymbol{\theta}}) \; p(z_n = k | {\boldsymbol{\theta}})\\
 &= \sum_n \sum_k \gamma_{nk}   \log N({\boldsymbol{x}}_n | {\boldsymbol{\mu}}_k, {\boldsymbol{\Sigma}}_k) \; \omega_k \\
 &= \sum_n \sum_k \gamma_{nk}\log  N({\boldsymbol{x}}_n | {\boldsymbol{\mu}}_k, {\boldsymbol{\Sigma}}_k) + \gamma_{nk} log \;  \omega_k\end{aligned}$$

To find the optimal ${\boldsymbol{\mu}}_k$ and
${\boldsymbol{\Sigma}}_k$, we use the stationary point condition

$$\begin{aligned}
\frac{d}{d{\boldsymbol{\mu}}_k}Q({\boldsymbol{\theta}},{\boldsymbol{\theta}}^{old}) &= \frac{d}{d{\boldsymbol{\mu}}_k}\sum_n \sum_k \gamma_{nk}log \; N({\boldsymbol{x}}_n | {\boldsymbol{\mu}}_k, {\boldsymbol{\Sigma}}_k) + \gamma_{nk} log \;  \omega_k\\
0 &= \frac{d}{d{\boldsymbol{\mu}}_k} \sum_n \sum_k \gamma_{nk}log \; \left((2 \pi)^{-d/2} |{\boldsymbol{\Sigma}}_k|^{-1/2} \text{exp}\{-\frac{1}{2}({\boldsymbol{x}}_n - {\boldsymbol{\mu}}_k){^{\textrm T}}{\boldsymbol{\Sigma}}_k ({\boldsymbol{x}}_n - {\boldsymbol{\mu}}_k)\}\right)\\
0 &= \frac{d}{d{\boldsymbol{\mu}}_k} \sum_n \sum_k -\gamma_{nk}\frac{1}{2}({\boldsymbol{x}}_n - {\boldsymbol{\mu}}_k){^{\textrm T}}{\boldsymbol{\Sigma}}_k ({\boldsymbol{x}}_n - {\boldsymbol{\mu}}_k)\\
0 &=  \sum_n \gamma_{nk}\frac{1}{2} {\boldsymbol{\Sigma}}_k ({\boldsymbol{x}}_n - {\boldsymbol{\mu}}_k)\\
\sum_n \gamma_{nk}  {\boldsymbol{\mu}}_k &=  \sum_n \gamma_{nk}{\boldsymbol{x}}_n\\
{\boldsymbol{\mu}}_k &=  \frac{\sum_n \gamma_{nk}{\boldsymbol{x}}_n}{\sum_n \gamma_{nk}}\\
{\boldsymbol{\mu}}_k &=  \frac{1}{N_k}\sum_n \gamma_{nk}{\boldsymbol{x}}_n\\\end{aligned}$$

$$\begin{aligned}
\frac{d}{d{\boldsymbol{\Sigma}}_k}Q({\boldsymbol{\theta}},{\boldsymbol{\theta}}^{old}) &= \frac{d}{d{\boldsymbol{\Sigma}}_k}\sum_n \sum_k \gamma_{nk}log \; N({\boldsymbol{x}}_n | {\boldsymbol{\mu}}_k, {\boldsymbol{\Sigma}}_k) + \gamma_{nk} log \;  \omega_k\\
0 &=\frac{d}{d{\boldsymbol{\Sigma}}_k} \sum_n \sum_k \gamma_{nk}log \; \left((2 \pi)^{-d/2} |{\boldsymbol{\Sigma}}_k|^{-1/2} \text{exp}\{-\frac{1}{2}({\boldsymbol{x}}_n - {\boldsymbol{\mu}}_k){^{\textrm T}}{\boldsymbol{\Sigma}}_k ({\boldsymbol{x}}_n - {\boldsymbol{\mu}}_k)\}\right)\\
0 &=\frac{d}{d{\boldsymbol{\Sigma}}_k} \sum_n \sum_k \gamma_{nk}log \; |{\boldsymbol{\Sigma}}_k|^{-1/2}  - \frac{d}{d{\boldsymbol{\Sigma}}_k} \sum_n \sum_k \gamma_{nk} \frac{1}{2}({\boldsymbol{x}}_n - {\boldsymbol{\mu}}_k){^{\textrm T}}{\boldsymbol{\Sigma}}_k ({\boldsymbol{x}}_n - {\boldsymbol{\mu}}_k)\\
0 &=\frac{d}{d{\boldsymbol{\Sigma}}_k} \sum_n \sum_k \gamma_{nk}log \; |{\boldsymbol{\Sigma}}_k|  + \frac{d}{d{\boldsymbol{\Sigma}}_k} \sum_n \sum_k \gamma_{nk} ({\boldsymbol{x}}_n - {\boldsymbol{\mu}}_k){^{\textrm T}}{\boldsymbol{\Sigma}}_k ({\boldsymbol{x}}_n - {\boldsymbol{\mu}}_k)\\
0 &= \left({\boldsymbol{\Sigma}}_k^{-1}\right){^{\textrm T}}\sum_n \gamma_{nk} + \sum_n \gamma_{nk} ({\boldsymbol{x}}_n - {\boldsymbol{\mu}}_k) ({\boldsymbol{x}}_n - {\boldsymbol{\mu}}_k){^{\textrm T}}\\
0 &= {\boldsymbol{\Sigma}}_k N_k + \sum_n \gamma_{nk} ({\boldsymbol{x}}_n - {\boldsymbol{\mu}}_k) ({\boldsymbol{x}}_n - {\boldsymbol{\mu}}_k){^{\textrm T}}\\
{\boldsymbol{\Sigma}}_k  &= \frac{1}{N_k} \sum_n \gamma_{nk} ({\boldsymbol{x}}_n - {\boldsymbol{\mu}}_k) ({\boldsymbol{x}}_n - {\boldsymbol{\mu}}_k){^{\textrm T}}\\\end{aligned}$$

To find the optimal $\omega_k$ we must use Lagrange to constrain
$\sum_{k} \omega_k = 1$.

$$\begin{aligned}
\mathcal{L}(\lambda,{\boldsymbol{\theta}})  &= Q({\boldsymbol{\theta}},{\boldsymbol{\theta}}^{old}) - \lambda\left( \sum_{k} \omega_k - 1\right)\end{aligned}$$

And now we solve for the optimal parameters

$$\begin{aligned}
\frac{d\mathcal{L}(\lambda,{\boldsymbol{\theta}}) }{d\omega_k} &= \frac{d}{d\omega_k}\left[\sum_n \sum_k \gamma_{nk}\left(log \; N({\boldsymbol{x}}_n | {\boldsymbol{\mu}}_k, {\boldsymbol{\Sigma}}_k) + log \;  \omega_k\right)  - \lambda\left( \sum_{k} \omega_k - 1\right)\right]\\
0 &=  \frac{\sum_n \gamma_{nk}}{\omega_k} - \lambda\\
\omega_k &=  \frac{\sum_n \gamma_{nk}}{\lambda}\\
\sum_k \omega_k &=  1\\
1 &= \sum_k \frac{\sum_n \gamma_{nk}}{\lambda}\\
\lambda &= \sum_k \sum_n \gamma_{nk}\\
\omega_k &=  \frac{\sum_n \gamma_{nk}}{\sum_k \sum_n \gamma_{nk}}\\
\omega_k &=  \frac{N_k}{\sum_k N_k}\\\end{aligned}$$

Dimensionality Reduction
========================

Very high dimensional data will make algorithms slower and even
intractable.

Principle Component Analysis (PCA)
----------------------------------

Often, data may be highly dimensional but highly correlated. If there
are correlated features, it is beneficial to reduce the dimensionality
of the data to have only uncorrelated features.

One way of deriving PCA is to maximize the projected variance in the
data. This derivation assumes the data has zero-mean and the projection
vector is given by ${\boldsymbol{u}}$.

Variance is given by (given the centered data assumption)

$$\begin{aligned}
{\boldsymbol{\Sigma}} = \frac{1}{N} {\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{X}}\end{aligned}$$

Thus the formulation is

$$\begin{aligned}
\max_{{\boldsymbol{u}}} {\boldsymbol{u}}{^{\textrm T}}{\boldsymbol{\Sigma}}{\boldsymbol{u}} = \max_{{\boldsymbol{u}}} \frac{1}{N} {\boldsymbol{u}}{^{\textrm T}}{\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{X}}{\boldsymbol{u}}\end{aligned}$$

but we must constrain ${\boldsymbol{u}}$ so that it doesn’t become
arbitrarily large. Thus

$$\begin{aligned}
\max_{{\boldsymbol{u}}} \;\;&\frac{1}{N} {\boldsymbol{u}}{^{\textrm T}}{\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{X}}{\boldsymbol{u}}\\
\text{s.t.}\;\;& \|{\boldsymbol{u}}\|_2^2 = 1\end{aligned}$$

Solve using the lagrangian

$$\begin{aligned}
\mathcal{L}(\lambda,{\boldsymbol{u}}) = \frac{1}{N} {\boldsymbol{u}}{^{\textrm T}}{\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{X}}{\boldsymbol{u}} + \lambda(1 - {\boldsymbol{u}}{^{\textrm T}}{\boldsymbol{u}})\end{aligned}$$

when solving for the optima, we see that $(\lambda,{\boldsymbol{u}})$ is
an eigenvalue-eigenvector pair!

$$\begin{aligned}
\frac{\partial \mathcal{L}(\lambda,{\boldsymbol{u}})}{\partial {\boldsymbol{u}}} &= 2{\boldsymbol{\Sigma}}{\boldsymbol{u}} - 2\lambda{\boldsymbol{u}} = 0\\
{\boldsymbol{\Sigma}}{\boldsymbol{u}} &= \lambda{\boldsymbol{u}}\\\end{aligned}$$

plugging back to the original formulation

$$\begin{aligned}
\max_{{\boldsymbol{u}}} {\boldsymbol{u}}{^{\textrm T}}\lambda{\boldsymbol{u}}\end{aligned}$$

We see that ${\boldsymbol{u}}$ must be the eigenvector with the largest
eigenvalue.

Its not a stretch to see that to project into multiple dimensions, $D$,
with maximal projected variance we will project using the eigenvectors
associated with the $D$ largest eigenvalues.

Note that if you do not center the data, you will not have maximized the
variance. However, funny enough, it still sort of works though you will
not get the same result as PCA. In fact, your first axis will likely be
very close to the mean vector.

Kernelized Principle Component Analysis (kPCA)
----------------------------------------------

Starting from where we left off with standard PCA

$$\begin{aligned}
{\boldsymbol{\Sigma}}{\boldsymbol{u}} &= \lambda{\boldsymbol{u}}\\
\frac{1}{N} {\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{X}}{\boldsymbol{u}} &= \lambda{\boldsymbol{u}}\\
{\boldsymbol{u}} &= \frac{1}{\lambda N} {\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{X}}{\boldsymbol{u}}\\
&= {\boldsymbol{X}}{^{\textrm T}}\left(\frac{1}{\lambda N} {\boldsymbol{X}}{\boldsymbol{u}}\right)\\
&= {\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{\alpha}}\end{aligned}$$

What is ${\boldsymbol{\alpha}}$? Plugging into for
${\boldsymbol{u}} = {\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{\alpha}}$

$$\begin{aligned}
\frac{1}{N} {\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{X}}{\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{\alpha}} &= \lambda{\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{\alpha}}\\
\frac{1}{N} {\boldsymbol{X}}{\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{X}}{\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{\alpha}} &= \lambda{\boldsymbol{X}}{\boldsymbol{X}}{^{\textrm T}}{\boldsymbol{\alpha}}\end{aligned}$$

Now replace ${\boldsymbol{X}}{\boldsymbol{X}}{^{\textrm T}}$ with a
kernel matrix, ${\boldsymbol{K}}$ and you have

$$\begin{aligned}
\frac{1}{N} {\boldsymbol{K}}{\boldsymbol{K}}{\boldsymbol{\alpha}} &= \lambda{\boldsymbol{K}}{\boldsymbol{\alpha}}\\
\frac{1}{N} {\boldsymbol{K}}{\boldsymbol{\alpha}} &= \lambda{\boldsymbol{\alpha}}\end{aligned}$$

${\boldsymbol{\alpha}}$ is the eigenvector with the largest associated
eigenvalue of the kernel matrix!

Subtle Issue 1: the kernel matrix must be centralized. This can be done
like so

$$\begin{aligned}
{\boldsymbol{K}}_{centered} = ({\boldsymbol{I}} - {\boldsymbol{1}}){\boldsymbol{K}}({\boldsymbol{I}} - {\boldsymbol{1}})\end{aligned}$$

Subtle Issue 2: to ensure $\|{\boldsymbol{u}}\|_2^2 = 1$, we must
rescale ${\boldsymbol{\alpha}}$ by $\frac{1}{\sqrt{\lambda N}}$.

Bias vs Variance Trade-off {#sec:bias-variance}
==========================

Bias vs Variance is the trade off between model complexity, over-fitting
and under-fitting. There are some strong theoretical proofs as well.

Lets assume a square-error loss function, and lets assume our prediction
function, $f(\cdot)$, is trained on the data, $\mathcal{D}$, and thus is
denoted $f_{\mathcal{D}}(\cdot)$. So, the expected risk is defined by

$$\begin{aligned}
R(f_{\mathcal{D}}) &= \int \int L(f_{\mathcal{D}}({\boldsymbol{x}}),y) \;p({\boldsymbol{x}},y) \;dy \;d{\boldsymbol{x}}\\
&=  \int \int (f_{\mathcal{D}}({\boldsymbol{x}})-y)^2 \;p({\boldsymbol{x}},y) \;dy \;d{\boldsymbol{x}}\\\end{aligned}$$

Lets define the averaged risk

$$\begin{aligned}
{\mathbb{E}}_{\mathcal{D}} R(f_{\mathcal{D}}) &=  \int \int \int (f_{\mathcal{D}}({\boldsymbol{x}})-y)^2 \;p({\boldsymbol{x}},y) \;dy \;d{\boldsymbol{x}}\; P(\mathcal{D})\;d\mathcal{D}\\\end{aligned}$$

This marginalized out the randomness with respect to the data,
$\mathcal{D}$ and the risk. Lets also define the averaged prediction

$$\begin{aligned}
{\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}}) &=  \int  f_{\mathcal{D}}({\boldsymbol{x}}) P(\mathcal{D})\;d\mathcal{D}\\\end{aligned}$$

Again, to marginalize out the randomness of the data on training the
model.

Now, lets add and subtract the averaged prediction

$$\begin{aligned}
{\mathbb{E}}_{\mathcal{D}} R(f_{\mathcal{D}}) &=  \int \int \int (f_{\mathcal{D}}({\boldsymbol{x}})-y)^2 \;p({\boldsymbol{x}},y) \;dy \;d{\boldsymbol{x}}\; P(\mathcal{D})\;d\mathcal{D}\\
&=  \int \int \int (f_{\mathcal{D}}({\boldsymbol{x}})-{\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}})  + {\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}}) -y)^2 \;p({\boldsymbol{x}},y) \;dy \;d{\boldsymbol{x}}\; P(\mathcal{D})\;d\mathcal{D}\\
&=  \int \int \int (f_{\mathcal{D}}({\boldsymbol{x}})-{\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}}))^2 \;p({\boldsymbol{x}},y) \;dy \;d{\boldsymbol{x}}\; P(\mathcal{D})\;d\mathcal{D}\\
&\;\;\;\;\;+ \int \int \int ({\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}}) -y)^2 \;p({\boldsymbol{x}},y) \;dy \;d{\boldsymbol{x}}\; P(\mathcal{D})\;d\mathcal{D}\\
&\;\;\;\;\;+  \int \int \int (f_{\mathcal{D}}({\boldsymbol{x}})-{\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}}))({\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}}) -y) \;p({\boldsymbol{x}},y) \;dy \;d{\boldsymbol{x}}\; P(\mathcal{D})\;d\mathcal{D}\\\end{aligned}$$

The third term equals zero

$$\begin{aligned}
\int \int \int (f_{\mathcal{D}}({\boldsymbol{x}})-{\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}}))({\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}}) -y) \;p({\boldsymbol{x}},y) \;dy \;d{\boldsymbol{x}}\; P(\mathcal{D})\;d\mathcal{D}\\
\int \int \cancel{\left[ \int (f_{\mathcal{D}}({\boldsymbol{x}})-{\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}})) \; P(\mathcal{D})\;d\mathcal{D}  \right]}({\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}}) -y) \;p({\boldsymbol{x}},y) \;dy \;d{\boldsymbol{x}}\\\end{aligned}$$

So now we are left with the two terms

$$\begin{aligned}
{\mathbb{E}}_{\mathcal{D}} R(f_{\mathcal{D}}) &=   \int \int \int [f_{\mathcal{D}}({\boldsymbol{x}})-{\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}})]^2 \;p({\boldsymbol{x}},y) \;dy \;d{\boldsymbol{x}}\; P(\mathcal{D})\;d\mathcal{D}\\
&\;\;\;\;\;+ \int \int \int [{\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}}) -y]^2 \;p({\boldsymbol{x}},y) \;dy \;d{\boldsymbol{x}}\; P(\mathcal{D})\;d\mathcal{D}\end{aligned}$$

Lets explore the second term further. It is no longer dependent on
$\mathcal{D}$ so lets get rid of that.

$$\begin{aligned}
&\int \int \int [{\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}}) -y]^2 \;p({\boldsymbol{x}},y) \;dy \;d{\boldsymbol{x}}\; P(\mathcal{D})\;d\mathcal{D}\\
&=\int \int  [{\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}}) -y]^2 \;p({\boldsymbol{x}},y) \;dy \;d{\boldsymbol{x}}\end{aligned}$$

To simplify further, we will use a similar trick by defining the the
averaged target

$$\begin{aligned}
{\mathbb{E}}_y y = \int y\; p(y|{\boldsymbol{x}})\;dy\end{aligned}$$

Plug it in

$$\begin{aligned}
&\int \int  [{\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}}) -y]^2 \;p({\boldsymbol{x}},y) \;dy \;d{\boldsymbol{x}}\\
&= \int \int  [{\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}}) - {\mathbb{E}}_y y + {\mathbb{E}}_y y -y]^2 \;p({\boldsymbol{x}},y) \;dy \;d{\boldsymbol{x}}\\
&= \int \int  [{\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}}) - {\mathbb{E}}_y y]^2 \;p({\boldsymbol{x}},y) \;dy \;d{\boldsymbol{x}} \\
&\;\;\;\;\;+ \int \int  [{\mathbb{E}}_y y - y]^2 \;p({\boldsymbol{x}},y) \;dy \;d{\boldsymbol{x}} \\
&\;\;\;\;\;+ \int \int  [{\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}}) - {\mathbb{E}}_y y][{\mathbb{E}}_y y -y] \;p({\boldsymbol{x}},y) \;dy \;d{\boldsymbol{x}}\end{aligned}$$

Again, the last term is zero

$$\begin{aligned}
\int \int  [{\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}}) - {\mathbb{E}}_y y][{\mathbb{E}}_y y -y] \;p({\boldsymbol{x}},y) \;dy \;d{\boldsymbol{x}}
&= \int  [{\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}}) - {\mathbb{E}}_y y]\left\{\int [{\mathbb{E}}_y y -y] p(y|{\boldsymbol{x}}) \; dy \; \right\} \;p({\boldsymbol{x}}) \;d{\boldsymbol{x}}\\
\int [{\mathbb{E}}_y y -y] p(y|{\boldsymbol{x}}) \; dy &= \int {\mathbb{E}}_y y p(y|{\boldsymbol{x}}) \; dy  - \int y p(y|{\boldsymbol{x}}) \; dy\\
&=  {\mathbb{E}}_y y \int p(y|{\boldsymbol{x}}) \; dy  - \int y p(y|{\boldsymbol{x}}) \; dy\\
&=  {\mathbb{E}}_y y - {\mathbb{E}}_y y\\
&=0\end{aligned}$$

And the first term simplifies by marginalizing $y$.

$$\begin{aligned}
&\int \int  [{\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}}) - {\mathbb{E}}_y y]^2 \;p({\boldsymbol{x}},y) \;dy \;d{\boldsymbol{x}}\\
&= \int  [{\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}}) - {\mathbb{E}}_y y]^2 \;p({\boldsymbol{x}}) \;d{\boldsymbol{x}}\end{aligned}$$

Finally, we are left with

$$\begin{aligned}
{\mathbb{E}}_{\mathcal{D}} R(f_{\mathcal{D}}) &=   \int \int \int [f_{\mathcal{D}}({\boldsymbol{x}})-{\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}})]^2 \;p({\boldsymbol{x}},y) \;dy \;d{\boldsymbol{x}}\; P(\mathcal{D})\;d\mathcal{D} \\ &\;\;\;\;\; +\int  [{\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}}) - {\mathbb{E}}_y y]^2 \;p({\boldsymbol{x}}) \;d{\boldsymbol{x}} \\&\;\;\;\;\; + \int \int  [{\mathbb{E}}_y y - y]^2 \;p({\boldsymbol{x}},y) \;dy \;d{\boldsymbol{x}} \\
{\mathbb{E}}_{\mathcal{D}} R(f_{\mathcal{D}}) &= \text{variance} + \text{bias}^2 + \text{noise}\end{aligned}$$

This first term is known as the <span>*variance*</span> of the model.

$$\begin{aligned}
\text{variance}&=\int \int \int [f_{\mathcal{D}}({\boldsymbol{x}})-{\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}})]^2 \;p({\boldsymbol{x}},y) \;dy \;d{\boldsymbol{x}}\; P(\mathcal{D})\;d\mathcal{D}\end{aligned}$$

There are two ways to reduce variance:\
1) Use a lot of data. Increasing $\mathcal{D}$ will decrease
$f_{\mathcal{D}}({\boldsymbol{x}})-{\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}})$.\
2) Use a simple model, $f(\cdot)$, so that $f_{\mathcal{D}}(\cdot)$ does
not vary much across datasets.

The second term is known as the <span>*bias*</span> of the model.

$$\begin{aligned}
\text{bias}^2 &= \int  [{\mathbb{E}}_{\mathcal{D}} f_{\mathcal{D}}({\boldsymbol{x}}) - {\mathbb{E}}_y y]^2 \;p({\boldsymbol{x}}) \;d{\boldsymbol{x}}\end{aligned}$$

We can reduce the bias by using more complex models allowing $f(\cdot)$
to be as flexible as possible to better approximate ${\mathbb{E}}_y y$.
However, this causes the variance to increase.

The third term is known as the <span>*noise*</span> of the mode.

$$\begin{aligned}
\text{noise} &= \int \int  [{\mathbb{E}}_y y - y]^2 \;p({\boldsymbol{x}},y) \;dy \;d{\boldsymbol{x}}\end{aligned}$$

There is nothing we can do about noise. Choosing $f(\cdot)$ or
$\mathcal{D}$ will not affect it.

As you can see, expected risk (error) breaks down into three terms. Bias
and variance both contribute to this error and fight each other over
model complexity/simplicity. Increasing the amount of training data will
help with the variance contribution to error. The noise is simply
inherent and nothing can be done about this.

Model Selection
===============

Given some model, $\mathcal{M}$, the Bayesian information criterion
(BIC) is used to approximate

$$\begin{aligned}
\log p(\mathcal{D}|\mathcal{M}) &\approx \log p(\mathcal{D}|{\boldsymbol{w}}^{MLE}) - \frac{D}{2}\log N\end{aligned}$$

Where ${\boldsymbol{w}}^{MLE}$ is the maximum likelihood estimation
solution to the parameters of the model. For linear regression, we have

$$\begin{aligned}
\text{BIC } &= -\frac{N}{2}\left(\frac{1}{N}\sum_n(y_n - {\boldsymbol{w}}^{MLE{^{\textrm T}}}{\boldsymbol{x}}_n)^2 \right) - \frac{D}{2}\log N\end{aligned}$$

Maximizing the BIC should give us a decent measure of whether our model
is too complicated or not complicated enough.

Hyperparameter Selection {#sec:hyperparameters}
========================

When training a model and comparing it against other models, it is
crucial to separate the training data from the test data. Training your
model on the data you test with will cause your model to overfit the
data and you have shown absolutely nothing.

When choosing hyperparameters for your model, it is crucial that you
tune these hyperparameters with data separate from the training data.
Well, there are two ways of going about this.

1\) Use training, cross-validation, and test data sets. None of these
data sets should overlap. Vary your hyperparameters and train your model
with the training dataset. Find the set of hyperparameters that give you
the minimal prediction error on the cross-validation set. Then run your
prediction on the test dataset and report that as the accuracy of your
model.

2\) Use training and test datasets that do not overlap and do a N-fold
cross-validation in your training data to tune your hyperparameters.
What this means is split your training data evenly into N groups. Train
your model on N-1 groups and use the left over group as the
cross-validation set. Do this N times with the same hyperparameters,
iterating over the groups so that each group takes a turn as the
cross-validation set. Then average these accuracies. This is the
cross-validation accuracy for one set of hyperparameters. Find an
optimal set of hyperparameters, then run your prediction on the test
dataset and report that as the accuracy of your model.
