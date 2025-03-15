# 論文

<https://arxiv.org/abs/2305.19044> $\newcommand{\bs}{\boldsymbol}$

## 適当な式

$$
\begin{align}
\bs{s}(t) &= \bs{x}(t)\bs{W} + \bs{h}(t - 1)\bs{R} \\
\bs{h}(t) &= \sigma(\bs{s}(t))
\end{align}
$$

これを要素ごとの計算で書くと

$$
\begin{align}
\bs{s} _ k(t) = \sum _ {n=1} ^ D \bs{x} _ {n}(t)\bs{W} _ {k, n} + \sum _ {n=1} ^ N \sigma(\bs{s} _ n(t - 1))\bs{R} _ {k, n}
\end{align}
$$

である。

勾配は

$$
\begin{align}
\frac{\partial \mathcal{L} ^ \text{total} (1, T)}{\partial \bs{W} _ {i, j}} &= \sum _ {t=1} ^ T \frac{\partial \mathcal{L}(t)}{\partial \bs{W} _ {i, j}} \\
&= \sum _ {t=1} ^ T \left(\sum _ {k=1} ^ N \frac{\partial \mathcal{L}(t)}{\partial \bs{s} _ k(t)} \times \frac{\partial \bs{s} _ k(t)}{\partial \bs{W} _ {i, j}} \right)
\end{align}
$$

となる。ここで

$$
\begin{align}
\frac{\partial \bs{s} _ k(t)}{\partial \bs{W} _ {i, j}} = \bs{x} _ j (t) \mathbb{1} _ {k=i} + \sum _ {n = 1} ^ N \bs{R} _ {k, n} \sigma'(\bs{s} _ n (t - 1)) \frac{\partial \bs{s} _ n(t - 1)}{\partial \bs{W} _ {i, j}}
\end{align}
$$

と再帰的に求まる。同様に

$$
\begin{align}
\frac{\partial \bs{s} _ k(t)}{\partial \bs{R} _ {i, j}} = \sum _ {n = 1} ^ N
\left\lbrack
  \left(\sigma(\bs{s} _ n(t - 1))\frac{\partial \bs{R} _ {k, n}}{\partial \bs{R} _ {i, j}}\right)
  +
  \left(\bs{R} _ {k, n} \sigma'(\bs{s} _ n (t - 1)) \frac{\partial \bs{s} _ n(t - 1)}{\partial \bs{R} _ {i, j}}\right)
\right\rbrack
\end{align}
$$
