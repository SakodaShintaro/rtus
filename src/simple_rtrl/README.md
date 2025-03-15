# Real-Time Recurrent Learningの実装

$$\newcommand{\bs}{\boldsymbol}$$

以下の論文を参考に実装をしてみます。

<https://arxiv.org/abs/2305.19044>

`flax.linen.SimpleCell` だとバイアスがあり、また実装する上では入力データなどが横ベクトルであるとした方が自然であるため（最初の次元にはバッチサイズが来ることが多いため）定式化が微妙に異なります。[以前](https://tokumini.hatenablog.com/entry/2025/02/23/120000)の繰り返しのようになりますが、RTRLの式から再度書いていきます。

## 定式化

まず時系列的な入力 $\bs{x}(t) \in \mathbb{R} ^ {D}$ を、内部状態 $\bs{s}(t), \bs{h}(t) \in \mathbb{R} ^ {N}$ を持つ再帰的ニューラルネットワーク(RNN)で処理することを考えます。入力に対する重みとバイアスを $\bs{W} \in \mathbb{R} ^ {D \times N}, \bs{B} \in \mathbb{R} ^ {N}$ および再帰成分の重みを $\bs{R} \in \mathbb{R} ^ {N \times N}$ 、要素ごとの活性化関数を $\sigma$ として

$$
\begin{align}
\bs{s}(t) &= \bs{x}(t)\bs{W} + \bs{B} + \bs{h}(t - 1)\bs{R} \\
\bs{h}(t) &= \sigma(\bs{s}(t))
\end{align}
$$

というRNNを定義します。

これを要素ごとの計算で書くと

$$
\begin{align}
\bs{s} _ k(t) = \left(\sum _ {d=1} ^ D \bs{x} _ {d}(t)\bs{W} _ {d, k}\right) + \bs{b} _ k + \sum _ {n=1} ^ N \sigma(\bs{s} _ n(t - 1))\bs{R} _ {n, k}
\end{align}
$$

となります。

Real-Time Recurrent Learningでは勾配を逐次的に求めます。たとえば重み $\bs{W}$ に注目して、時間 $T$ 分の勾配を考えると

$$
\begin{align}
\frac{\partial \mathcal{L} ^ \text{total} (1, T)}{\partial \bs{W} _ {i, j}} &= \sum _ {t=1} ^ T \frac{\partial \mathcal{L}(t)}{\partial \bs{W} _ {i, j}} \\
&= \sum _ {t=1} ^ T \left(\sum _ {k=1} ^ N
\frac{\partial \mathcal{L}(t)}{\partial \bs{s} _ k(t)} \times \frac{\partial \bs{s} _ k(t)}{\partial \bs{W} _ {i, j}}
\right)
\end{align}
$$

となります。 $\frac{\partial \mathcal{L}(t)}{\partial \bs{s} _ k(t)}$ は逆伝播をすれば求まるので、 $\frac{\partial \bs{s} _ k(t)}{\partial \bs{W} _ {i, j}}$ を保持しながら更新していけば良いというのがRTRLの考え方になります（これを感度行列と呼びます）。

これはRNNの式から

$$
\begin{align}
\frac{\partial \bs{s} _ k(t)}{\partial \bs{W} _ {i, j}} &= \frac{\partial}{\partial \bs{W} _ {i, j}} \left\lbrack \left(\sum _ {d=1} ^ D \bs{x} _ {d}(t)\bs{W} _ {d, k}\right) + \bs{b} _ k + \sum _ {n=1} ^ N \sigma(\bs{s} _ n(t - 1))\bs{R} _ {n, k} \right\rbrack \\
&= \bs{x} _ i (t) \mathbb{1} _ {j=k} + \sum _ {n = 1} ^ N \bs{R} _ {n, k} \sigma'(\bs{s} _ n (t - 1)) \frac{\partial \bs{s} _ n(t - 1)}{\partial \bs{W} _ {i, j}}\\
\end{align}
$$

と再帰的に求まります。 $\bs{B}, \bs{R}$ についても同様に考えると、

$\bs{B}$ について

$$
\begin{align}
\frac{\partial \bs{s} _ k(t)}{\partial \bs{B} _ {j}} = \mathbb{1} _ {j=k} + \sum _ {n = 1} ^ N \bs{R} _ {n, k} \sigma'(\bs{s} _ n (t - 1)) \frac{\partial \bs{s} _ n(t - 1)}{\partial \bs{B} _ {j}}
\end{align}
$$

$\bs{R}$ について

$$
\begin{align}
\frac{\partial \bs{s} _ k(t)}{\partial \bs{R} _ {i, j}} &= \sum _ {n = 1} ^ N
\left\lbrack
  \left(\sigma(\bs{s} _ n(t - 1))\frac{\partial \bs{R} _ {n, k}}{\partial \bs{R} _ {i, j}}\right)
  +
  \left(\bs{R} _ {n, k} \sigma'(\bs{s} _ n (t - 1)) \frac{\partial \bs{s} _ n(t - 1)}{\partial \bs{R} _ {i, j}}\right)
\right\rbrack \\
&= \left(\sigma(\bs{s} _ i (t - 1))\right) \mathbb{1} _ {j=k} + \sum _ {n = 1} ^ N
  \left(\bs{R} _ {n, k} \sigma'(\bs{s} _ n (t - 1)) \frac{\partial \bs{s} _ n(t - 1)}{\partial \bs{R} _ {i, j}}\right) \\
\end{align}
$$

となります。

## 実装

これらの式において $\mathbb{1} _ {j = k}$ は $j=k$ のときだけ $1$ 、それ以外の場合は $0$ となるものなので、単位行列との演算と考えます。各式の添字を素直に `jnp.einsum` に書き換えていきます。感度行列なので変数名は `S` として

(1) $\bs{W}$ について

$$
\begin{align}
\frac{\partial \bs{s} _ k(t)}{\partial \bs{W} _ {i, j}} &= \bs{x} _ i (t) \mathbb{1} _ {j=k} + \sum _ {n = 1} ^ N \bs{R} _ {n, k} \sigma'(\bs{s} _ n (t - 1)) \frac{\partial \bs{s} _ n(t - 1)}{\partial \bs{W} _ {i, j}}\\
\end{align}
$$

```python
S_W = jnp.einsum("bi,jk->bkij", x_t, eye) \
    + jnp.einsum("nk,bn,bnij->bkij", R, d_s, S_W)
```

(2) $\bs{B}$ について

$$
\begin{align}
\frac{\partial \bs{s} _ k(t)}{\partial \bs{B} _ {j}} = \mathbb{1} _ {j=k} + \sum _ {n = 1} ^ N \bs{R} _ {n, k} \sigma'(\bs{s} _ n (t - 1)) \frac{\partial \bs{s} _ n(t - 1)}{\partial \bs{B} _ {j}}
\end{align}
$$

```python
S_B = eye[None, :, :] + jnp.einsum("nk,bn,bnj->bkj", R, d_s, S_B)
```

(3) $\bs{R}$ について

$$
\begin{align}
\frac{\partial \bs{s} _ k(t)}{\partial \bs{R} _ {i, j}} = \left(\sigma(\bs{s} _ i (t - 1))\right) \mathbb{1} _ {j=k} + \sum _ {n = 1} ^ N
  \left(\bs{R} _ {n, k} \sigma'(\bs{s} _ n (t - 1)) \frac{\partial \bs{s} _ n(t - 1)}{\partial \bs{R} _ {i, j}}\right)
\end{align}
$$

```python
h_t_minus_1 = jnp.tanh(s_t_minus_1)
S_R = jnp.einsum("bi,jk->bkij", h_t_minus_1, eye) \
    + jnp.einsum("nk,bn,bnij->bkij", R, d_s, S_R)
```

というわけで、要素ごとの計算で書くと `einsum` での実装がそのまま書き写すだけになるため非常に便利でした。
