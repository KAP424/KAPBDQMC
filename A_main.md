# Boson-PQMC

## 两味试态

考虑两味玻色子 $\{\hat b_i, \hat c_i\}_{i=1}^{M}$。选取 **rank-1 试态**（每个味的所有玻色子占据同一个单粒子轨道 $P_i$）：

$$
|\Phi_T\rangle = |\Phi_T^b\rangle \otimes |\Phi_T^c\rangle
$$

其中：

$$
|\Phi_T^b\rangle = \frac{1}{\sqrt{N_b!}} \left( \sum_{i=1}^{M} P_i\hat b_i^\dagger  \right)^{N_b} |0\rangle
$$

$$
|\Phi_T^c\rangle = \frac{1}{\sqrt{N_b!}} \left( \sum_{i=1}^{M} P_i^* \hat c_i^\dagger  \right)^{N_b} |0\rangle
$$

符号约定：

- $P_i$（$i=1,\ldots,M$）是复系数，描述单粒子轨道的空间波形
- $N_b$ 是每种味的玻色子数（取两味相等）
- $P_i^*$ 出现在 $|\Phi_T^c\rangle$ 中——来自 TRS/RP 对称性的要求

## 内积性质 (proof.1)

$$
|R\rangle
=\frac{1}{\sqrt{N_b!}}
\Big(\sum_{i=1}^N R_i\,\hat b_i^\dagger\Big)^{N_b}|0\rangle,
\qquad
\langle L|
=\frac{1}{\sqrt{N_b!}}
\langle 0|
\Big(\sum_{i=1}^N L_i\,\hat b_i\Big)^{N_b}.
$$

$$
\boxed{\bra{L}\ket{R}= (\vec{L}\cdot\vec{R})^{N_b} }
$$

## 投影算符的辅助场分解--proof.2

投影算符 $e^{-2\theta \hat H}$ 做 Trotter 分解和 HS 退耦（与有限温 BAFQMC 完全相同的步骤）：

$$
e^{-2\theta \hat H} \longrightarrow \int \mathcal D\phi\, P_{\text{aux}}(\phi)\, \hat U(\phi)
$$

其中 $\hat U(\phi) = \prod_{\ell=1}^{N_\tau} \exp(\hat{\mathbf b}^\dagger B_\ell \hat{\mathbf b} + \hat{\mathbf c}^\dagger \bar B_\ell \hat{\mathbf c})$ 是数字守恒切片传播子的乘积。注意这里用的是数字守恒形式（$\Delta_\ell = 0$），与有限温数字守恒 BAFQMC 一致。

**关键简化**：由于 TRS，两味的传播子是复共轭关系：$\hat{\mathbf c}^\dagger$ 部分 = $\hat{\mathbf b}^\dagger$ 部分的复共轭。

定义**投影子传输矩阵**：

$$
\boxed{ B(\tau_2, \tau_1) = e^{-B(\tau_2)} e^{-B(\tau_2 - \Delta\tau)} \cdots e^{-B(\tau_1)}}
$$

这是从虚时 $\tau_1$ 到 $\tau_2$ 的单粒子传播子（注意：指数上是 $-B$ 而非 $+B$——来自 $\exp(\hat{\mathbf a}^\dagger B \hat{\mathbf a})$ 中 $B$ 的定义约定）。

$$
\boxed{\hat U_b(\phi) |\Phi_T^b\rangle = \frac{1}{\sqrt{N_b!}} \left( \sum_i \hat b_i^\dagger [\mathcal B(2\theta, 0) P]_i \right)^{N_b} |0\rangle}
$$

$$
\boxed{\langle \Phi_T^b | \hat U_b(\phi) | \Phi_T^b\rangle = \big(P^\dagger \mathcal B(2\theta, 0) P\big)^{N_b}}
$$

$$
\Rightarrow Z=<\Phi_T|e^{-2\theta H}|\Phi_T>=\int \mathcal D\phi\, P_{\text{aux}}(\phi) [P^\dagger \mathcal B_\phi(2\theta, 0) P]^{2N_b}
$$

## 物理量计算

$$
\begin{aligned}
  <\hat{O}> &= <\Phi_T|e^{-\theta H}\hat{O}e^{-\theta H}|\Phi_T>\\
&=\int  \mathcal D\phi  P_{\text{aux}}(\phi) <\Phi_T|U_\phi(2\theta,\theta)\hat{O}U_\phi(\theta,0)|\Phi_T>\\
  &=\int  \mathcal D\phi \textcolor{red}{ P_{\text{aux}}(\phi) <\Phi_T|U_\phi(2\theta,0)|\Phi_T>]}\cdot 
  		\textcolor{blue}{\frac{<\Phi_T|U_\phi(2\theta,\theta)\hat{O}U_\phi(\theta,0)|\Phi_T>}{ <\Phi_T|U_\phi(2\theta,0)|\Phi_T>}}

  
  \end{aligned}
$$

等时 Green 函数:

$$
G_{ij}(\tau) = \frac{\langle \Phi_T | e^{-(2\theta-\tau)\hat H} \, \hat b_i \hat b_j^\dagger \, e^{-\tau\hat H} | \Phi_T \rangle}{\langle \Phi_T | e^{-2\theta\hat H} | \Phi_T \rangle}
\\
=\delta_{ij}+N_b \cdot \frac{[\mathcal B(\theta, 0) P]_i^* \cdot [\mathcal B(2\theta, \theta) P]_j}{P^\dagger \mathcal B(2\theta, 0) P}
$$

$$
\text{Define:} 
\begin{cases}
  P_L^\dagger = [\mathcal B(\theta, 0)P]^\dagger \\
  P_R=\mathcal B(\theta, 0) P
  \end{cases}
\qquad
G=I+N_b \frac{P_R^\dagger \otimes P_L}{P_L\cdot P_R}
$$

KEEP：$|P_L| =  |P_R|=1$  保持数值稳定

**没有wick定理！！！**

$$
<a_{i_1}^\dagger  \cdots a_{in}^\dagger a_{j_1}  \cdots a_{jn}>=N_b(N_b-1)\cdots(N_b-n+1)\frac{  (P_L^\dagger)_{i1}\cdots (P_L^\dagger)_{in} (P_R)_{jn} \cdots (P_R)_{j1} }  {(P_L^\dagger P_R)^n}
$$

# Boson-EE

从二阶 Rényi 熵的定义开始，有限投影时间下的投影态密度矩阵是

$$
\hat\rho(\theta) =
\frac{
e^{-\theta \hat H}
|\Phi_T\rangle
\langle \Phi_T|
e^{-\theta \hat H}
}{
\langle \Phi_T|e^{-2\theta \hat H}|\Phi_T\rangle
}.
$$

对空间区域 $A$，约化密度矩阵为

$$
\hat{\rho}_A(\theta) =
\mathrm{Tr}_{\bar A}\hat\rho(\theta)
$$

二阶 Rényi 熵定义为

$$
S_2(A;\theta) =
-\ln \mathrm{Tr}_A\left[\hat\rho_A(\theta)^2\right].
$$

所以真正需要在 QMC 中计算的是

$$
e^{-S_2(A;\theta)} =
\mathrm{Tr}_A\left[\hat\rho_A(\theta)^2\right].
$$

$$
|R_\alpha^b\rangle =
\frac{1}{\sqrt{N_b!}}
\left(
\sum_i
( R^b_\alpha)_i
\hat b_i^\dagger
\right)^{N_b}
|0\rangle
\\
\langle L^b_\alpha| =
\frac{1}{\sqrt{N_b!}}
\langle 0|
\left(
\sum_i
( L^b_\alpha)_i
\hat b_i
\right)^{N_b}.
$$

1. 先对单个 flavor 做 $\bar A$ trace

   把右态中的单粒子轨道拆成 $A$ 和 $\bar A$ 两部分：

   $$
   \sum_i
   ( R^b_\alpha)_i \hat b_i^\dagger =
   \sum_{i\in A}
   ( R^b_\alpha)_i
   \hat b_i^\dagger
   +
   \sum_{i\in \bar A}
   ( R^b_\alpha)_i
   \hat b_i^\dagger .
   $$

   于是

   $$
   |R_\alpha^b\rangle =
   \sum_{k=0}^{N_b}
   \sqrt{\binom{N_b}{k}}
   |R_{\alpha,A}^b;k\rangle
   |R_{\alpha,\bar A}^b;N_b-k\rangle,
   $$

   其中

   $$
   |R_{\alpha,A}^b;k\rangle =
   \frac{1}{\sqrt{k!}}
   \left(
   \sum_{i\in A}
   ( R_\alpha)_i
   \hat b_i^\dagger
   \right)^k
   |0_A\rangle,
   $$

   $$
   |R_{\alpha,\bar A}^b;N_b-k\rangle =
   \frac{1}{\sqrt{(N_b-k)!}}
   \left(
   \sum_{i\in \bar A}
   ( R_\alpha)
   \hat b_i^\dagger
   \right)^{N_b-k}
   |0_{\bar A}\rangle.
   $$

   左态同理：

   $$
   \langle L_\alpha^b| =
   \sum_{k=0}^{N_b}
   \sqrt{\binom{N_b}{k}}
   \langle L_{\alpha,A}^b;k|
   \langle L_{\alpha,\bar A}^b;N_b-k|.
   $$

   现在对 $\bar A$ trace。由于 $\bar A$ 中粒子数不同的 sectors 正交，只有相同的 (N_b-k) sector 存活。于是

   $$
   \rho_A=\mathrm{Tr}_{\bar A}
   \frac{
   |R_\alpha^b\rangle
   \langle L_\alpha^b|
   }{
   \left(
    L_\alpha^\dagger  R_\alpha
   \right)^{N_b}
   }=
   \sum_{k=0}^{N_b}
   \binom{N_b}{k}
   \frac{
   \left(
    L_{\alpha,\bar A}^\dagger
    R_{\alpha,\bar A}
   \right)^{N_b-k}
   }{
   \left(
    L_\alpha^\dagger
    R_\alpha
   \right)^{N_b}
   }
   |R_{\alpha,A}^b;k\rangle
   \langle L_{\alpha,A}^b;k|.
   $$

   > 证明：1、不同粒子的态正交	2、k粒子下的完备基：$\sum_e \ket{e,k}\bra{e,k}=I$
   >
   > $\mathrm{Tr}_{\bar A} [\ket{R_{\bar A},k}\bra{L_{\bar A},k}]=\sum_e \bra{e,k}  \ket{R_{\bar A},k}\bra{L_{\bar A},k} \ket{e,k}=\sum_e \bra{L_{\bar A},k} \ket{e,k} \bra{e,k}  \ket{R_{\bar A},k} = \bra{L_{\bar A},k}\ket{R_{\bar A},k}$
   >
2. 对两个Replica的乘机在求迹。现在取 replica 1 和 replica 2 的单 flavor 约化算符相乘。由于区粒子数守恒，两个算符中的粒子数也必须相同，所以最后仍然只剩一个求和指标。对固定的 $k$，有

   $$
   \mathrm{Tr}_A 
   \left[
   |R_{1,A}^b;k\rangle
   \langle L_{1,A}^b;k|
   |R_{2,A}^b;k\rangle
   \langle L_{2,A}^b;k|
   \right]=
   ({L_{1,A}^\dagger}{R_{2,A}})^k
   
   (
    L_{2,A}^\dagger
    R_{1,A}
   )^k .
   $$
3. $\rho_A = \rho_A^c \otimes \rho_A^b$ ，避免符号问题：$\rho_A^c = (\rho_A^b) ^\star$

$$
e^{-S_2^A}=\mathrm{Tr}_A [ \rho_{A,1}\rho_{A,2}]
=\left\langle
\left|
\sum_{k=0}^{N_b}
\binom{N_b}{k}^2 
\frac{
\left(
 L_{1,\bar A}^\dagger
 R_{1,\bar A}
\right)^{N_b-k}
\left(
 L_{2,\bar A}^\dagger
 R_{2,\bar A}
\right)^{N_b-k}
\left(
 L_{1,A}^\dagger
 R_{2,A}
\right)^k
\left(
 L_{2,A}^\dagger
 R_{1,A}
\right)^k
}{
\left(
 L_1^\dagger
 R_1
\right)^{N_b}
\left(
 L_2^\dagger
 R_2
\right)^{N_b}
}
\right|^2
\right\rangle
$$

# 更新

$$
\begin{aligned}
&\bra{L(\tau)}=\bra{P}B(2N_\tau,\tau)
\\
&\ket{R(\tau)}=B(\tau,0)\ket{P}
\end{aligned}
$$

Update  $\tau$：$B(\tau)^\prime=(I+\Delta)B(\tau),\Delta \equiv e^{V^\prime-V}-I$

$$
\begin{aligned}
&\bra{L^\prime(\tau)}=\bra{L(\tau)}
\\
&\ket{R^\prime(\tau)}=(I+\Delta)B(\tau,0)\ket{P}=\ket{R(\tau)}+\ket{(0\dots \Delta R(\tau)_i \dots0 )}
\end{aligned}
$$

概率

$$
Ratio=\frac{\bra{L^\prime(\tau)} \ket{R^\prime(\tau)}}{\bra{L^(\tau)} \ket{R(\tau)}}= 1+ \frac{\bra{L(\tau)}\Delta\ket{R(\tau)}}{\bra{L(\tau)\ket{R(\tau)}}}=1+\frac{(L_i \Delta_{ii}R_i)^{N_b}}{(\vec{L}\cdot \vec{R})^{N_b}}
$$






As for Entanglement Entropy:
$$
\begin{aligned}
&\bra{L(\theta)}=\bra{P}B(2\theta,\theta)
\\
&\ket{R(\theta)}=B(\theta,0)\ket{P}
\end{aligned}
$$

1. $\tau \leq \theta$ ，$B(\theta,\tau)$ 不用更新

   $$
   \begin{aligned}
   \bra{L^\prime(\theta)}&=\bra{L(\theta)}
   \\
   \ket{R^\prime(\theta)}&=B(\theta,\tau)(I+\Delta)B(\tau,0)\ket{P}
   \\
   &=\ket{R(\theta)}+\Delta_{ii} B(\theta,\tau)[:,i]\otimes B(\tau,0)[i,:]\ket{P}
   \\
   &=\ket{R(\theta)}+\Delta_{ii} B(\theta,\tau)[:,i]\cdot \ket{R(\tau)}_i
   \end{aligned}
   $$
2. $\tau>\theta$，$B^\prime(\tau,\theta)=(I+\Delta)B(\tau,\theta)$

$$
\begin{aligned}
\ket{R^\prime(\theta)}&=\ket{R(\theta)}
\\
\bra{L^\prime(\theta)}&=\bra{P}B(2\theta,\tau)(I+\Delta)B(\tau,\theta)
\\
&=\bra{L(\theta)}+\Delta_{ii} \bra{L(\tau)}_i\cdot B(\tau,\theta)[i,:]
\end{aligned}
$$

<pre class="vditor-reset" placeholder="" contenteditable="true" spellcheck="false"><hr data-block="0"/></pre>

# 自由玻色子约化密度矩阵（有限温也许有用）

自由玻色子 Gaussian 态的最一般形式：

$$
\rho = C \; e^{-\hat{\mathbf b}^\dagger h \hat{\mathbf b}}, \qquad h \text{ 为正定 Hermite 矩阵（保证可归一化）}
\tag{4}
$$

**关键**：玻色子没有 Pauli 不相容原理，$h$ 必须正定才能保证 $\rho$ 迹类（$\text{Tr}(\rho) < \infty$）。

$$
F_{ij} = \text{Tr}\big[\rho\, \hat b_i^\dagger \hat b_j\big]=G_{ji}-\delta_{ij}
$$

$$
\boxed{F = (I - e^{-h})^{-1}}
$$

要求 $\text{Tr}(\rho) = 1$：

$$
1 = C \; \frac{1}{\det(I - e^{-h})} \quad\Longrightarrow\quad \boxed{C = \det(I - e^{-h})=\frac{1}{\det F}}
$$

$$
\boxed{\text{Tr}\big[e^{-\hat{\mathbf b}^\dagger h_1 \hat{\mathbf b}} e^{-\hat{\mathbf b}^\dagger h_2 \hat{\mathbf b}}\big] = \frac{1}{\det(I - e^{-h_1} e^{-h_2})}}
$$

$$
e^{-S_2}=\sum_{s,s'} P_s P_{s'} \text{Tr}  [ {\rho_{A,s} \rho_{A,s'}} ]=\sum_{s,s'} P_s P_{s'} \frac{1}{\det(F_{s,A} + F_{s',A} - I_{N_A})}
$$
