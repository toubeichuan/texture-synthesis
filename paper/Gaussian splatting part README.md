# Gaussian Splatting Part 说明与翻译

## 大纲

1. **从合成纹理到网格引导的 Gaussian Splatting**
   - 承接 image transfer / texture synthesis 阶段的输出。
   - 输入为带纹理网格 $\hat{\mathcal{M}}=(\mathcal{V},\mathcal{F},\mathcal{T})$。
   - 目标是学习 mesh-guided Gaussian representation $\mathcal{G}$，用于新视角渲染和可编辑渲染。

2. **多视角监督数据构建**
   - 使用 `transforms_train.json` 和 `transforms_test.json` 中的相机位姿与内参。
   - 从 textured mesh 渲染监督图像，或直接使用前一阶段生成的 multi-view refined / inpainted images。
   - 训练相机用于优化，测试相机用于 held-out 或修改后的 novel-view rendering。

3. **绑定到网格的高斯参数化**
   - 每个 mesh face 初始化 $K$ 个 Gaussian splats。
   - 高斯中心使用 barycentric coordinates 表示，使其保持在 mesh surface 上。
   - 高斯参数包含 center、covariance、opacity、color / spherical harmonics coefficients。
   - covariance 由 face normal 和 tangent directions 构造，使高斯继承 mesh geometry。

4. **可微 Gaussian 渲染**
   - 将高斯投影到相机图像平面。
   - 使用 screen-space covariance 和 alpha compositing 得到渲染图像 $\hat{I}_i$。
   - 渲染过程依赖 camera intrinsics 与 extrinsics，并对高斯参数可微。

5. **优化目标**
   - 使用 $\hat{I}_i$ 与监督图像 $I_i$ 之间的 L1 + SSIM loss。
   - 优化 barycentric coordinates、opacity、scale、SH color，并可选优化 mesh vertices。
   - 与 vanilla 3DGS 的区别在于高斯受 mesh topology 约束，而不是自由分布在空间中。

6. **新视角渲染与评估**
   - 训练后输入任意 query camera $c^*$ 即可渲染 novel-view image。
   - 如果 test camera 被修改，PSNR / SSIM / LPIPS 不再具有严格 full-reference 意义。
   - 可使用 silhouette coverage、hole ratio、leakage ratio、sharpness 等 geometry-aware metrics。

## 中文翻译

### 从合成纹理到网格引导的高斯溅射

前一阶段的 reference-guided texture synthesis 通过将多视角精修图像反投影回 UV 域，得到带纹理的网格。我们将该结果表示为

```tex
\hat{\mathcal{M}} = (\mathcal{V}, \mathcal{F}, \mathcal{T}),
```

其中 $\mathcal{V}=\{v_j\}_{j=1}^{|\mathcal{V}|}$ 表示顶点集合，$\mathcal{F}$ 表示三角面片集合，$\mathcal{T}$ 表示合成得到的 UV 纹理图。基于 $\hat{\mathcal{M}}$，我们进一步学习一个网格引导的高斯表示 $\mathcal{G}$，用于 novel-view rendering 和可编辑渲染。与将 Gaussian Splatting 视为无约束点云式表示不同，我们将 Gaussian splats 绑定到网格面片上，使学习到的辐射场始终与合成纹理所在的表面保持对齐。

### 多视角监督数据构建

令 $\mathcal{C}=\{c_i\}_{i=1}^{N}$ 表示用于高斯优化的相机集合。每个相机记为

```tex
c_i=(K_i,E_i),
```

其中 $K_i$ 为相机内参矩阵，$E_i=[R_i|t_i]$ 为 world-to-camera 外参变换。相机位姿和内参从 NeRF 风格的 `transforms_train.json` 与 `transforms_test.json` 中读取。每个 frame 保存 camera-to-world 位姿和对应图像路径，水平视场角会被转换为 Gaussian rasterization 所使用的焦距。

对于每个监督相机 $c_i$，我们构造目标 RGB 图像 $I_i \in \mathbb{R}^{H\times W\times 3}$。当最终 UV 纹理可用时，目标视角由带纹理网格渲染得到：

```tex
I_i=\mathbb{R}_{\mathrm{mesh}}(\hat{\mathcal{M}},c_i).
```

等价地，如果前一阶段保存了逐视角的合成结果，也可以直接将对应的 refined 或 inpainted image 作为 $I_i$。由此得到高斯训练数据集

```tex
\mathcal{D}_{\mathrm{GS}}=\{(I_i,c_i)\}_{i=1}^{N}.
```

训练相机和测试相机保持分离。`transforms_train.json` 中的相机用于定义光度监督，`transforms_test.json` 则用于 held-out rendering 或修改后的 camera-view testing。因此，当测试位姿被修改时，渲染结果反映的是新的相机坐标，而不是强制复现原始合成图像。

### 绑定到网格的高斯参数化

我们将高斯表示定义为

```tex
\mathcal{G}=\{g_{f,k}\mid f\in\mathcal{F},\;k=1,\ldots,K\},
```

其中每个网格面片上初始化 $K$ 个 Gaussian splats。对于三角面片 $f=(v_{f,1},v_{f,2},v_{f,3})$，第 $k$ 个高斯的中心用 barycentric coordinates 表示：

```tex
x_{f,k}
=
\alpha_{f,k,1}v_{f,1}
+\alpha_{f,k,2}v_{f,2}
+\alpha_{f,k,3}v_{f,3},
```

并满足 simplex constraint：

```tex
\alpha_{f,k,r}\ge 0,\qquad
\sum_{r=1}^{3}\alpha_{f,k,r}=1.
```

实际优化中，barycentric weights 由无约束可学习参数归一化得到。该约束保证高斯中心位于网格表面上，并避免表示漂移到无关的自由空间中。

每个高斯包含中心、透明度、各向异性尺度、旋转以及颜色系数：

```tex
g_{f,k}=
\left(
x_{f,k},
\Sigma_{f,k},
o_{f,k},
\mathbf{h}_{f,k}
\right),
```

其中 $o_{f,k}\in[0,1]$ 表示 opacity，$\mathbf{h}_{f,k}$ 表示 RGB color 或 spherical harmonics coefficients。对于 view-dependent color，沿视线方向 $d$ 的辐射颜色写为

```tex
\mathbf{c}_{f,k}(d)
=
\sum_{\ell=0}^{L}\sum_{m=-\ell}^{\ell}
\mathbf{h}_{f,k}^{\ell m}Y_{\ell m}(d),
```

其中 $Y_{\ell m}$ 为 spherical harmonics basis functions。

协方差同样由面片几何约束。令面片法向为

```tex
n_f=
\frac{(v_{f,2}-v_{f,1})\times(v_{f,3}-v_{f,1})}
{\|(v_{f,2}-v_{f,1})\times(v_{f,3}-v_{f,1})\|_2}.
```

我们通过对面片边向量进行正交化得到两个切向方向 $t_{f,1}$ 和 $t_{f,2}$，并构造局部坐标系

```tex
R_f=[n_f,\;t_{f,1},\;t_{f,2}].
```

协方差参数化为

```tex
\Sigma_{f,k}
=
R_fS_{f,k}S_{f,k}^{T}R_f^{T},
```

其中

```tex
S_{f,k}
=
\mathrm{diag}(\epsilon,\rho_{f,k}s_{f,1},\rho_{f,k}s_{f,2}).
```

这里 $\epsilon$ 是法向方向上的小固定尺度，$s_{f,1}$ 和 $s_{f,2}$ 是与面片大小相关的切向尺度，$\rho_{f,k}>0$ 是可学习的尺度乘子。因此，高斯在表面法向上近似为平坦分布，而在面片切向上呈各向异性。由于 $x_{f,k}$ 和 $\Sigma_{f,k}$ 都是网格顶点的函数，当网格被编辑时，高斯中心、方向和尺度会随之自动更新。

### 可微高斯渲染

给定相机 $c_i=(K_i,E_i)$，所有高斯都会被投影到图像平面。对于第 $j$ 个高斯，其中 $j$ 索引某个 $(f,k)$ 对，其投影均值为

```tex
\mu_{i,j}
=
\pi\!\left(K_iE_i x_j\right),
```

其中 $\pi(\cdot)$ 表示 perspective division。3D 协方差通过相机投影的局部 Jacobian $J_{i,j}$ 映射到屏幕空间：

```tex
\Sigma'_{i,j}
=
J_{i,j}R_i\Sigma_jR_i^{T}J_{i,j}^{T}.
```

对于像素 $p$，投影高斯的 opacity-weighted contribution 为

```tex
a_{i,j}(p)
=
o_j
\exp\!\left(
-\frac{1}{2}(p-\mu_{i,j})^T
(\Sigma'_{i,j})^{-1}
(p-\mu_{i,j})
\right).
```

经过深度排序后，可微 alpha compositing 得到渲染图像：

```tex
\hat{I}_i(p)
=
\sum_{j\in\mathcal{N}_i(p)}
T_{i,j}(p)\,
a_{i,j}(p)\,
\mathbf{c}_j(d_{i,j}),
```

其中 $\mathcal{N}_i(p)$ 为覆盖像素 $p$ 的 splat 集合，$d_{i,j}$ 为视线方向，且

```tex
T_{i,j}(p)
=
\prod_{\ell<j}
\left(1-a_{i,\ell}(p)\right)
```

表示第 $j$ 个高斯之前累积的 transmittance。该渲染器对 opacity、spherical harmonics color、切向尺度、barycentric coordinates，以及可选的 mesh vertex positions 都是可微的。

### 优化目标

网格引导的高斯表示通过比较渲染图像 $\hat{I}_i$ 与目标图像 $I_i$ 进行优化。对于一组训练相机 mini-batch $\mathcal{B}$，我们采用如下损失：

```tex
\mathcal{L}_{\mathrm{photo}}
=
\frac{1}{|\mathcal{B}|}
\sum_{i\in\mathcal{B}}
\left[
(1-\lambda)\mathcal{L}_{1}(\hat{I}_i,I_i)
+
\lambda\left(1-\mathrm{SSIM}(\hat{I}_i,I_i)\right)
\right],
```

其中 $\lambda$ 控制结构相似性项的相对权重。需要优化的参数为

```tex
\Theta=
\{\alpha_{f,k},o_{f,k},\rho_{f,k},\mathbf{h}_{f,k}\}_{f,k},
```

当需要进行小幅几何校正时，也可以将顶点集合 $\mathcal{V}$ 纳入优化。在我们的设定中，带纹理网格已经提供了可靠的几何先验，因此固定拓扑结构有助于稳定训练，并保持高斯表示与合成纹理对齐。

这与 vanilla 3D Gaussian Splatting 不同。普通 3DGS 可写为

```tex
\mathcal{G}_{\mathrm{free}}
=
\{(\mu_j,\Sigma_j,o_j,\mathbf{h}_j)\}_{j=1}^{M},
\qquad
\mu_j\in\mathbb{R}^{3},
```

其中高斯中心是空间中的无约束点。通过使用 mesh topology 约束高斯中心和协方差，我们的方法能更好地保留合成网格的几何结构，并更适合后续 mesh-level editing。

### 新视角渲染与评估

训练完成后，给定查询相机

```tex
c^{*}=(K^{*},E^{*}),
```

即可通过相同的 Gaussian rasterizer 渲染新视角：

```tex
\hat{I}^{*}
=
\mathbb{R}_{\mathrm{GS}}(\mathcal{G},c^{*}).
```

输出图像仅依赖学习到的高斯表示和输入相机参数。因此，如果测试相机文件被修改，渲染结果会跟随新的相机坐标，而不依赖旧的 ground-truth images。

当测试相机与目标图像相匹配时，可以在 $\hat{I}_i$ 和 $I_i$ 之间计算 PSNR、SSIM 和 LPIPS 等 full-reference metrics。若测试相机被修改为 novel poses，由于目标图像和渲染图像对应不同射线，这些指标不再具有严格的一一对应意义。因此，我们进一步使用 geometry-aware criteria。令 $S_i^{\mathrm{mesh}}$ 为相机 $c_i$ 下的网格 silhouette，令 $S_i^{\mathrm{GS}}$ 为由高斯累积 opacity 得到的前景 mask。我们定义

```tex
\mathrm{Coverage}
=
\frac{|S_i^{\mathrm{GS}}\cap S_i^{\mathrm{mesh}}|}
{|S_i^{\mathrm{mesh}}|+\epsilon},
```

```tex
\mathrm{Hole}
=
\frac{|S_i^{\mathrm{mesh}}\setminus S_i^{\mathrm{GS}}|}
{|S_i^{\mathrm{mesh}}|+\epsilon},
\qquad
\mathrm{Leakage}
=
\frac{|S_i^{\mathrm{GS}}\setminus S_i^{\mathrm{mesh}}|}
{|S_i^{\mathrm{GS}}|+\epsilon}.
```

Foreground coverage、hole ratio、leakage ratio 以及 sharpness 能更直接地反映网格引导的高斯表示在新视角下是否保持完整、紧凑和稳定。
