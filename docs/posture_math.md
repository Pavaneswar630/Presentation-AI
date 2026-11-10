# Mathematical Formulation: Posture Stability & Movement Analytics

## 1. Overview

The **Presentation AI Post-Session Analytics Engine** converts discrete per-frame 3D landmark points into continuous kinematic and geometric time-series. These time-series are evaluated using **PyTorch/NumPy** tensor operations to produce calibrated 0–100 scores for physical presence and speech delivery.

---

## 2. Geometric Trigonometry Formulations

### 2.1. Shoulder Asymmetry Angle ($\theta_{shoulder}$)
Let $\vec{P}_{L\_shoulder}(t) = (x_{11}, y_{11})$ and $\vec{P}_{R\_shoulder}(t) = (x_{12}, y_{12})$ denote the normalized 2D image coordinates of the left and right shoulders at time $t$.

$$\Delta x(t) = x_{12}(t) - x_{11}(t)$$
$$\Delta y(t) = y_{12}(t) - y_{11}(t)$$

The instantaneous shoulder tilt angle relative to the horizontal plane is:
$$\theta_{shoulder}(t) = \left| \arctan2(\Delta y(t), \Delta x(t)) \right| \times \frac{180^\circ}{\pi}$$

* **Normal Baseline:** $\theta_{shoulder} \in [0.0^\circ, 3.5^\circ]$.
* **Defect Alert Threshold:** $\theta_{shoulder} > 5.0^\circ$ triggered for duration $\ge 3.0\text{s}$.

---

### 2.2. Spinal Plumb-Line Deviation ($\phi_{spine}$)
We compute the midpoints of the shoulder girdle and pelvic hip axis:
$$\vec{P}_{mid\_shoulder}(t) = \frac{\vec{P}_{11}(t) + \vec{P}_{12}(t)}{2}$$
$$\vec{P}_{mid\_hip}(t) = \frac{\vec{P}_{23}(t) + \vec{P}_{24}(t)}{2}$$

The upward spine vector is defined as:
$$\vec{V}_{spine}(t) = \vec{P}_{mid\_shoulder}(t) - \vec{P}_{mid\_hip}(t) = \begin{pmatrix} v_x(t) \\ v_y(t) \end{pmatrix}$$

Because screen $y$-coordinates increase downward, the true vertical upward vector is $\hat{u}_{vertical} = \begin{pmatrix} 0 \\ -1 \end{pmatrix}$. The angular deflection from vertical is:

$$\cos(\phi_{spine}(t)) = \frac{\vec{V}_{spine}(t) \cdot \hat{u}_{vertical}}{\|\vec{V}_{spine}(t)\|_2 \|\hat{u}_{vertical}\|_2} = \frac{-v_y(t)}{\sqrt{v_x^2(t) + v_y^2(t)}}$$

$$\phi_{spine}(t) = \arccos\left(\mathrm{clip}\left(\frac{-v_y(t)}{\|\vec{V}_{spine}(t)\|_2}, -1.0, 1.0\right)\right) \times \frac{180^\circ}{\pi}$$

* **Normal Upright Stance:** $\phi_{spine} \in [0.0^\circ, 6.0^\circ]$.
* **Slump Warning Threshold:** $\phi_{spine} > 12.0^\circ$.

---

### 2.3. Hand Kinetic Jitter & Fidgeting Velocity ($v_{hand}$)
Given consecutive wrist landmarks $\vec{P}_{wrist}(t)$ and $\vec{P}_{wrist}(t - \Delta t)$:
$$v_{wrist}(t) = \frac{\|\vec{P}_{wrist}(t) - \vec{P}_{wrist}(t - \Delta t)\|_2}{\Delta t}$$
$$v_{hand}(t) = \frac{v_{L\_wrist}(t) + v_{R\_wrist}(t)}{2}$$

---

## 3. Vectorized Posture Stability Score ($S_{posture}$)

To evaluate overall stage presence, we construct temporal tensors $\mathbf{\Theta} \in \mathbb{R}^N$, $\mathbf{\Phi} \in \mathbb{R}^N$, and $\mathbf{V}_{hand} \in \mathbb{R}^N$ across $N$ session frames:

$$\sigma^2_\theta = \mathrm{Var}(\mathbf{\Theta}) = \frac{1}{N}\sum_{i=1}^N (\theta_i - \bar{\theta})^2$$
$$\sigma^2_\phi = \mathrm{Var}(\mathbf{\Phi}) = \frac{1}{N}\sum_{i=1}^N (\phi_i - \bar{\phi})^2$$
$$\bar{v}_{hand} = \frac{1}{N}\sum_{i=1}^N v_{hand, i}$$

The global stability score $S_{posture} \in [0, 100]$ uses an exponential penalty function:

$$S_{posture} = 100 \cdot \exp\left( - \left[ \alpha \sigma^2_\theta + \beta \sigma^2_\phi + \gamma \bar{v}_{hand} \right] \right)$$

### Calibrated Weight Parameters:
* $\alpha = 0.03$: Sensitivity to lateral shoulder swaying variance.
* $\beta = 0.05$: Sensitivity to forward/backward slumping variance.
* $\gamma = 3.00$: Sensitivity to rapid hand fidgeting.

---

## 4. Speaking Cadence & Filler Frequency

1. **Cadence (Words Per Minute):**
   $$\mathrm{WPM} = \frac{N_{words}}{\Delta t_{minutes}}$$
2. **Pacing Score ($S_{pacing} \in [0, 100]$):**
   $$S_{pacing} = \max\left(0.0, \, \min\left(100.0, \, 100.0 - 1.5 \cdot |\mathrm{WPM} - 145|\right)\right)$$
3. **Filler Word Ratio:**
   $$\text{Filler Ratio (\%)} = \frac{N_{fillers}}{N_{words}} \times 100\%$$
   $$\text{Verbal Clarity Score} = \max(0.0, 100.0 - 10.0 \times \text{Filler Ratio})$$
