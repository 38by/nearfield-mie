import numpy as np
from mpmath import besselj, sqrt, pi, mp, mpc, acos, atan2, re, cos, sin, exp, diff

import matplotlib.pyplot as plt
from tqdm import tqdm  # 進捗バー表示
from scipy.special import lpmv

# === 定数設定 ===
λ = 0.6943e-6        # 波長 [m]
d = 20e-6            # 粒子径 [m]
m = mpc(1.5, 0)   # 粒子の複素屈折率
k = 2 * np.pi / λ    # 波数 [1/m]
α = np.pi * d / λ    # サイズパラメータ
β = m * α            # 複素サイズパラメータ
N = round(α + 4 * α**(1/3))  # 最大次数
E0 = 1.0             # 電場振幅 [V/m]
ϵ = 8.854e-12        # 真空の誘電率
μ = 4 * np.pi * 1e-7 # 真空の透磁率

# =======================
# Eq. 10: ψ_n(x)
# =======================
def ψ(n, x):
    return sqrt(pi * x / 2) * besselj(n + 0.5, x)

# =======================
# ψ_n'(x)（Eq. 10 の導関数）
# =======================
def ψdiff(n, x):
    return sqrt(pi / (2 * x)) * ((n + 1) * besselj(n + 0.5, x) - x * besselj(n + 1.5, x))

# =======================
# ψ_n''(x)
# =======================
def ψdiff2(n, x):
    x = mp.mpf(x)
    term1 = x**2 * besselj(n - 1.5, x)
    term2 = -2 * x**2 * besselj(n + 0.5, x)
    term3 = x**2 * besselj(n + 2.5, x)
    term4 = 2 * x * besselj(n - 0.5, x)
    term5 = -2 * x * besselj(n + 1.5, x)
    term6 = -besselj(n + 0.5, x)
    return (1 / (4 * x**(3/2))) * sqrt(pi / 2) * (term1 + term2 + term3 + term4 + term5 + term6)

# =======================
# Eq. 11: χ_n(x)
# =======================
def χ(n, x):
    return (-1)**n * sqrt(pi * x / 2) * besselj(-n - 0.5, x)

# =======================
# χ_n'(x)
# =======================
def χdiff(n, x):
    x = mp.mpf(x)
    term1 = x * besselj(-n - 1.5, x)
    term2 = besselj(-n - 0.5, x)
    term3 = -x * besselj(-n + 0.5, x)
    return 0.5 * (-1)**n * sqrt(pi / (2 * x)) * (term1 + term2 + term3)

# =======================
# χ_n''(x)
# =======================
def χdiff2(n, x):
    x = mp.mpf(x)
    term1 = x**2 * besselj(-n - 2.5, x)
    term2 = -2 * x**2 * besselj(-n - 0.5, x)
    term3 = x**2 * besselj(-n + 1.5, x)
    term4 = 2 * x * besselj(-n - 1.5, x)
    term5 = -2 * x
    return (1 / (4 * x**(3/2))) * sqrt(pi / 2) * (-1)**n * (term1 + term2 + term3 + term4 + term5)

# =======================
# Eq. 9: ξ_n(x)
# =======================
def ξ(n, x):
    return ψ(n, x) + 1j * χ(n, x)

def ξdiff(n, x):
    return ψdiff(n, x) + 1j * χdiff(n, x)

def ξdiff2(n, x):
    return ψdiff2(n, x) + 1j * χdiff2(n, x)

# =======================
# Eq. 12: a_n
# =======================
def a(n, α, β, m):
    α = mp.mpf(α)
    β = mp.mpc(β)
    num = ψ(n, α) * ψdiff(n, β) - m * ψdiff(n, α) * ψ(n, β)
    den = ξ(n, α) * ψdiff(n, β) - m * ξdiff(n, α) * ψ(n, β)
    return num / den

# =======================
# Eq. 13: b_n
# =======================
def b(n, α, β, m):
    α = mp.mpf(α)
    β = mp.mpc(β)
    num = m * ψ(n, α) * ψdiff(n, β) - ψdiff(n, α) * ψ(n, β)
    den = m * ξ(n, α) * ψdiff(n, β) - ξdiff(n, α) * ψ(n, β)
    return num / den

# ===== Legendre P_l^m と導関数（Eq. 8）=====
def Plm(x, l, m=1):
    return lpmv(m, l, float(x))  # m=1 固定でOK、xはfloatに変換

def dnPl(x, l, m=1):
    """P_l^m(x) の x による微分"""
    x_mp = mp.mpf(x)
    return float(diff(lambda t: lpmv(m, l, float(t)), x_mp))

# ===== Eq. 20: π_n(θ) =====
def pifunc(n, θ):
    cθ = float(mp.cos(θ))
    sθ = float(mp.sin(θ))
    if abs(sθ) < 1e-12:
        return 0.5 * n * (n + 1)
    else:
        return Plm(cθ, n, 1) / sθ

# ===== Eq. 21: τ_n(θ) =====
def τ(n, θ):
    cθ = float(mp.cos(θ))
    sθ = float(mp.sin(θ))
    if abs(sθ) < 1e-12:
        return 0.0
    else:
        return -sθ * dnPl(cθ, n, 1)

# ===== Eq. 22: dξ/dr =====
def dξdr(n, r):
    return k * ξdiff(n, k * r)

# ===== Eq. 23: d²ξ/dr² =====
def d2ξdr2(n, r):
    return k**2 * ξdiff2(n, k * r)

# ===== Eq. 33: Etr (r, θ, φ) =====
def Etr(r, θ, ϕ):
    sum_term = mp.mpc(0)
    for n in range(1, N+1):
        coeff = (1j)**(n + 1) * (-1)**n * (2*n + 1) / (n*(n + 1))
        term = coeff * a(n, α, β, m) * (ξdiff2(n, k*r) + ξ(n, k*r)) * Plm(cos(θ), n, 1)
        sum_term += term
    return E0 * cos(ϕ) * (sin(θ) * exp(-1j * k * r * cos(θ)) + sum_term)

# ===== Eq. 34: Etθ (r, θ, φ) =====
def Etθ(r, θ, ϕ):
    sum_term = mp.mpc(0)
    for n in range(1, N+1):
        coeff = (1j)**(n + 1) * (-1)**n * (2*n + 1) / (n*(n + 1))
        term = coeff * (
            a(n, α, β, m) * ξdiff(n, k*r) * τ(n, θ) -
            1j * b(n, α, β, m) * ξ(n, k*r) * pifunc(n, θ)
        )
        sum_term += term
    return E0 * cos(ϕ) / (k * r) * (k*r * cos(θ) * exp(-1j * k * r * cos(θ)) + sum_term)

# ===== Eq. 35: Etφ (r, θ, φ) =====
def Etϕ(r, θ, ϕ):
    sum_term = mp.mpc(0)
    for n in range(1, N+1):
        coeff = (1j)**(n + 1) * (-1)**n * (2*n + 1) / (n*(n + 1))
        term = coeff * (
            a(n, α, β, m) * ξdiff(n, k*r) * pifunc(n, θ) -
            1j * b(n, α, β, m) * ξ(n, k*r) * τ(n, θ)
        )
        sum_term += term
    return -E0 * sin(ϕ) / (k * r) * (k * r * exp(-1j * k * r * cos(θ)) + sum_term)

# ===== Eq. 36: Htr(r, θ, φ) =====
def Htr(r, θ, ϕ):
    sum_term = mp.mpc(0)
    for n in range(1, N+1):
        coeff = (1j)**(n+1) * (-1)**n * (2*n+1) / (n*(n+1))
        term = coeff * b(n, α, β, m) * (ξdiff2(n, k*r) + ξ(n, k*r)) * Plm(cos(θ), n, 1)
        sum_term += term
    return E0 * sqrt(ϵ/μ) * sin(ϕ) * (sin(θ) * exp(-1j * k * r * cos(θ)) + sum_term)

# ===== Eq. 37: Htθ(r, θ, φ) =====
def Htθ(r, θ, ϕ):
    sum_term = mp.mpc(0)
    for n in range(1, N+1):
        coeff = (1j)**(n+1) * (-1)**n * (2*n+1) / (n*(n+1))
        term = coeff * (
            -1j * a(n, α, β, m) * ξ(n, k*r) * pifunc(n, θ) +
            b(n, α, β, m) * ξdiff(n, k*r) * τ(n, θ)
        )
        sum_term += term
    return E0 / (k * r) * sqrt(ϵ / μ) * sin(ϕ) * (
        k * r * cos(θ) * exp(-1j * k * r * cos(θ)) + sum_term
    )

# ===== Eq. 38: Htϕ(r, θ, φ) =====
def Htϕ(r, θ, ϕ):
    sum_term = mp.mpc(0)
    for n in range(1, N+1):
        coeff = (1j)**(n+1) * (-1)**n * (2*n+1) / (n*(n+1))
        term = coeff * (
            -1j * a(n, α, β, m) * ξ(n, k*r) * τ(n, θ) +
            b(n, α, β, m) * ξdiff(n, k*r) * pifunc(n, θ)
        )
        sum_term += term
    return E0 / (k * r) * sqrt(ϵ / μ) * cos(ϕ) * (
        k * r * exp(-1j * k * r * cos(θ)) + sum_term
    )

# ===== Eq. 40: 放射強度 S(r, θ, φ) =====
def S(r, θ, ϕ):
    etθ = Etθ(r, θ, ϕ)
    etϕ = Etϕ(r, θ, ϕ)
    etr = Etr(r, θ, ϕ)
    htθ = Htθ(r, θ, ϕ)
    htϕ = Htϕ(r, θ, ϕ)
    htr = Htr(r, θ, ϕ)

    term1 = cos(θ) * (etθ * mp.conj(htϕ) - etϕ * mp.conj(htθ))
    term2 = -sin(θ) * (etϕ * mp.conj(htr) - etr * mp.conj(htϕ))
    return mp.re(0.5 * (term1 + term2))







# === 任意精度設定 ===
mp.dps = 10  # 小数点以下の精度

# === 空間グリッド設定 ===
x = np.linspace(0.0, 20.0e-6, 40)  # [m]
y = np.linspace(0.0, 20.0e-6, 40)  # [m]
z = mp.mpf(10.0e-6)                # z位置 [m]
Nx = len(x)
Ny = len(y)

# === 結果格納用配列 ===
result_arr = np.zeros((Ny, Nx))

# === 散乱強度 S の評価（Eq. 40）===
for j in range(Ny):
    print("j =", j)
    for i in tqdm(range(Nx), desc="計算中"):
        xi = mp.mpf(x[i])
        yj = mp.mpf(y[j])
        ri = sqrt(xi**2 + yj**2 + z**2)
        θ = acos(z / ri)
        ϕ = mp.mpf(0.0) if (x[i] == 0 and y[j] == 0) else atan2(yj, xi)
        result_arr[j, i] = float(re(S(ri, θ, ϕ)))  # mpmath → floatへ変換


# µm単位に変換
x_um = x * 1e6
y_um = y * 1e6
X, Y = np.meshgrid(x_um, y_um)

import os

# フォルダとファイル名を別々に定義
data_dir = "00_Mie/data"
fig_dir = "00_Mie/figs"
dat_filename = "Mie_20um_opaque_particle_05umres_z10um.dat"
# fig1_filename = "raw_isophote_20um_opaque_particle_05umres_z10um.svg"
# fig2_filename = "raw_heatmap_20um_opaque_particle_05umres_z10um.svg"

# フォルダがなければ作成
os.makedirs(data_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# データ保存
with open(os.path.join(data_dir, dat_filename), "w") as f:
    for j in range(Ny):
        for i in range(Nx):
            f.write(f"{x_um[i]:.6f} {y_um[j]:.6f} {result_arr[j, i]:.8e}\n")

# # 等高線付きグラフ保存
# plt.figure(figsize=(8, 6))
# cp = plt.contourf(X, Y, result_arr, levels=60, cmap='viridis', vmin=0.0, vmax=0.0025)
# plt.colorbar(cp, label="S [W/m²]")
# plt.xlabel("x [µm]")
# plt.ylabel("y [µm]")
# plt.title("Isophote behind 20-µm opaque Particle")
# plt.gca().set_aspect('equal')
# plt.xlim(0, 20)
# plt.ylim(0, 20)
# plt.tight_layout()
# plt.savefig(os.path.join(fig_dir, fig1_filename))
# plt.show()

# # 等高線なしグラフ保存
# plt.figure(figsize=(8, 6))
# im = plt.imshow(result_arr, extent=[0, 20, 0, 20], origin='lower', cmap='viridis', vmin=0.0, vmax=0.0025, aspect='equal')
# plt.colorbar(im, label="S [W/m²]")
# plt.xlabel("x [µm]")
# plt.ylabel("y [µm]")
# plt.title("Isophote Heatmap behind 20-µm opaque Particle")
# plt.tight_layout()
# plt.savefig(os.path.join(fig_dir, fig2_filename))
# plt.show()
