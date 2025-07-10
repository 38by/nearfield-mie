import numpy as np
from mpmath import besselj, sqrt, pi, mp, mpc, acos, atan2, re, cos, sin, exp, diff
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import lpmv
import os

# === 定数設定 ===
λ = 0.6943e-6
d = 20e-6
m = mpc(1.5, -100)
k = 2 * np.pi / λ
α = np.pi * d / λ
β = m * α
N = int(round(α + 4 * α**(1/3))) ### ▼▼▼ 変更点: int型にキャスト
E0 = 1.0
ϵ = 8.854e-12
μ = 4 * np.pi * 1e-7

# === 任意精度設定 ===
mp.dps = 15 # 精度を少し上げておくことを推奨

# =======================
# Eq. 10: ψ_n(x)
# =======================
def ψ(n, x):
    if x == 0: return mp.mpc(0)
    return sqrt(pi * x / 2) * besselj(n + 0.5, x)

def ψ_n_minus_1(n, x):
    """ψ_{n-1}(x)を計算する。n=1のときにψ_0が必要。"""
    return ψ(n - 1, x)

# ... (ψdiff, ψdiff2, χ, χdiff, χdiff2, ξ, ξdiff, ξdiff2 は変更なし) ...
def ψdiff(n, x):
    if x == 0: return mp.mpc(0)
    return sqrt(pi / (2 * x)) * ((n + 1) * besselj(n + 0.5, x) - x * besselj(n + 1.5, x))

def ψdiff2(n, x):
    if x == 0: return mp.mpc(0)
    x = mp.mpf(x)
    term1 = x**2 * besselj(n - 1.5, x)
    term2 = -2 * x**2 * besselj(n + 0.5, x)
    term3 = x**2 * besselj(n + 2.5, x)
    term4 = 2 * x * besselj(n - 0.5, x)
    term5 = -2 * x * besselj(n + 1.5, x)
    term6 = -besselj(n + 0.5, x)
    return (1 / (4 * x**(3/2))) * sqrt(pi / 2) * (term1 + term2 + term3 + term4 + term5 + term6)

def χ(n, x):
    if x == 0: return mp.mpc(float('inf')) # 特異点
    return (-1)**n * sqrt(pi * x / 2) * besselj(-n - 0.5, x)

def χdiff(n, x):
    if x == 0: return mp.mpc(float('inf')) # 特異点
    x = mp.mpf(x)
    term1 = x * besselj(-n - 1.5, x)
    term2 = besselj(-n - 0.5, x)
    term3 = -x * besselj(-n + 0.5, x)
    return 0.5 * (-1)**n * sqrt(pi / (2 * x)) * (term1 + term2 + term3)

def χdiff2(n, x):
    if x == 0: return mp.mpc(float('inf')) # 特異点
    x = mp.mpf(x)
    term1 = x**2 * besselj(-n - 2.5, x)
    term2 = -2 * x**2 * besselj(-n - 0.5, x)
    term3 = x**2 * besselj(-n + 1.5, x)
    term4 = 2 * x * besselj(-n - 1.5, x)
    term5 = -2 * x * besselj(-n + 0.5,x) # 元のコードのバグ修正
    return (1 / (4 * x**(3/2))) * sqrt(pi / 2) * (-1)**n * (term1 + term2 + term3 + term4 + term5)

def ξ(n, x):
    return ψ(n, x) + 1j * χ(n, x)

def ξdiff(n, x):
    return ψdiff(n, x) + 1j * χdiff(n, x)

def ξdiff2(n, x):
    return ψdiff2(n, x) + 1j * χdiff2(n, x)

### ▼▼▼ 変更点: ここから新しい関数群を追加 ▼▼▼

# =======================
# 対数微分 D_n(ρ) の計算 (Bohren & Huffman, p.127, Eq. 4.89)
# 下方漸化式で計算
# =======================
def calculate_D(rho, n_max):
    # D_n(rho) = ψ_n'(rho) / ψ_n(rho)
    # 実際にはn_maxより少し大きい次数から計算を開始するため、配列を大きめに確保
    D = np.zeros(n_max + 16, dtype=np.complex128)
    # mpmathの型に変換
    rho_mpc = mpc(rho)
    
    # 漸化式は次数を下げながら計算
    for n in range(n_max + 14, 0, -1):
        D[n] = n / rho_mpc - 1.0 / (D[n + 1] + n / rho_mpc)
        
    # n=1からn_maxまでの結果をmpmathの型で返す
    return [mpc(val) for val in D[1:n_max+1]]

# =======================
# 効率的な a_n, b_n の計算 (Bohren & Huffman, p.127, Eq. 4.88)
# =======================
def calculate_an(n, α, m, D_n_β):
    """D_n(β) を使って a_n を計算"""
    temp = (D_n_β / m + n / α)
    num = temp * ψ(n, α) - ψ_n_minus_1(n, α)
    den = temp * ξ(n, α) - (ψ(n-1, α) + 1j*χ(n-1, α)) # ξ_{n-1}(α)
    return num / den

def calculate_bn(n, α, m, D_n_β):
    """D_n(β) を使って b_n を計算"""
    temp = (m * D_n_β + n / α)
    num = temp * ψ(n, α) - ψ_n_minus_1(n, α)
    den = temp * ξ(n, α) - (ψ(n-1, α) + 1j*χ(n-1, α)) # ξ_{n-1}(α)
    return num / den

### ▲▲▲ 変更点: ここまで ▲▲▲


# ... (Plm, dnPl, pifunc, τ, dξdr, d2ξdr2 は変更なし) ...
def Plm(x, l, m=1):
    return lpmv(m, l, float(x))

def dnPl(x, l, m=1):
    x_mp = mp.mpf(x)
    return float(diff(lambda t: lpmv(m, l, float(t)), x_mp))

def pifunc(n, θ):
    cθ = float(cos(θ))
    sθ = float(sin(θ))
    if abs(sθ) < 1e-12: return 0.5 * n * (n + 1)
    return Plm(cθ, n, 1) / sθ

def τ(n, θ):
    cθ = float(cos(θ))
    sθ = float(sin(θ))
    if abs(sθ) < 1e-12: return 0.0
    return -sθ * dnPl(cθ, n, 1)

def dξdr(n, r):
    return k * ξdiff(n, k * r)

def d2ξdr2(n, r):
    return k**2 * ξdiff2(n, k * r)


### ▼▼▼ 変更点: 電場・磁場・S関数の引数を変更 ▼▼▼
# 散乱係数を引数として受け取るように変更

def Etr(r, θ, ϕ, a_n_coeffs):
    sum_term = mpc(0)
    for n in range(1, N + 1):
        coeff = (1j)**(n + 1) * (-1)**n * (2*n + 1) / (n*(n + 1))
        term = coeff * a_n_coeffs[n-1] * (ξdiff2(n, k*r) + ξ(n, k*r)) * Plm(cos(θ), n, 1)
        sum_term += term
    return E0 * cos(ϕ) * (sin(θ) * exp(-1j * k * r * cos(θ)) + sum_term)

def Etθ(r, θ, ϕ, a_n_coeffs, b_n_coeffs):
    sum_term = mpc(0)
    for n in range(1, N + 1):
        coeff = (1j)**(n + 1) * (-1)**n * (2*n + 1) / (n*(n + 1))
        term = coeff * (a_n_coeffs[n-1] * ξdiff(n, k*r) * τ(n, θ) - 1j * b_n_coeffs[n-1] * ξ(n, k*r) * pifunc(n, θ))
        sum_term += term
    return E0 * cos(ϕ) / (k * r) * (k*r * cos(θ) * exp(-1j * k * r * cos(θ)) + sum_term)

def Etϕ(r, θ, ϕ, a_n_coeffs, b_n_coeffs):
    sum_term = mpc(0)
    for n in range(1, N + 1):
        coeff = (1j)**(n + 1) * (-1)**n * (2*n + 1) / (n*(n + 1))
        term = coeff * (a_n_coeffs[n-1] * ξdiff(n, k*r) * pifunc(n, θ) - 1j * b_n_coeffs[n-1] * ξ(n, k*r) * τ(n, θ))
        sum_term += term
    return -E0 * sin(ϕ) / (k * r) * (k * r * exp(-1j * k * r * cos(θ)) + sum_term)

def Htr(r, θ, ϕ, b_n_coeffs):
    sum_term = mpc(0)
    for n in range(1, N + 1):
        coeff = (1j)**(n+1) * (-1)**n * (2*n+1) / (n*(n+1))
        term = coeff * b_n_coeffs[n-1] * (ξdiff2(n, k*r) + ξ(n, k*r)) * Plm(cos(θ), n, 1)
        sum_term += term
    return E0 * sqrt(ϵ/μ) * sin(ϕ) * (sin(θ) * exp(-1j * k * r * cos(θ)) + sum_term)

def Htθ(r, θ, ϕ, a_n_coeffs, b_n_coeffs):
    sum_term = mpc(0)
    for n in range(1, N + 1):
        coeff = (1j)**(n+1) * (-1)**n * (2*n+1) / (n*(n+1))
        term = coeff * (-1j * a_n_coeffs[n-1] * ξ(n, k*r) * pifunc(n, θ) + b_n_coeffs[n-1] * ξdiff(n, k*r) * τ(n, θ))
        sum_term += term
    return E0 / (k * r) * sqrt(ϵ / μ) * sin(ϕ) * (k * r * cos(θ) * exp(-1j * k * r * cos(θ)) + sum_term)

def Htϕ(r, θ, ϕ, a_n_coeffs, b_n_coeffs):
    sum_term = mpc(0)
    for n in range(1, N + 1):
        coeff = (1j)**(n+1) * (-1)**n * (2*n+1) / (n*(n+1))
        term = coeff * (-1j * a_n_coeffs[n-1] * ξ(n, k*r) * τ(n, θ) + b_n_coeffs[n-1] * ξdiff(n, k*r) * pifunc(n, θ))
        sum_term += term
    return E0 / (k * r) * sqrt(ϵ / μ) * cos(ϕ) * (k * r * exp(-1j * k * r * cos(θ)) + sum_term)

def S(r, θ, ϕ, a_n_coeffs, b_n_coeffs):
    etθ = Etθ(r, θ, ϕ, a_n_coeffs, b_n_coeffs)
    etϕ = Etϕ(r, θ, ϕ, a_n_coeffs, b_n_coeffs)
    etr = Etr(r, θ, ϕ, a_n_coeffs)
    htθ = Htθ(r, θ, ϕ, a_n_coeffs, b_n_coeffs)
    htϕ = Htϕ(r, θ, ϕ, a_n_coeffs, b_n_coeffs)
    htr = Htr(r, θ, ϕ, b_n_coeffs)
    term1 = cos(θ) * (etθ * mp.conj(htϕ) - etϕ * mp.conj(htθ))
    term2 = -sin(θ) * (etϕ * mp.conj(htr) - etr * mp.conj(htϕ))
    return re(0.5 * (term1 + term2))


# === 空間グリッド設定 ===
x = np.linspace(0.0, 20.0e-6, 40)
y = np.linspace(0.0, 20.0e-6, 40)
z = mp.mpf(10.0e-6)
Nx = len(x)
Ny = len(y)

# === 結果格納用配列 ===
result_arr = np.zeros((Ny, Nx))

### ▼▼▼ 変更点: 散乱係数の事前計算 ▼▼▼
print("散乱係数 a_n, b_n を事前計算中...")
D_n_beta_values = calculate_D(β, N)
a_n_coeffs = [calculate_an(n, α, m, D_n_beta_values[n-1]) for n in tqdm(range(1, N + 1), desc="a_n")]
b_n_coeffs = [calculate_bn(n, α, m, D_n_beta_values[n-1]) for n in tqdm(range(1, N + 1), desc="b_n")]
print("事前計算完了。")

# === 散乱強度 S の評価 ===
for j in tqdm(range(Ny), desc="全行の進捗"):
    for i in range(Nx):
        xi = mp.mpf(x[i])
        yj = mp.mpf(y[j])
        
        # 座標変換
        ri = sqrt(xi**2 + yj**2 + z**2)
        θi = acos(z / ri)
        ϕi = atan2(yj, xi) if not (xi == 0 and yj == 0) else mp.mpf(0.0)
        
        ### ▼▼▼ 変更点: S関数に事前計算した係数を渡す ▼▼▼
        s_val = S(ri, θi, ϕi, a_n_coeffs, b_n_coeffs)
        result_arr[j, i] = float(s_val)

# === データ保存とプロット ===
# µm単位に変換
x_um = x * 1e6
y_um = y * 1e6

# ... (以降のファイル保存、プロット部分は変更なし) ...
data_dir = "00_Mie/data"
fig_dir = "00_Mie/figs"
dat_filename = "Mie_20um_opaque_particle_05umres_z10um_optimized.dat"

os.makedirs(data_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

print(f"データを {os.path.join(data_dir, dat_filename)} に保存中...")
with open(os.path.join(data_dir, dat_filename), "w") as f:
    for j in range(Ny):
        for i in range(Nx):
            f.write(f"{x_um[i]:.6f} {y_um[j]:.6f} {result_arr[j, i]:.8e}\n")
print("保存完了。")