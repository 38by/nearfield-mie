using SpecialFunctions, LegendrePolynomials, ProgressMeter
using ForwardDiff
# using Unitful
using Plots
using Statistics          # Added for the mean() function
using Interpolations      # Added for linear_interpolation()
using Images, FileIO      # Added for image processing and saving (Gray, save)


# === 設定値 ===
λ = 0.6943e-6      # 波長 [m]
d = 20e-6          # 粒子径 [m]
m = 1.5 - 100.0im    # 粒子の屈折率（複素数）[-]（無次元）# transparent particle
k = 2π / λ         # 波数 [1/m]
α = π * d / λ      # サイズパラメータ（無次元）[-]
β = m * α          # 複素サイズパラメータ（無次元）[-]
N = round(Int, α + 4α^(1/3))  # 計算に使う最大次数 [-]
E0 = 1.0           # 入射電場の振幅（規格化）[V/m]
ϵ = 8.854e-12      # 真空の誘電率 [F/m]（ファラッド毎メートル）
μ = 4π * 1e-7      # 真空の透磁率 [H/m]（ヘンリー毎メートル）

# x = 0.0:0.2e-6:20.0e-6  # [m] ← 0〜20 µm を m単位で
# y = 0.0:0.2e-6:20.0e-6  # [m]
x = range(0.0, stop=20.0e-6, length=40)  # [m]
y = range(0.0, stop=20.0e-6, length=40)  # [m]
z = 10.0e-6             # [m] ← 粒子の直後

Nx = length(x)
Ny = length(y)

# Eq. 10
ψ(n, x) = sqrt(π*x/2) * besselj(n+1/2, x)
ψdiff(n, x) = ψ(n-1, x) - n/x * ψ(n, x)
ψdiff2(n, x) = ( n * (n + 1) / x^2 - 1 ) * ψ(n, x)

# Eq. 11
χ(n, x) = (-1)^n * sqrt(π * x / 2) * besselj(-n - 1/2, x)
χdiff(n, x) = χ(n-1, x) - n/x * χ(n, x)
χdiff2(n, x) = ( n * (n + 1) / x^2 - 1 ) * χ(n, x)

# Eq. 8 is LegendrePolynomials.Plm

# Eq. 9
ξ(n, x) = ψ(n, x) + im * χ(n, x)
ξdiff(n, x) = ψdiff(n, x) + im * χdiff(n, x)
ξdiff2(n, x) = ψdiff2(n, x) + im * χdiff2(n, x)

# Eq. 12 and 13
# function a(n, α, β, m)
#     num = ψ(n,α)*ψdiff(n,β) - m*ψdiff(n,α)*ψ(n,β)
#     den = ξ(n,α)*ψdiff(n,β) - m*ξdiff(n,α)*ψ(n,β)
#     return num / den
# end
# function b(n, α, β, m)
#     num = m * ψ(n,α) * ψdiff(n,β) - ψdiff(n,α) * ψ(n,β)
#     den = m * ξ(n,α) * ψdiff(n,β) - ξdiff(n,α) * ψ(n,β)
#     return num / den
# end

# Bohren & Huffman (1983)　p.127 Eq. (4.88)
function a(n, α, m, D_n_β)
    temp = (D_n_β[n] / m + n / α)
    num = temp * ψ(n, α) - ψ(n - 1, α)
    den = temp * ξ(n, α) - ξ(n - 1, α)
    return num / den
end
function b(n, α, m, D_n_β)
    temp = (m * D_n_β[n] + n / α)
    num = temp * ψ(n, α) - ψ(n - 1, α)
    den = temp * ξ(n, α) - ξ(n - 1, α)
    return num / den
end

function calculate_D(ρ::Complex, n_max::Int) #対数微分 D_n(ρ)を下方漸化式で計算。
    D =zeros(ComplexF64, n_max + 15) # 実際にはn_maxより少し大きい次数から計算を始めるため、配列も大きめに取る。
    for n in (n_max + 14) : -1 : 1 # 漸化式は次数を下げながら計算する。
        D[n] = n / ρ - 1.0 / (D[n + 1] + n / ρ) # Bohren & Huffman (1983)　p.127 Eq. (4.89)
    end
    return D[1:n_max]
end

D_n_β_value = calculate_D(β, N)  # 対数微分 D_n(β) を計算
a_n_coeffs = [a(n, α, m, D_n_β_value) for n in 1:N]  # a(n, α, β, m) の値を事前計算
b_n_coeffs = [b(n, α, m, D_n_β_value) for n in 1:N]  # b(n, α, β, m) の値を事前計算

# Eq. 20, 21
# pifunc(n, θ) = Plm(cos(θ), n, 1) / sin(θ)
# τ(n, θ) = -sin(θ) * dnPl(cos(θ), n, 1)

# Eq. 20, 21 修正版
pifunc(n, θ) = abs(sin(θ)) < 1e-12 ? 0.5 * n * (n + 1) : Plm(cos(θ), n, 1) / sin(θ)
τ(n, θ) = abs(sin(θ)) < 1e-12 ? 0.0 : -sin(θ) * dnPl(cos(θ), n, 1)

# Eq. 22
dξdr(n, r) = k * ξdiff(n, k * r)

# Eq. 23
d2ξdr2(n, r) = k^2 * ξdiff2(n, k * r)

# Eq. 24
ω = 2π * 3e8 / λ         # または ω = 2π * f, f = c / λ
H0 = -k / (ω * μ) * E0   # これで十分

# Eq. 33
Etr(r, θ, ϕ) = E0 * cos(ϕ) * (
    sin(θ) * exp(-im * k * r * cos(θ)) +
    sum(im^(n+1) * (-1)^n * (2n+1)/(n*(n+1)) * a_n_coeffs[n] *
        (ξdiff2(n, k*r) + ξ(n, k*r)) * Plm(cos(θ), n, 1) for n in 1:N)
)

# Eq. 34
Etθ(r, θ, ϕ) = E0 * cos(ϕ) / (k * r) * (
    k * r * cos(θ) * exp(-im * k * r * cos(θ)) +
    sum(
        im^(n+1) * (-1)^n * (2n + 1) / (n * (n + 1)) *
        (a_n_coeffs[n] * ξdiff(n, k*r) * τ(n, θ) - im * b_n_coeffs[n] * ξ(n, k*r) * pifunc(n, θ))
        for n in 1:N
    )
)

# Eq. 35
Etϕ(r, θ, ϕ) = -E0 * sin(ϕ) / (k * r) * (
    k * r * exp(-im * k * r * cos(θ)) +
    sum(
        im^(n + 1) * (-1)^n * (2n + 1) / (n * (n + 1)) *
        (a_n_coeffs[n] * ξdiff(n, k * r) * pifunc(n, θ) - im * b_n_coeffs[n] * ξ(n, k * r) * τ(n, θ))
        for n in 1:N
    )
)

# Eq. 36
Htr(r, θ, ϕ) = E0 * sqrt(ϵ/μ) * sin(ϕ) * (
    sin(θ) * exp(-im * k * r * cos(θ)) +
    sum(
        im^(n+1) * (-1)^n * (2n+1)/(n*(n+1)) *
        b_n_coeffs[n] * (ξdiff2(n,k*r)+ξ(n,k*r)) * Plm(cos(θ), n, 1)
        for n in 1:N
    )
)

# Eq. 37
Htθ(r, θ, ϕ) = E0/(k*r) * sqrt(ϵ/μ) * sin(ϕ) * (
    k * r * cos(θ) * exp(-im * k * r * cos(θ)) +
    sum(
        im^(n+1) * (-1)^n * (2n+1)/(n*(n+1)) *
        (-im*a_n_coeffs[n] * ξ(n,k*r) * pifunc(n, θ) + b_n_coeffs[n] * ξdiff(n,k*r) * τ(n, θ))
        for n in 1:N
    )
)

# Eq. 38
Htϕ(r, θ, ϕ) = E0/(k*r) * sqrt(ϵ/μ) * cos(ϕ) * (
    k * r * exp(-im * k * r * cos(θ)) +
    sum(
        im^(n+1) * (-1)^n * (2n+1)/(n*(n+1)) *
        (-im*a_n_coeffs[n] * ξ(n,k*r) * τ(n, θ) + b_n_coeffs[n] * ξdiff(n,k*r) * pifunc(n, θ))
        for n in 1:N
    )
)

# Eq. 40
S(r, θ, ϕ) = 1/2 * real(
    cos(θ) * (Etθ(r, θ, ϕ) * conj(Htϕ(r, θ, ϕ)) - Etϕ(r, θ, ϕ) * conj(Htθ(r, θ, ϕ))) -
    sin(θ) * (Etϕ(r, θ, ϕ) * conj(Htr(r, θ, ϕ)) - Etr(r, θ, ϕ) * conj(Htϕ(r, θ, ϕ)))
)

#==============================================================#
# === 散乱強度（S）を評価 ===
p = Progress(Ny, "Computing in parallel...")

result_arr = zeros(Float64, Ny, Nx)

Threads.@threads for j in 1:Ny
    for i in 1:Nx
        r = sqrt(x[i]^2 + y[j]^2 + z^2)
        θ = acos(z / r)
        ϕ = (x[i] == 0 && y[j] == 0) ? 0.0 : atan(y[j], x[i])

        result_arr[j, i] = real(S(r, θ, ϕ))
    end
    next!(p)
end

finish!(p)

# === 出力先の設定 ===
output_dir = "./00_data/Mie_data_v4"  # ここで自由にフォルダ名を指定
mkpath(output_dir)  # フォルダがなければ自動作成

# 出力ファイルパス
output_file = joinpath(output_dir, "mie_isophote.dat")

# === ファイル保存処理 ===
open(output_file, "w") do io
    for j in 1:Ny
        for i in 1:Nx
            xval = x[i] * 1e6  # [m] → [μm]
            yval = y[j] * 1e6
            sval = result_arr[j, i]
            println(io, "$(xval) $(yval) $(sval)")
        end
    end
end

# # === プロット ===
# using Plots
# contour(
#     x .* 1e6, y .* 1e6, result_arr,         # [m] → [µm]
#     xlabel = "x [µm]",
#     ylabel = "y [µm]",
#     title = "Isophote behind 20-µm Transparent Particle (Fig.3)",
#     fill = true,
#     levels = 60,                         # 等高線を細かく
#     clim = (0.0, 0.002),                 # 明るい部分の saturate を防ぐ
#     aspect_ratio = :equal,         # 縦横1:1 (同義)
#     xlims = (0, 20),               # 明示的に範囲指定
#     ylims = (0, 20),
#     colorbar = true,
#     colorbar_title = "S [W/m²]"
# )
# savefig("Mie_data.png")