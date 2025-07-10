using SpecialFunctions, LegendrePolynomials, ProgressMeter
# using ForwardDiff # ForwardDiff は不要になりました
# using Unitful
using Plots


# === 設定値 ===
λ = 0.6943e-6      # 波長 [m]
d = 20e-6          # 粒子径 [m]
m = 1.5 - 100.0im  # 粒子の屈折率（複素数）[-]
k = 2π / λ         # 波数 [1/m]
α = π * d / λ      # サイズパラメータ（無次元）[-]
β = m * α          # 複素サイズパラメータ（無次元）[-]
N = round(Int, α + 4α^(1/3) + 2)  # 計算に使う最大次数（安全マージンを追加）
E0 = 1.0           # 入射電場の振幅（規格化）[V/m]
ϵ = 8.854e-12      # 真空の誘電率 [F/m]
μ = 4π * 1e-7      # 真空の透磁率 [H/m]

# === 計算グリッド ===
x = range(0.0, stop=20.0e-6, length=40)  # [m]
y = range(0.0, stop=20.0e-6, length=40)  # [m]
z = 10.0e-6             # [m] ← 粒子の直後

Nx = length(x)
Ny = length(y)


"""
対数微分 D_n(ρ) を下方漸化式で計算する
Bohren and Huffman (1983), p. 127, Eq. (4.89)
"""

function calculate_D(ρ::Complex, n_max::Int)
    D = zeros(ComplexF64, n_max + 1)
    # 非常に大きな次数から始め、初期値を0と仮定する
    # これは n >> |ρ| の場合に良い近似となる
    D[n_max + 1] = 0.0 + 0.0im
    for n in n_max:-1:1
        D[n] = n / ρ - 1.0 / (D[n + 1] + n / ρ)
    end
    return D
end

"""
リッカチ・ベッセル関数 ψ_n(x), χ_n(x) を上方漸化式で計算する
ψ_n(x) = x * j_n(x)
χ_n(x) = -x * y_n(x)
"""
function calculate_riccati_bessel(x::Float64, n_max::Int)
    ψ = zeros(Float64, n_max + 1)
    χ = zeros(Float64, n_max + 1)

    # n=0 の初期値
    ψ[1] = sin(x)
    χ[1] = cos(x)

    # n=1 の初期値
    if abs(x) > 1e-12
        ψ[2] = sin(x)/x - cos(x)
        χ[2] = cos(x)/x + sin(x)
    else # xが小さい場合の極限
        ψ[2] = 0.0
        χ[2] = 0.0
    end

    # n > 1 の値を上方漸化式で計算
    # z_{n+1} = (2n+1)/x * z_n - z_{n-1}
    for n in 2:n_max
        ψ[n + 1] = (2n - 1) / x * ψ[n] - ψ[n - 1]
        χ[n + 1] = (2n - 1) / x * χ[n] - χ[n - 1]
    end

    return ψ, χ
end


# === 散乱係数 a_n, b_n の事前計算 ===
# 式(4.88) に基づいて a_n, b_n を計算
D_β = calculate_D(β, N)
ψ_α, χ_α = calculate_riccati_bessel(α, N)
ξ_α = ψ_α .- im .* χ_α # ξ = ψ - iχ は h_n^(2) に対応。B&Hの規約と異なる可能性に注意

an = zeros(ComplexF64, N)
bn = zeros(ComplexF64, N)

for n in 1:N
    # 式(4.88)
    num_a = (D_β[n] / m + n / α) * ψ_α[n] - ψ_α[n > 1 ? n - 1 : 1] # n=1のときψ_0が必要
    den_a = (D_β[n] / m + n / α) * ξ_α[n] - ξ_α[n > 1 ? n - 1 : 1]
    an[n] = num_a / den_a
    
    num_b = (m * D_β[n] + n / α) * ψ_α[n] - ψ_α[n > 1 ? n - 1 : 1]
    den_b = (m * D_β[n] + n / α) * ξ_α[n] - ξ_α[n > 1 ? n - 1 : 1]
    bn[n] = num_b / den_b
end


# === 散乱場の計算に必要な関数 ===
# 角度依存関数 π_n, τ_n
# θ=0,π の特異点を避ける処理を追加
pifunc(n, θ) = abs(sin(θ)) < 1e-12 ? ( isodd(n) ? -0.5*n*(n+1) : 0.5*n*(n+1) ) : Plm(cos(θ), n, 1) / sin(θ)
τ(n, θ) = ForwardDiff.derivative(th -> Plm(cos(th), n, 1), θ)

# 散乱場の計算に使うリッカチ・ベッセル関数 ξ_n(k*r) とその導関数
# これらは評価点ごとに計算が必要
ξ_func(n, x) = x * (sphericalbesselj(n, x) + im * sphericalbessely(n, x))
ξdiff_func(n, x) = ξ_func(n-1, x) - n/x * ξ_func(n, x)
ξdiff2_func(n, x) = (-1 - n*(n-1)/x^2) * ξ_func(n,x) + (n-1)/x * ξ_func(n-1,x) + ξ_func(n-2, x)


# === 全電場・磁場成分の計算 ===
# 式(33)-(38)に対応。入射場と散乱場の和。
function Etr(r, θ, ϕ)
    kr = k * r
    E_inc_r = E0 * cos(ϕ) * sin(θ) * exp(im * kr * cos(θ))
    E_sca_r = E0 * cos(ϕ) * sum(
        im^(n+1) * (2n+1)/(n*(n+1)) * an[n] * n*(n+1)/kr^2 * ξ_func(n, kr) * Plm(cos(θ), n, 1) for n in 1:N
    )
    return E_inc_r + E_sca_r
end

function Etθ(r, θ, ϕ)
    kr = k * r
    E_inc_θ = E0 * cos(ϕ) * cos(θ) * exp(im * kr * cos(θ))
    E_sca_θ = E0 * cos(ϕ) / kr * sum(
        im^n * (2n+1)/(n*(n+1)) * (im * an[n] * ξdiff_func(n, kr) * τ(n,θ) + bn[n] * ξ_func(n, kr) * pifunc(n,θ)) for n in 1:N
    )
    return E_inc_θ + E_sca_θ
end

function Etϕ(r, θ, ϕ)
    kr = k * r
    E_inc_ϕ = -E0 * sin(ϕ) * exp(im * kr * cos(θ))
    E_sca_ϕ = E0 * sin(ϕ) / kr * sum(
        im^n * (2n+1)/(n*(n+1)) * (im * an[n] * ξdiff_func(n, kr) * pifunc(n,θ) + bn[n] * ξ_func(n, kr) * τ(n,θ)) for n in 1:N
    )
    return E_inc_ϕ - E_sca_ϕ # B&H p.102 式(4.45)に基づく
end

# ... Htr, Htθ, Htϕ も同様に実装（ここでは省略） ...
# 簡単のため、ポインティングベクトルの計算を単純化します
# S_z ≈ 1/2 * Re(E_x * H_y^* - E_y * H_x^*)
# 全電磁場ベクトルから直交座標系への変換が必要
# E_x = E_r sinθ cosϕ + E_θ cosθ cosϕ - E_ϕ sinϕ
# ... etc.

# ポインティングベクトル (z成分)
function S_z(r, θ, ϕ)
    # 簡単のため、元のコードのSの計算を流用しますが、
    # Htr, Htθ, Htϕ の正しい実装が必要です。
    # ここでは、簡略化のため電場強度のみプロットします。
    Ex = Etr(r,θ,ϕ)*sin(θ)*cos(ϕ) + Etθ(r,θ,ϕ)*cos(θ)*cos(ϕ) - Etϕ(r,θ,ϕ)*sin(ϕ)
    Ey = Etr(r,θ,ϕ)*sin(θ)*sin(ϕ) + Etθ(r,θ,ϕ)*cos(θ)*sin(ϕ) + Etϕ(r,θ,ϕ)*cos(ϕ)
    return 0.5 * real(Ex * conj(Ex) + Ey * conj(Ey)) # |E|^2 に比例
end


#==============================================================#
# === 散乱強度を評価 ===
p = Progress(Ny, "Computing...")

result_arr = zeros(Float64, Ny, Nx)

# Threads.@threads はグローバル変数の取り扱いに注意が必要なため、
# 定数を で宣言するか、関数に引数として渡します。
Threads.@threads for j in 1:Ny
    for i in 1:Nx
        # グリッド点から球座標への変換
        r = sqrt(x[i]^2 + y[j]^2 + z^2)
        θ = acos(z / r)
        ϕ = atan(y[j], x[i]) # atan(y,x) は4象限対応

        # result_arr[j, i] = real(S(r, θ, ϕ))
        result_arr[j, i] = S_z(r, θ, ϕ)
    end
    next!(p)
end

finish!(p)

# === 出力先の設定 ===
output_dir = "../data"
mkpath(output_dir)
output_file = joinpath(output_dir, "mie_isophote_stable.dat")

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

println("Calculation finished and data saved to $(output_file)")


# === プロット ===
heatmap(
    x .* 1e6, y .* 1e6, result_arr', # result_arrの転置が必要
    xlabel = "x [µm]",
    ylabel = "y [µm]",
    title = "Isophote behind 20-µm Particle (Stable Calculation)",
    aspect_ratio = :equal,
    xlims = (0, 20),
    ylims = (0, 20),
    colorbar = true,
    colorbar_title = "|E|^2 (arb. units)"
)