using SpecialFunctions
using Plots

# --- 計算関数の定義 (変更なし) ---

# リッカティ・ベッセル関数 ψ_n(x)
ψ(n, x) = sqrt(π*x/2) * besselj(n + 0.5, x)

# ψ_n(x) の導関数
ψdiff(n, x) = sqrt(π*x/2) * (besselj(n - 0.5, x) - (n + 0.5)/x * besselj(n + 0.5, x))

# D_n(ρ) の定義に基づく直接計算（不安定）
function calculate_D_original(n::Int, ρ::ComplexF64)
    try
        return ψdiff(n, ρ) / ψ(n, ρ)
    catch e
        return complex(NaN, NaN)
    end
end

# 安定な下方漸化式による計算
function calculate_D_stable(ρ::ComplexF64, n_max::Int)
    D = zeros(ComplexF64, n_max + 20)
    for n in (n_max + 19):-1:1
        D[n] = n / ρ - 1.0 / (D[n+1] + n / ρ)
    end
    return D[1:n_max]
end


# --- グラフ化および誤差率計算処理 ---

betas_to_test = [
    complex(1.5, 0.0),
    complex(1.5, -5.0),
    complex(1.5, -10.0),
    complex(1.5, -100.0) # オーバーフローが起きやすいケース
]
n_max = 50
n_range = 1:n_max

output_dir = "Dn_graphs_with_error_rate"
mkpath(output_dir)

println("="^60)
println("D_n Calculation Stability and Error Rate Analysis")
println("="^60)

for β in betas_to_test
    # --- データ生成 ---
    d_stable = calculate_D_stable(β, n_max)
    d_original = [calculate_D_original(n, β) for n in n_range]

    # --- 平均誤差率の計算 ---
    total_error_rate = 0.0
    valid_points_count = 0
    
    for n in n_range
        stable_val = d_stable[n]
        original_val = d_original[n]

        # `d_original`が有限の数で、`d_stable`がゼロでない場合のみ誤差を計算
        if isfinite(original_val) && abs(stable_val) > 1e-12
            error_rate = abs(stable_val - original_val) / abs(stable_val)
            total_error_rate += error_rate
            valid_points_count += 1
        end
    end
    
    # 平均誤差率を計算
    mean_error_percentage = if valid_points_count > 0
        (total_error_rate / valid_points_count) * 100
    else
        0.0 # 有効な点がなければ誤差率0
    end
    
    failure_points_count = n_max - valid_points_count

    # --- 結果のサマリーを出力 ---
    println("\n--- Analysis for β = $(β) ---")
    @printf("Mean Error Rate (on %d valid points): %.6f %%\n", valid_points_count, mean_error_percentage)
    @printf("Calculation Failure Points: %d / %d\n", failure_points_count, n_max)


    # --- グラフ作成 ---
    # title="D_n(ρ) for β = $(β)",
    default(framestyle=:box, grid=false, dpi=300, size=(800, 700),labelfontsize=16, legendfontsize=14, tickfontsize=14, legend=:outertop)
    p = plot(n_range, real.(d_stable), label="Re(D_n) - Stable", 
             xlabel="Order n", ylabel="Value of D_n", linewidth=2.5, color=:blue)
    plot!(p, n_range, imag.(d_stable), label="Im(D_n) - Stable", linewidth=2.5, color=:red)
    plot!(p, n_range, real.(d_original), label="Re(D_n) - Unstable", linestyle=:dash, linewidth=2, color=:cyan)
    plot!(p, n_range, imag.(d_original), label="Im(D_n) - Unstable", linestyle=:dash, linewidth=2, color=:magenta)
    
    # --- 画像保存 ---
    beta_str = replace(string(β), " " => "", "+" => "", "im" => "i")
    filename = joinpath(output_dir, "Dn_comparison_beta_$(beta_str).png")
    savefig(p, filename)
    println("Graph saved to: $(filename)")
end

println("\n" * "="^60)
println("All analyses are complete.")