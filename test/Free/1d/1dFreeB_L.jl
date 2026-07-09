include("C:\\Users\\22423\\Desktop\\KAPDQMC\\code\\KAPBDQMC\\src\\public\\Boson.jl")
include("C:\\Users\\22423\\Desktop\\KAPDQMC\\code\\KAPBDQMC\\src\\public\\Geometry.jl")
using LinearAlgebra, LinearAlgebra.BLAS, LinearAlgebra.LAPACK
using Plots

# set EE area = half Lattice 
function EEvsL_1d(nb)
    Lattice = "1d"

    L_set = collect(10:2:40)
    EE_set = []
    for L in L_set
        site = [L]
        Ns = L
        Nb = Int(Ns * nb)
        K = nnK_Matrix(Lattice, site)
        E, V = LAPACK.syevd!('V', 'L', K)
        Pt = V[:, 1]

        idxA = area_index(Lattice, site, ([1], [div(L, 2)]))
        idxAbar = idxbar_F(Lattice, site, idxA)

        AA = dot(Pt[idxA], Pt[idxA])
        AbarAbar = dot(Pt[idxAbar], Pt[idxAbar])

        if abs(AA - AbarAbar) > 1e-10
            println("AA ≠ AAbar")
            EE = 0
            for k in 0:Nb
                EE += binomial(Nb, k)^2 * AA^(2 * (Nb - k)) * AbarAbar^(2 * k)
            end
            push!(EE_set, -log(EE))
        else
            para = AA
            EE = -log(binomial(big(2 * Nb), Nb) * para^(2 * Nb))
            push!(EE_set, EE)
        end
    end

    return L_set, EE_set
end

nb_set = [0.5, 1.0, 1.5]

p = plot()
for nb in nb_set
    L_set, EE_set = EEvsL_1d(nb)
    plot!(L_set, EE_set, xlabel="L", ylabel="EE",
        title="Half area", legend=true,
        label="nb=$nb", xscale=:ln, lw=2)
end

L_set, EE_set = EEvsL_1d(0.5)


X = log.(L_set)
Y = EE_set
# 构建矩阵 [X 1]
A = hcat(X, ones(length(X)))
# 求解最小二乘：A * [a; b] ≈ Y
coeffs = inv(A' * A) * (A' * Y)
a = Float64(coeffs[1])
b = Float64(coeffs[2])
y = a * log.(L_set) .+ b .+ 0.15

plot!(L_set, y,
    label="ln Fit: y=$(round(a, digits=3)) lnx + $(round(b, digits=3))",
    lw=2, color=:black)
display(p)

savefig("test\\Free\\1d\\EE_vs_L_half.png")

