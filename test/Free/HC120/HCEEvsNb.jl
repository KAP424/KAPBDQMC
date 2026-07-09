# 一维，半系统纠缠熵在固定系统大小，变化boson数的纠缠熵变化

# 还需要考察非半系统的情况

include("C:\\Users\\22423\\Desktop\\KAPDQMC\\code\\KAPBDQMC\\src\\public\\Boson.jl")
include("C:\\Users\\22423\\Desktop\\KAPDQMC\\code\\KAPBDQMC\\src\\public\\Geometry.jl")
using LinearAlgebra, LinearAlgebra.BLAS, LinearAlgebra.LAPACK
using Plots

function EEvsNb_hc(L, area)
    Lattice = "HoneyComb120"
    site = [L, L]
    K = nnK_Matrix(Lattice, site)
    Ns = size(K, 1)
    E, V = LAPACK.syevd!('V', 'L', K)
    Pt = V[:, 1]

    if area == "half"
        idxA = area_index(Lattice, site, ([1, 1], [div(L, 2), L]))
    elseif area == "quarter"
        idxA = area_index(Lattice, site, ([1, 1], [div(L, 2), div(L, 2)]))
    end
    idxAbar = idxbar_F(Lattice, site, idxA)

    AA = dot(Pt[idxA], Pt[idxA])
    AbarAbar = dot(Pt[idxAbar], Pt[idxAbar])


    if abs(AA - AbarAbar) > 1e-10
        println("area:$area AA ≠ AbarAbar, AA=$AA, AbarAbar=$AbarAbar")
        Nb_set = collect(1:30)
        EE_set = []
        for Nb in Nb_set
            EE = 0
            for k in 0:Nb
                EE += binomial(big(Nb), k)^2 * AA^(2 * (Nb - k)) * AbarAbar^(2 * k)
            end
            push!(EE_set, -log(EE))
        end
    else
        para = AA
        Nb_set = collect(1:30)
        EE_set = []
        for Nb in Nb_set
            EE = -log(binomial(big(2 * Nb), Nb) * para^(2 * Nb))
            push!(EE_set, EE)
        end
    end
    return Nb_set, EE_set
end

L = 12
area_set = ["half", "quarter"]

p = plot()
for LEE in area_set
    Nb_set, EE_set = EEvsNb_hc(L, LEE)
    plot!(Nb_set, EE_set, markershape=:star5, xscale=:ln,
        xlabel="Nb", ylabel="EE", title="L=$L HC120",
        legend=true, label="area=$LEE", lw=2)
end

Nb_set, EE_set = EEvsNb_hc(L, "half")


X = log.(Nb_set[15:end])
Y = EE_set[15:end]
# 构建矩阵 [X 1]
A = hcat(X, ones(length(X)))
# 求解最小二乘：A * [a; b] ≈ Y
coeffs = inv(A' * A) * (A' * Y)
a = Float64(coeffs[1])
b = Float64(coeffs[2])
y = a * log.(Nb_set) .+ b .+ 0.3

plot!(Nb_set, y, label="ln Fit: y=$(round(a; digits=3)) lnx + $(round(b; digits=3))",
    lw=2, color=:black)
display(p)


savefig("test\\Free\\HC120\\EE_vs_Nb_L=$L.png")
