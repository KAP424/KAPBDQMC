# 一维，半系统纠缠熵在固定系统大小，变化boson数的纠缠熵变化

# 还需要考察非半系统的情况

include("C:\\Users\\22423\\Desktop\\KAPDQMC\\code\\KAPBDQMC\\src\\public\\Geometry.jl")
using LinearAlgebra, LinearAlgebra.BLAS, LinearAlgebra.LAPACK
using Plots, Random

function EEvsArea_hc(L, nb)
    Lattice = "HoneyComb120"
    EE_set = []
    site = [L, L]
    K = nnK_Matrix(Lattice, site)
    Ns = size(K, 1)
    Nb = Int(nb * Ns)
    E, V = LAPACK.syevd!('V', 'L', K)
    Pt = V[:, 1]

    for ll in 0:Ns
        idxA = randperm(Ns)[1:ll]

        idxAbar = idxbar_F(Lattice, site, idxA)

        AA = dot(Pt[idxA], Pt[idxA])
        AbarAbar = dot(Pt[idxAbar], Pt[idxAbar])

        if abs(AA - AbarAbar) > 1e-10
            # println("area:$area AA ≠ AbarAbar, AA=$AA, AbarAbar=$AbarAbar")
            EE = 0
            for k in 0:Nb
                EE += binomial(big(Nb), k)^2 * AA^(2 * (Nb - k)) * AbarAbar^(2 * k)
            end
            push!(EE_set, -log(EE))
        else
            para = AA
            EE = -log(binomial(big(2 * Nb), Nb) * para^(2 * Nb))
            push!(EE_set, EE)
        end
    end
    return collect(0:Ns), EE_set
end

L = 12

nb_set = [1 // 3, 1 // 2, 1]

p = plot()

for nb in nb_set
    Area_set, EE_set = EEvsArea_hc(L, nb)
    plot!(Area_set, EE_set, markershape=:star5,
    xlabel="Area", ylabel="EE", title="L=$L HC120",
    legend=true, label="nb=$(nb.num)/$(nb.den)", lw=2)
end

display(p)

savefig("test\\Free\\HC120\\EE_vs_area.png")