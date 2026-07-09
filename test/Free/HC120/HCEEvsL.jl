# 一维，半系统纠缠熵在固定系统大小，变化boson数的纠缠熵变化

# 还需要考察非半系统的情况

include("C:\\Users\\22423\\Desktop\\KAPDQMC\\code\\KAPBDQMC\\src\\public\\Boson.jl")
include("C:\\Users\\22423\\Desktop\\KAPDQMC\\code\\KAPBDQMC\\src\\public\\Geometry.jl")
using LinearAlgebra, LinearAlgebra.BLAS, LinearAlgebra.LAPACK
using Plots

function EEvsL_hc(nb, area)
    Lattice = "HoneyComb120"
    L_set = collect(3:3:18)
    EE_set = []
    for L in L_set
        site = [L, L]
        K = nnK_Matrix(Lattice, site)
        Ns = size(K, 1)
        E, V = LAPACK.syevd!('V', 'L', K)
        Pt = V[:, 1]

        if area == "half"
            idxA = area_index(Lattice, site, ([1, 1], [div(L, 3), L]))
        elseif area == "quarter"
            idxA = area_index(Lattice, site, ([1, 1], [div(L, 3), div(2 * L, 3)]))
        end
        idxAbar = idxbar_F(Lattice, site, idxA)

        AA = dot(Pt[idxA], Pt[idxA])
        AbarAbar = dot(Pt[idxAbar], Pt[idxAbar])

        Nb = Int(nb * Ns)

        if abs(AA - AbarAbar) > 1e-10
            println("area:$area AA ≠ AbarAbar, AA=$AA, AbarAbar=$AbarAbar")
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
    return L_set, EE_set
end

nb_set = [1 / 3, 0.5, 1]
area_set = ["half", "quarter"]

p = plot()
for nb in nb_set
    for LEE in area_set
        L_set, EE_set = EEvsL_hc(nb, LEE)
        plot!(L_set, EE_set, markershape=:star5, xscale=:ln,
            xlabel="L", ylabel="EE", title="HC120",
            legend=true, label="nb=$nb area=$LEE", lw=2)
    end
end
display(p)


savefig("test\\Free\\HC120\\EE_vs_L.png")