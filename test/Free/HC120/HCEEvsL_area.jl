# 一维，半系统纠缠熵在固定系统大小，变化boson数的纠缠熵变化

# 还需要考察非半系统的情况

include("C:\\Users\\22423\\Desktop\\KAPDQMC\\code\\KAPBDQMC\\src\\public\\Boson.jl")
include("C:\\Users\\22423\\Desktop\\KAPDQMC\\code\\KAPBDQMC\\src\\public\\Geometry.jl")
using LinearAlgebra, LinearAlgebra.BLAS, LinearAlgebra.LAPACK
using Plots, Random


function EEvsL_area_hc(nb)
    Lattice = "HoneyComb120"
    L_set = collect(3:3:18)
    EE_set = []
    for L in L_set
        site = [L, L]
        K = nnK_Matrix(Lattice, site)
        Ns = size(K, 1)
        E, V = LAPACK.syevd!('V', 'L', K)
        Pt = V[:, 1]

        idxA_set = [area_index(Lattice, site, ([1, 1], [div(L, 3), L])), randperm(Ns)[1:div(Ns, 3)],
            area_index(Lattice, site, ([1, 1], [div(L, 3), div(2 * L, 3)])), randperm(Ns)[1:div(2 * Ns, 9)]]

        for idxA in idxA_set
            idxAbar = idxbar_F(Lattice, site, idxA)

            AA = dot(Pt[idxA], Pt[idxA])
            AbarAbar = dot(Pt[idxAbar], Pt[idxAbar])

            Nb = Int(nb * Ns)

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
    end
    return L_set, EE_set
end

nb_set = [1//3, 1//2, 1//1]

p = plot(legend=:outerright)

colors = distinguishable_colors(2 * length(nb_set))

i = 0
for nb in nb_set
    L_set, EE_set = EEvsL_area_hc(nb)
    ll = length(L_set)

    plot!(L_set, EE_set[1:4:end], markershape=:star, markersize=10, xscale=:ln,
        xlabel="L", ylabel="EE", title="HC120", color=colors[2*i+1],
        legend=true, label="nb=$(nb.num)/$(nb.den) half1", lw=2)
    plot!(L_set, EE_set[2:4:end], markershape=:rect, markersize=5, xscale=:ln,
        xlabel="L", ylabel="EE", title="HC120", color=colors[2*i+1],
        legend=true, label="nb=$(nb.num)/$(nb.den) half2", lw=2)
    plot!(L_set, EE_set[3:4:end], markershape=:rect, markersize=5, xscale=:ln,
        xlabel="L", ylabel="EE", title="HC120", color=colors[2*i+2],
        legend=true, label="nb=$(nb.num)/$(nb.den) quarter1", lw=2)
    plot!(L_set, EE_set[4:4:end], markershape=:cross, markersize=10, xscale=:ln,
        xlabel="L", ylabel="EE", title="HC120", color=colors[2*i+2],
        legend=true, label="nb=$(nb.num)/$(nb.den) quarter2", lw=2)
    i += 1
end
display(p)

savefig("test\\Free\\HC120\\EE_vs_area.png")