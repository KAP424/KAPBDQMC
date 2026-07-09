include("C:\\Users\\22423\\Desktop\\KAPDQMC\\code\\KAPBDQMC\\src\\public\\Boson.jl")
include("C:\\Users\\22423\\Desktop\\KAPDQMC\\code\\KAPBDQMC\\src\\public\\Geometry.jl")
using LinearAlgebra, LinearAlgebra.BLAS, LinearAlgebra.LAPACK
using Plots

function EEvsNb_area(L, Nb)
    Lattice = "1d"
    site = [L]
    Ns = L
    K = nnK_Matrix(Lattice, site)
    E, V = LAPACK.syevd!('V', 'L', K)
    Pt = V[:, 1]

    EE_set = []
    area_set = collect(0:L)

    for LEE in area_set
        idxA = area_index(Lattice, site, ([1], [LEE]))
        idxAbar = idxbar_F(Lattice, site, idxA)

        AA = dot(Pt[idxA], Pt[idxA])
        AbarAbar = dot(Pt[idxAbar], Pt[idxAbar])

        EE = 0
        for k in 0:Nb
            EE += binomial(Nb, k)^2 * AA^(2 * (Nb - k)) * AbarAbar^(2 * k)
        end
        push!(EE_set, -log(EE))

    end

    return area_set, EE_set

end

L = 40
p = plot()
for Nb in 0:30
    area_set, EE_set = EEvsNb_area(L, Nb)
    plot!(area_set, EE_set, xlabel="area",
        ylabel="EE", title="L=$L",
        legend=:outerright, label="Nb=$Nb", lw=2)
end

display(p)

savefig("test\\Free\\1d\\EE_vs_area_L=$L.png")

