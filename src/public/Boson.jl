
function get_F!(Nb, PL, PR, F)
    mul!(F, PR, PL')
    nrm = dot(PL, PR)
    F .*= Nb / nrm
end

function EE_cal(Nb, L1, L2, R1, R2, indexA, indexAbar)
    ans = 0
    a = dot(L1[indexAbar], R1[indexAbar]) * dot(L2[indexAbar], R2[indexAbar])
    b = dot(L1[indexA], R2[indexA]) * dot(L2[indexA], R1[indexA])

    for k in 0:Nb
        ans += binomial(Nb, k)^2 * a^(Nb - k) * b^k
    end
    return ans
end


# using LinearAlgebra

# Lattice = "HoneyComb120"
# Initial = "H0"
# L = 6
# site = [L, L]
# Nb = div(prod(site), 2)
# Ns = 2 * prod(site)

# K = nnK_Matrix(Lattice, site)
# E, V = LAPACK.syevd!('V', 'L', K)


# L1 = V[:, 1]
# L2 = V[:, 1]
# R1 = V[:, 1]
# R2 = V[:, 1]

# L1 /= sqrt(dot(L1, R1))
# R1 /= sqrt(dot(L1, R1))
# L2 /= sqrt(dot(L2, R2))
# R2 /= sqrt(dot(L2, R2))

# idxA = area_index(Lattice, site, ([1, 1], [L, div(L, 2)]))
# idxAbar = idxAbar_F(Lattice, site, idxA)


# -log(EE_cal(Nb, L1, L2, R1, R2, idxA, idxAbar))


