
function get_F!(Nb, PL, PR, F)
    mul!(F, PR, PL')
    nrm = dot(PL, PR)
    F .*= Nb / nrm
end

function EE_cal(binoms_sq, L1, L2, R1, R2, indexA, indexAbar)
    # println(norm(L1), norm(L2), norm(R1), norm(R2))
    Nb = length(binoms_sq) - 1
    a = (dot(L1[indexAbar], R1[indexAbar]) * dot(L2[indexAbar], R2[indexAbar]) / dot(L1, R1) / dot(L2, R2))^Nb
    b = dot(L1[indexA], R2[indexA]) * dot(L2[indexA], R1[indexA]) / dot(L1[indexAbar], R1[indexAbar]) / dot(L2[indexAbar], R2[indexAbar])

    ans = 0
    bk = 1
    for k in 0:Nb
        ans += binoms_sq[k+1] * bk
        bk *= b
    end
    # println("EE_cal:  ", a, b, abs2(ans * a))
    return abs2(ans * a)
end


A = rand(10, 10)
a = rand(10)

a .+= 10 .* A[:, 1]



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


