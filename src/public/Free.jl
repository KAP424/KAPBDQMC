function Free_EE(Lattice, site, idxA, Nb)

    K = nnK_Matrix(Lattice, site)

    Ns = size(K, 1)
    idxAbar = idxbar_F(Lattice, site, idxA)

    E, V = LAPACK.syevd!('V', 'L', K)
    Pt = V[:, 1]

    AA = dot(Pt[idxA], Pt[idxA])
    AbarAbar = dot(Pt[idxAbar], Pt[idxAbar])

    if abs(AA - AbarAbar) > 1e-10
        EE = 0
        for k in 0:Nb
            EE += binomial(big(Nb), k)^2 * AA^(2 * (Nb - k)) * AbarAbar^(2 * k)
        end
        return -log(EE)
    else
        EE = -log(binomial(big(2 * Nb), Nb) * AA^(2 * Nb))
        return -log(EE)
    end

end
