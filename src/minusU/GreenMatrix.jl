function Initial_s(model, rng::MersenneTwister)::Array{UInt8,2}
    sp = Random.Sampler(rng, [1, 2, 3, 4])
    s = zeros(model.Ns, model.Nt)
    for i in eachindex(s)
        s[i] = rand(rng, sp)
    end
    return s
end

function BM_F!(tmpN, tmpNN, BM, model, s::Array{UInt8,2}, idx::Int64)
    """
    不包头包尾
    """
    @assert 0 < idx <= length(model.nodes)

    fill!(BM, 0)
    @inbounds for i in diagind(BM)
        BM[i] = 1
    end
    for lt in model.nodes[idx]+1:model.nodes[idx+1]
        tmpN .= model.exp_αη_pos[lt, s[:, lt]]
        mul!(tmpNN, model.eK, BM)
        mul!(BM, Diagonal(tmpN), tmpNN)
    end

end


function WrapB!(tmpNN, eK, eKinv, D, G, direction, tless0)
    if direction == "Forward"
        if tless0
            # D is inversed!
            mul!(tmpNN, G, eKinv)
            mul!(G, tmpNN, Diagonal(D))
        else
            mul!(tmpNN, eK, G)
            mul!(G, Diagonal(D), tmpNN)
        end
    elseif direction == "Backward"
        if tless0
            mul!(tmpNN, G, Diagonal(D))
            mul!(G, tmpNN, eK)
        else
            # D is inversed!
            mul!(tmpNN, Diagonal(D), G)
            mul!(G, eKinv, tmpNN)
        end
    end
end

function WrapKV!(tmpN, eK, eKinv, D, G, direction, LR)
    if direction == "Forward"
        if LR == "L"
            mul!(tmpN, eKinv, G)
            mul!(G, Diagonal(D), tmpN)
        elseif LR == "R"
            mul!(tmpN, eK, G)
            mul!(G, Diagonal(D), tmpN)
        else
            error("WrapKV! LR must be L or R")
        end
    elseif direction == "Backward"
        if LR == "L"
            mul!(tmpN, Diagonal(D), G)
            mul!(G, eK, tmpN)
        elseif LR == "R"
            mul!(tmpN, eKinv, G)
            mul!(G, Diagonal(D), tmpN)
        else
            error("WrapKV! LR must be L or R")
        end
    end
end

# --------------------------------------------------------------------

function LRt(t, model, s)
    L = copy(model.Pt)
    R = copy(model.Pt)

    eV = zeros(model.Ns)

    count = 0
    for i in 1:t
        eV .= model.exp_αη_pos[i, s[:, i]]
        R = Diagonal(eV) * model.eK * R
        count += 1
        if count == 5
            R ./= norm(R)
            count = 0
        end
    end

    count = 0
    for i in model.Nt:-1:t+1
        eV .= model.exp_αη_pos[i, s[:, i]]
        L = model.eK * Diagonal(eV) * L
        count += 1
        if count == 5
            L ./= norm(L)
            count = 0
        end
    end
    return L / norm(L), R / norm(R)
end

