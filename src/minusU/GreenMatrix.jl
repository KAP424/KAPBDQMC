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
        @inbounds @simd for i in 1:model.Ns
            tmpN[i] = model.exp_αη_pos[lt, s[i, lt]]
        end
        mul!(tmpNN, model.eK, BM)
        mul!(BM, Diagonal(tmpN), tmpNN)
    end

end

function BMinv_F!(tmpN, tmpNN, BM, model, s::Array{UInt8,2}, idx::Int64)
    """
    不包头包尾
    """
    @assert 0 < idx <= length(model.nodes)

    fill!(BM, 0)
    @inbounds for i in diagind(BM)
        BM[i] = 1
    end

    for lt in model.nodes[idx]+1:model.nodes[idx+1]
        @inbounds for i in 1:model.Ns
            tmpN[i] = model.exp_αη_neg[lt, s[i, lt]]
        end
        mul!(tmpNN, BM, model.eKinv)
        mul!(BM, tmpNN, Diagonal(tmpN))
    end
end


function WrapKV!(tmpNN, eK, eKinv, D, G, direction, LR)
    if direction == "Forward"
        if LR == "L"
            mul!(tmpNN, eK, G)
            mul!(G, Diagonal(D), tmpNN)
        elseif LR == "R"
            mul!(tmpNN, G, eKinv)
            mul!(G, tmpNN, Diagonal(D))
        elseif LR == "B"
            mul!(tmpNN, eK, G)
            mul!(G, tmpNN, eKinv)
            mul!(tmpNN, Diagonal(D), G)
            D .= 1 ./ D
            mul!(G, tmpNN, Diagonal(D))
        end
    elseif direction == "Backward"
        if LR == "L"
            mul!(tmpNN, Diagonal(D), G)
            mul!(G, eKinv, tmpNN)
        elseif LR == "R"
            mul!(tmpNN, G, Diagonal(D))
            mul!(G, tmpNN, eK)
        elseif LR == "B"
            mul!(tmpNN, Diagonal(D), G)
            D .= 1 ./ D
            mul!(G, tmpNN, Diagonal(D))
            mul!(tmpNN, eKinv, G)
            mul!(G, tmpNN, eK)
        end
    end
end
