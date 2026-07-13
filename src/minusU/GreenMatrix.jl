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

function Initial_PhyBuffer!(G::PhyBuffer_, model, s)
    # Initial G.Ls, G.Rs[:,1]
    Ns, NN = size(G.Ls)
    G.Ls[:, NN] .= model.Pt
    G.Rs[:, 1] .= model.Pt

    for idx in NN-1:-1:1
        BM_F!(G.tmpN, G.tmpNN, G.BM, model, s, idx)
        mul!(view(G.Ls, :, idx), G.BM', view(G.Ls, :, idx + 1))
        G.Ls[:, idx] ./= norm(G.Ls[:, idx])
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
            mul!(tmpN, Diagonal(D), G)
            mul!(G, eKinv, tmpN)
        else
            error("WrapKV! LR must be L or R")
        end
    end
end


function UpdatePhyLayer!(rng, s, lt, G, model, UPD)
    LR = dot(G.Lt, G.Rt)
    for i in axes(s, 1)
        sx = rand(rng, model.samplers_vec[s[i]])
        UPD.Δ = exp(model.αη[lt, sx] - model.αη[lt, s[i]]) - 1
        UPD.r = abs2(1 + G.Lt[i] * UPD.Δ * G.Rt[i] / LR)^model.Nb
        p = UPD.r * model.γ[sx] / model.γ[s[i]]

        if rand(rng) < p
            s[i] = sx
            UPD.acc += 1
            G.Rt[i] += UPD.Δ * G.Rt[i]
            LR = dot(G.Lt, G.Rt)
        end
    end
end


function WrapLR0!(model, lt, s, G)
    G.L0 .= G.Lt
    G.R0 .= G.Rt
    if lt > model.Nt / 2
        for t in lt:-1:div(model.Nt, 2)+1
            G.D .= @view model.exp_αη_pos[lt, view(s, :, t)]
            WrapKV!(G.tmpN, model.eK, model.eKinv, G.D, G.L0, "Backward", "L")
            G.D .= 1 ./ G.D
            WrapKV!(G.tmpN, model.eK, model.eKinv, G.D, G.R0, "Backward", "R")
        end
    else
        for t in lt+1:div(model.Nt, 2)
            G.D .= @view model.exp_αη_pos[lt, view(s, :, t)]
            WrapKV!(G.tmpN, model.eK, model.eKinv, G.D, G.R0, "Forward", "R")
            G.D .= 1 ./ G.D
            WrapKV!(G.tmpN, model.eK, model.eKinv, G.D, G.L0, "Forward", "L")
        end
    end
    G.L0 ./= norm(G.L0)
    G.R0 ./= norm(G.R0)
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

