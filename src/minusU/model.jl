# H_{U_1}=U_1 \sum_i (b_i^\dagger b_i + c_i^\dagger c_i)^2

struct Model_Para_
    Nb::Int64
    Lattice::String
    Ht::Float64
    Hu1::Float64
    Hu2::Float64
    site::Vector{Int64}
    Θrelax::Float64
    Θquench::Float64
    Ns::Int64
    Nt::Int64
    K::Array{Float64,2}
    BatchSize::Int64
    Δt::Float64
    # α::Vector{Float64}
    # η::Vector{Float64}
    exp_αη_pos::Matrix{Float64}  # 大小: length(α) × 4
    exp_αη_neg::Matrix{Float64}  # 大小: length(α) × 4
    αη::Matrix{Float64}  # 大小: length(α) × 4
    γ::Vector{Float64}
    Pt::Vector{Float64}
    HalfeK::Array{Float64,2}
    eK::Array{Float64,2}
    HalfeKinv::Array{Float64,2}
    eKinv::Array{Float64,2}
    nodes::Vector{Int64}
    samplers_vec::Vector{Random.Sampler}
    binoms_sq::Vector{Float64}
    # flux::Float64
end


function Model_Para(; nb, Ht, Hu1, Hu2, Δt, Θrelax, Θquench, Lattice::String, site, BatchSize, Initial::String)
    # flux=0.0, opt="xy"
    Nt = round(Int, 2 * (Θrelax + Θquench) / Δt)
    if (Θquench > 0.0) & (abs(Hu1 - Hu2) > 0)
        Hu = LinRange(Hu1, Hu2, round(Int, Θquench / Δt) + 1)[2:end]
        Hu = vcat(fill(Hu1, round(Int, Θrelax / Δt)), collect(Hu), reverse(collect(Hu)), fill(Hu1, round(Int, Θrelax / Δt)))
    else
        @assert (Hu1 == Hu2) & (Θquench < 1e-7) "For Θquench=0, Hu1 must equal Hu2"
        Hu = Hu1 .* ones(Float64, Nt)
    end

    @assert norm(reverse(Hu) - Hu) < 1e-10 "HU profile is not symmetric!"
    @assert length(Hu) == Nt "Length of Hu profile does not match Nt!"

    α = sqrt.(Δt .* Hu)
    γ = [1 + sqrt(6) / 3, 1 + sqrt(6) / 3, 1 - sqrt(6) / 3, 1 - sqrt(6) / 3]
    η = [sqrt(2 * (3 - sqrt(6))), -sqrt(2 * (3 - sqrt(6))), sqrt(2 * (3 + sqrt(6))), -sqrt(2 * (3 + sqrt(6)))]

    # 预计算每个状态对应的指数值，避免重复计算
    # exp(-α*η) 和 exp(α*η) 对于每个状态值是常数
    # 状态值 s ∈ {1,2,3,4} 对应 model.η 的索引
    exp_αη_neg = [exp(-i * j) for i in α, j in η]  # 大小: length(α) × 4
    exp_αη_pos = [exp(i * j) for i in α, j in η]    # 大小: length(α) × 4
    αη = [i * j for i in α, j in η]


    K = nnK_Matrix(Lattice, site)
    Ns = size(K, 1)
    Nb = Int(nb * Ns)

    E, V = LAPACK.syevd!('V', 'L', Ht * K[:, :])
    if abs(E[div(Ns, 2)] - E[div(Ns, 2)+1]) > 1e-10
        @warn "Warning: The non-interacting system may be gapped!"
    end
    HalfeK = V * Diagonal(exp.(-Δt .* E ./ 2)) * V'
    eK = V * Diagonal(exp.(-Δt .* E)) * V'
    HalfeKinv = V * Diagonal(exp.(Δt .* E ./ 2)) * V'
    eKinv = V * Diagonal(exp.(Δt .* E)) * V'
    # @assert norm(eK * eKinv - I(size(eK, 1))) < 1e-10 "eK*eKinv does not equal identity!"

    Pt = V[:, 1]  # 预分配 Pt

    if div(Nt, 2) % BatchSize == 0
        nodes = collect(0:BatchSize:Nt)
    else
        nodes = vcat(0, reverse(collect(div(Nt, 2)-BatchSize:-BatchSize:1)), collect(div(Nt, 2):BatchSize:Nt), Nt)
    end

    rng = MersenneTwister(Threads.threadid() + time_ns())
    samplers_vec = Vector{Random.Sampler}(undef, 4)
    for excluded in 1:4
        allowed = [i for i in 1:4 if i != excluded]
        samplers_vec[excluded] = Random.Sampler(rng, allowed)
    end

    println("$(Lattice) size=$(site)  Δt=$(Δt)  Θ=$(Θrelax)+$(Θquench)  U=$(Hu1)--$(Hu2)  Initial=$Initial  BS=$(BatchSize)  $(Nt)*$(Ns)*$(size(K))")

    binoms_sq = [Float64(binomial(Nb, k))^2 for k in 0:Nb]

    return Model_Para_(Nb, Lattice, Ht, Hu1, Hu2, site, Θrelax, Θquench,
        Ns, Nt, K, BatchSize, Δt, exp_αη_pos, exp_αη_neg, αη, γ,
        Pt, HalfeK, eK, HalfeKinv, eKinv, nodes, samplers_vec, binoms_sq)
end

mutable struct UpdateBuffer_
    acc::Int64
    Δ::Float64
    r::Float64
    # subidx::Vector{Int64}
end

function UpdateBuffer()
    return UpdateBuffer_(
        0,
        0,
        0
        # [0],
    )
end

function PhyBuffer(Ns, NN)
    return PhyBuffer_{Float64}(
        Matrix{Float64}(undef, Ns, NN),
        Matrix{Float64}(undef, Ns, NN),
        Matrix{Float64}(undef, Ns, Ns),
        Matrix{Float64}(undef, Ns, Ns),
        Vector{Float64}(undef, Ns),
        Matrix{Float64}(undef, Ns, Ns)
    )
end

function G4Buffer(Ns, NN)
    return G4Buffer_{Float64}(
        Array{Float64}(undef, Ns, Ns, NN - 1),
        Vector{Float64}(undef, Ns),
        Vector{Float64}(undef, Ns),
        Vector{Float64}(undef, Ns),
        Vector{Float64}(undef, Ns),
        Matrix{Float64}(undef, Ns, Ns),
        Matrix{Float64}(undef, Ns, NN),
        Matrix{Float64}(undef, Ns, NN),
        Array{Float64}(undef, Ns, Ns, NN),
    )
end

function SCEEBuffer(Ns)
    return SCEEBuffer_{Float64}(
        Vector{Float64}(undef, Ns),
        Vector{Float64}(undef, Ns),
        Vector{Float64}(undef, Ns),
        Matrix{Float64}(undef, Ns, Ns)
    )
end
