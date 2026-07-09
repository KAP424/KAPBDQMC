mutable struct PhyBuffer_{T<:Number}
    PLs::Matrix{T}
    PRs::Matrix{T}
    BM::Matrix{T}
    F::Matrix{T}

    # temporaries
    tmpN::Vector{T}
    tmpNN::Matrix{T}
end

mod1(10, 2)

mutable struct G4Buffer_{T<:Number}
    BMs::Array{T,3}
    Lt::Vector{T}
    Rt::Vector{T}
    L0::Vector{T}
    R0::Vector{T}
    Bt0::Matrix{T}

    Ls::Matrix{T}
    Rs::Matrix{T}
    Bt0s::Array{T,3}
    # F::Matrix{T}

end

mutable struct EEBuffer_{T<:Number}
    N::Vector{T}
    N_::Vector{T}
    NN::Matrix{T}

end