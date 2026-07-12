mutable struct PhyBuffer_{T<:Number}
    Ls::Matrix{T}
    Rs::Matrix{T}
    BM::Matrix{T}
    Lt::Vector{T}
    Rt::Vector{T}

    # temporaries
    tmpN::Vector{T}
    tmpNN::Matrix{T}
end

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

mutable struct SCEEBuffer_{T<:Number}
    D::Vector{T}
    D_::Vector{T}
    N::Vector{T}
    NN::Matrix{T}
end