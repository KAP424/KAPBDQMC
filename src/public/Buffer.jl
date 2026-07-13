mutable struct PhyBuffer_{T<:Number}
    Ls::Matrix{T}
    Rs::Matrix{T}
    BM::Matrix{T}
    Lt::Vector{T}
    Rt::Vector{T}
    L0::Vector{T}
    R0::Vector{T}

    # temporaries
    D::Vector{T}
    tmpN::Vector{T}
    tmpNN::Matrix{T}
end
