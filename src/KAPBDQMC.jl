module KAPBDQMC
using LinearAlgebra, LinearAlgebra.BLAS, LinearAlgebra.LAPACK

include("public/Geometry.jl")
export name_Lattice, nnidx_F, area_index, i_xy, xy_i, idxbar_F
# , nnn2idx, n3n2idx
# export nnK_Matrix, nnnK_Matrix, n3nK_Matrix, Initial_Pt!

include("public/Buffer.jl")

include("public/Boson.jl")
export EE_cal

# Declare unified API to be extended by submodules via multiple dispatch
function phy_update end
function Initial_s end
function EE_update end
# function ctrl_EEicr end
# function ctrl_SCDOPicr end

include("minusU/minusU.jl")
using .minusU: Model_Para
export Model_Para

export Initial_s, phy_update, EE_update
# ctrl_SCEEicr, ctrl_EEicr, ctrl_SCDOPicr
end


