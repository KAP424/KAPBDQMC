module minusU
import ..KAPBDQMC: phy_update, Initial_s, get_F!
# , ctrl_SCEEicr, ctrl_SCDOPicr, ctrl_EEicr

using ..KAPBDQMC: name_Lattice, nn2idx, xy_i, i_xy, nnK_Matrix, area_index, nnidx_F
# , Initial_Pt!
using ..KAPBDQMC: PhyBuffer_
# , G4Buffer_, SCEEBuffer_, AreaBuffer_, DOPBuffer_
# using ..KAPBDQMC: inv22!, GroverMatrix, GroverMatrix!

using LinearAlgebra, LinearAlgebra.BLAS, LinearAlgebra.LAPACK
using DelimitedFiles, Random

# 扩展父模块统一 API：导入 `phy_update` 并在本模块中添加方法

include("model.jl")
# include("../public/Gupdate.jl")
include("phy_update.jl")
include("EE.jl")
include("GreenMatrix.jl")

export Model_Para
end

