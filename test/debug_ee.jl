push!(LOAD_PATH, "C:\\Users\\22423\\Desktop\\KAPDQMC\\code\\KAPBDQMC\\src")
using KAPBDQMC
using Random, LinearAlgebra

rng = MersenneTwister(1234)
model = Model_Para(nb=0.5, Ht=1.0, Hu1=2.0, Hu2=2.0, Θrelax=2.4, Θquench=0.0, Lattice="HoneyComb120",
    site=[6, 6], Δt=0.1, BatchSize=10, Initial="V")

s = Initial_s(model, rng)
ss = [copy(s), copy(s)]

Ns = model.Ns
NN = length(model.nodes)
Θidx = div(NN, 2) + 1
lθ = div(model.Nt, 2)

G1 = KAPBDQMC.minusU.G4Buffer(Ns, NN)
G2 = KAPBDQMC.minusU.G4Buffer(Ns, NN)
SCEE = KAPBDQMC.minusU.SCEEBuffer(Ns)

D, D_, tmpN, tmpNN = SCEE.D, SCEE.D_, SCEE.N, SCEE.NN

G1.Ls[:, NN] .= model.Pt
G2.Ls[:, NN] .= model.Pt
G1.Rs[:, 1] .= model.Pt
G2.Rs[:, 1] .= model.Pt

for idx in 1:NN-1
    KAPBDQMC.minusU.BM_F!(tmpN, tmpNN, view(G1.BMs, :, :, idx), model, ss[1], idx)
    KAPBDQMC.minusU.BM_F!(tmpN, tmpNN, view(G2.BMs, :, :, idx), model, ss[2], idx)
    mul!(view(G1.Rs, :, idx + 1), view(G1.BMs, :, :, idx), view(G1.Rs, :, idx))
    mul!(view(G2.Rs, :, idx + 1), view(G2.BMs, :, :, idx), view(G2.Rs, :, idx))
    G1.Rs[:, idx+1] ./= norm(G1.Rs[:, idx+1])
    G2.Rs[:, idx+1] ./= norm(G2.Rs[:, idx+1])
end
for idx in NN-1:-1:1
    mul!(view(G1.Ls, :, idx), view(G1.BMs, :, :, idx)', G1.Ls[:, idx+1])
    mul!(view(G2.Ls, :, idx), view(G2.BMs, :, :, idx)', G2.Ls[:, idx+1])
    G1.Ls[:, idx] ./= norm(G1.Ls[:, idx])
    G2.Ls[:, idx] ./= norm(G2.Ls[:, idx])
end

G1.Bt0s[:, :, Θidx] .= I(Ns)
G2.Bt0s[:, :, Θidx] .= I(Ns)
for idx in 1:div(NN - 1, 2)
    mul!(view(G1.Bt0s, :, :, Θidx - idx), view(G1.Bt0s, :, :, Θidx - idx + 1), view(G1.BMs, :, :, Θidx - idx))
    mul!(view(G2.Bt0s, :, :, Θidx - idx), view(G2.Bt0s, :, :, Θidx - idx + 1), view(G2.BMs, :, :, Θidx - idx))
end

# Set up sweep state
G1.Lt .= G1.Ls[:, 1]
G1.Rt .= G1.Rs[:, 1]
G1.Bt0 .= G1.Bt0s[:, :, 1]
G1.L0 .= G1.Ls[:, Θidx]
G1.R0 .= G1.Rs[:, Θidx]

G2.Lt .= G2.Ls[:, 1]
G2.Rt .= G2.Rs[:, 1]
G2.Bt0 .= G2.Bt0s[:, :, 1]
G2.L0 .= G2.Ls[:, Θidx]
G2.R0 .= G2.Rs[:, Θidx]

lt = 1
D .= @view model.exp_αη_pos[lt, view(ss[1], :, lt)]
D_ .= @view model.exp_αη_pos[lt, view(ss[2], :, lt)]

KAPBDQMC.minusU.WrapKV!(tmpN, model.eK, model.eKinv, D, G1.Rt, "Forward", "R")
KAPBDQMC.minusU.WrapKV!(tmpN, model.eK, model.eKinv, D_, G2.Rt, "Forward", "R")

D .= 1 ./ D
D_ .= 1 ./ D_
KAPBDQMC.minusU.WrapB!(tmpNN, model.eK, model.eKinv, D, G1.Bt0, "Forward", true)
KAPBDQMC.minusU.WrapB!(tmpNN, model.eK, model.eKinv, D_, G2.Bt0, "Forward", true)

KAPBDQMC.minusU.WrapKV!(tmpN, model.eK, model.eKinv, D, G1.Lt, "Forward", "L")
KAPBDQMC.minusU.WrapKV!(tmpN, model.eK, model.eKinv, D_, G2.Lt, "Forward", "L")

# Force an update: change s1[1] to sx
i = 1
s1 = view(ss[1], :, lt)
sx = rand(rng, model.samplers_vec[s1[i]])
Δval = exp(model.αη[lt, sx] - model.αη[lt, s1[i]]) - 1

# Save old state
s1_old = s1[i]
R0_old = copy(G1.R0)
Bt0_old = copy(G1.Bt0)
Rt_old = copy(G1.Rt)
LR1_old = dot(G1.Lt, G1.Rt)

# Compute the CORRECT new R0 by brute force
ss_correct = deepcopy(ss)
ss_correct[1][i, lt] = sx
R0_correct, _ = KAPBDQMC.minusU.LRt(lθ, model, ss_correct[1])
L0_correct = KAPBDQMC.minusU.LRt(lθ, model, ss_correct[1])[1]

# Apply the incremental update formula
G1.R0 .+= Δval * G1.Rt[i] .* G1.Bt0[:, i]
G1.Rt[i] += Δval * G1.Rt[i]
s1[i] = sx

println("=== R0 update test ===")
println("Old R0 norm = ", norm(R0_old))
println("R0 (update) norm diff vs correct = ", norm(R0_correct - G1.R0 / norm(G1.R0)))
println("dot(R0_correct, R0_update) = ", dot(R0_correct, G1.R0 / norm(G1.R0)))

# Check Bt0 consistency
println("\n=== Bt0 consistency ===")
println("Bt0 norm = ", norm(G1.Bt0))
# Bt0 should satisfy: B(lθ, 1) * R(1) = R(lθ)
# (approximately, up to normalization)
Bt0_Rt = G1.Bt0 * Rt_old
Bt0_Rt_normed = Bt0_Rt / norm(Bt0_Rt)
R0_old_normed = R0_old / norm(R0_old)
println("Bt0*R(1) vs R(lθ): norm diff = ", norm(Bt0_Rt_normed - R0_old_normed))
println("dot = ", dot(Bt0_Rt_normed, R0_old_normed))

# Check the key components of the update
println("\n=== Update components ===")
println("Δ = ", Δval)
println("Rt[i] = ", Rt_old[i])
println("Bt0[:,i] norm = ", norm(G1.Bt0[:, i]))
println("Δ * Rt[i] * Bt0[:,i] norm = ", norm(Δval * Rt_old[i] .* G1.Bt0[:, i]))
println("R0 correction / R0 ratio = ", norm(Δval * Rt_old[i] .* G1.Bt0[:, i]) / norm(R0_old))

# Verify the formula step by step
# R0_correct = B(lθ, 0) * (I + Δ e_i e_i^T) * P / norm(...)
# R0_update = R0_old + Δ * R(1)_i * B(lθ, 1)[:,i]
# Are these equal?

# Compute the full update matrix at time 1: (I + Δ e_i e_i^T) * D_1 * eK
M_orig = Diagonal(model.exp_αη_pos[lt, view(ss[1], :, lt)]) * model.eK
D_new = copy(model.exp_αη_pos[lt, view(ss[1], :, lt)])
D_new[i] = model.exp_αη_pos[lt, sx]
M_new = Diagonal(D_new) * model.eK

println("\n=== Full matrix check ===")
println("Δ * e_i e_i^T: norm = ", Δval)
println("M_orig[", i, ",:] norm = ", norm(M_orig[i, :]))
println("M_new[", i, ",:] norm = ", norm(M_new[i, :]))
println("M_orig - M_new: row ", i, " norm diff = ", norm(M_orig[i, :] - M_new[i, :]))
println("Expected diff: Δ * M_orig[", i, ",:] norm = ", abs(Δval) * norm(M_orig[i, :]))
