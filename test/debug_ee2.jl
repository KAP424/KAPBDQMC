push!(LOAD_PATH, "C:\\Users\\22423\\Desktop\\KAPDQMC\\code\\KAPBDQMC\\src")
using KAPBDQMC
using Random, LinearAlgebra

function debug_sweep2()
    rng = MersenneTwister(1234)
    model = Model_Para(nb=0.5, Ht=1.0, Hu1=2.0, Hu2=2.0, Θrelax=2.4, Θquench=0.0, Lattice="HoneyComb120",
        site=[6, 6], Δt=0.1, BatchSize=5, Initial="V")

    s = Initial_s(model, rng)
    ss = [copy(s), copy(s)]

    Ns = model.Ns
    NN = length(model.nodes)
    Θidx = div(NN, 2) + 1
    lθ = div(model.Nt, 2)

    UPD = KAPBDQMC.minusU.UpdateBuffer()
    SCEE = KAPBDQMC.minusU.SCEEBuffer(model.Ns)
    G1 = KAPBDQMC.minusU.G4Buffer(model.Ns, NN)
    G2 = KAPBDQMC.minusU.G4Buffer(model.Ns, NN)

    local D = SCEE.D
    local D_ = SCEE.D_
    local tmpN = SCEE.N
    local tmpNN = SCEE.NN

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

    G1.Lt .= G1.Ls[:, 1]
    G2.Lt .= G2.Ls[:, 1]
    G1.Rt .= G1.Rs[:, 1]
    G2.Rt .= G2.Rs[:, 1]
    G1.Bt0 .= G1.Bt0s[:, :, 1]
    G1.L0 .= G1.Ls[:, Θidx]
    G1.R0 .= G1.Rs[:, Θidx]
    G2.L0 .= G2.Ls[:, Θidx]
    G2.R0 .= G2.Rs[:, Θidx]
    G2.Bt0 .= G2.Bt0s[:, :, 1]

    # Storage for R0 manual check: R0_ref = Bt0 * Rt should equal R(θ)
    local R0_manual = zeros(Ns)

    local idx = 1
    local ERROR_THRESH = 5e-4

    for lt in 1:model.Nt
        D .= @view model.exp_αη_pos[lt, view(ss[1], :, lt)]
        D_ .= @view model.exp_αη_pos[lt, view(ss[2], :, lt)]

        KAPBDQMC.minusU.WrapKV!(tmpN, model.eK, model.eKinv, D, G1.Rt, "Forward", "R")
        KAPBDQMC.minusU.WrapKV!(tmpN, model.eK, model.eKinv, D_, G2.Rt, "Forward", "R")

        if lt < lθ
            D .= 1 ./ D
            D_ .= 1 ./ D_
            KAPBDQMC.minusU.WrapB!(tmpNN, model.eK, model.eKinv, D, G1.Bt0, "Forward", true)
            KAPBDQMC.minusU.WrapB!(tmpNN, model.eK, model.eKinv, D_, G2.Bt0, "Forward", true)
        elseif lt == lθ
            fill!(G1.Bt0, 0)
            for j in diagind(G1.Bt0); G1.Bt0[j] = 1.0; end
            fill!(G2.Bt0, 0)
            for j in diagind(G2.Bt0); G2.Bt0[j] = 1.0; end
            D .= 1 ./ D
            D_ .= 1 ./ D_
        else
            KAPBDQMC.minusU.WrapB!(tmpNN, model.eK, model.eKinv, D, G1.Bt0, "Forward", false)
            KAPBDQMC.minusU.WrapB!(tmpNN, model.eK, model.eKinv, D_, G2.Bt0, "Forward", false)
            D .= 1 ./ D
            D_ .= 1 ./ D_
        end

        KAPBDQMC.minusU.WrapKV!(tmpN, model.eK, model.eKinv, D, G1.Lt, "Forward", "L")
        KAPBDQMC.minusU.WrapKV!(tmpN, model.eK, model.eKinv, D_, G2.Lt, "Forward", "L")

        # BEFORE HS update: check Bt0*Rt vs reference R0
        local Lref1, Rref1 = KAPBDQMC.minusU.LRt(lθ, model, ss[1])
        local Lref2, Rref2 = KAPBDQMC.minusU.LRt(lθ, model, ss[2])
        mul!(R0_manual, G1.Bt0, G1.Rt)
        normalize!(R0_manual)
        local bt0_rt_err = norm(R0_manual - Rref1)
        if bt0_rt_err > 1e-10 || lt <= 5
            println("lt=$lt PRE-update: Bt0*Rt vs LRt(θ) err = $bt0_rt_err")
        end

        if lt == lθ
            KAPBDQMC.minusU.UpdateEELayerTheta!(rng, view(ss[1], :, lt), view(ss[2], :, lt), lt, G1, G2, model, UPD, tmpN, tmpNN)
        else
            KAPBDQMC.minusU.UpdateEELayer!(lt < lθ, rng, view(ss[1], :, lt), view(ss[2], :, lt), lt, G1, G2, model, UPD, tmpN, tmpNN)
        end

        # AFTER HS update: verify
        local Lref1p, Rref1p = KAPBDQMC.minusU.LRt(lθ, model, ss[1])
        local Lref2p, Rref2p = KAPBDQMC.minusU.LRt(lθ, model, ss[2])
        local r0_err1 = norm(Rref1p - G1.R0 / norm(G1.R0))
        local r0_err2 = norm(Rref2p - G2.R0 / norm(G2.R0))

        if r0_err1 > 1e-8 || r0_err2 > 1e-8
            println("  POST-update: R0_err = ($r0_err1, $r0_err2)")
        end

        if lt in model.nodes
            idx += 1
            KAPBDQMC.minusU.BM_F!(tmpN, tmpNN, view(G1.BMs, :, :, idx - 1), model, ss[1], idx - 1)
            KAPBDQMC.minusU.BM_F!(tmpN, tmpNN, view(G2.BMs, :, :, idx - 1), model, ss[2], idx - 1)
            mul!(view(G1.Rs, :, idx), view(G1.BMs, :, :, idx - 1), view(G1.Rs, :, idx - 1))
            mul!(view(G2.Rs, :, idx), view(G2.BMs, :, :, idx - 1), view(G2.Rs, :, idx - 1))

            if idx == Θidx
                for i in Θidx-1:-1:1
                    mul!(view(G1.Bt0s, :, :, i), view(G1.Bt0s, :, :, i + 1), view(G1.BMs, :, :, i))
                    mul!(view(G2.Bt0s, :, :, i), view(G2.Bt0s, :, :, i + 1), view(G2.BMs, :, :, i))
                end
            elseif idx > Θidx
                mul!(view(G1.Bt0s, :, :, idx), view(G1.BMs, :, :, idx - 1), view(G1.Bt0s, :, :, idx - 1))
                mul!(view(G2.Bt0s, :, :, idx), view(G2.BMs, :, :, idx - 1), view(G2.Bt0s, :, :, idx - 1))
            end

            G1.Lt .= G1.Ls[:, idx]
            G1.Rt .= G1.Rs[:, idx]
            G2.Lt .= G2.Ls[:, idx]
            G2.Rt .= G2.Rs[:, idx]
            if idx >= Θidx
                G1.Bt0 .= G1.Bt0s[:, :, idx]
                G2.Bt0 .= G2.Bt0s[:, :, idx]
            end
            println("  NODE idx=$idx: Bt0*Rt err after reset = $(norm(G1.Bt0 * G1.Rt / norm(G1.Bt0 * G1.Rt) - Rref1p))")
        end
    end
end

debug_sweep2()
