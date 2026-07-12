function EE_update(path::String, model, indexA::Vector{Int64}, Sweeps::Int64, ss::Vector{Matrix{UInt8}}, record)
    name = name_Lattice(model.Lattice)
    global LOCK = ReentrantLock()
    if model.Θquench == 0.0
        file = "$(path)/minusUEE$(name)_t$(model.Ht)U$(model.Hu1)size$(model.site)Δt$(model.Δt)Θ$(model.Θrelax).csv"
    else
        file = "$(path)/minusUEE$(name)_t$(model.Ht)U$(model.Hu1)_$(model.Hu2)size$(model.site)Δt$(model.Δt)Θ$(model.Θrelax)_$(model.Θquench).csv"
    end

    rng = MersenneTwister(Threads.threadid() + time_ns())
    TTT = time_ns()
    Ns = model.Ns
    lθ = div(model.Nt, 2)
    NN = length(model.nodes)
    Θidx = div(NN, 2) + 1

    UPD = UpdateBuffer()
    SCEE = SCEEBuffer(model.Ns)
    G1 = G4Buffer(model.Ns, NN)
    G2 = G4Buffer(model.Ns, NN)

    indexAbar = idxbar_F(model.Lattice, model.site, indexA)

    D, D_, tmpN, tmpNN = SCEE.D, SCEE.D_, SCEE.N, SCEE.NN

    tmpO = 0.0
    counter = 0

    # ============ INIT: Build Ls, Rs, BMs, Bt0s ============
    G1.Ls[:, NN] .= model.Pt
    G2.Ls[:, NN] .= model.Pt
    G1.Rs[:, 1] .= model.Pt
    G2.Rs[:, 1] .= model.Pt

    for idx in 1:NN-1
        BM_F!(tmpN, tmpNN, view(G1.BMs, :, :, idx), model, ss[1], idx)
        BM_F!(tmpN, tmpNN, view(G2.BMs, :, :, idx), model, ss[2], idx)
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

    # ============ SWEEP LOOP ============
    idx = 1
    for loop in 1:Sweeps
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

        for lt in 1:model.Nt
            D .= @view model.exp_αη_pos[lt, view(ss[1], :, lt)]
            D_ .= @view model.exp_αη_pos[lt, view(ss[2], :, lt)]

            # --- Propagate Rt ---
            WrapKV!(tmpN, model.eK, model.eKinv, D, G1.Rt, "Forward", "R")
            WrapKV!(tmpN, model.eK, model.eKinv, D_, G2.Rt, "Forward", "R")

            # --- Propagate Bt0 per-step ---
            if lt < lθ
                D .= 1 ./ D
                D_ .= 1 ./ D_
                WrapB!(tmpNN, model.eK, model.eKinv, D, G1.Bt0, "Forward", lt < lθ)
                WrapB!(tmpNN, model.eK, model.eKinv, D_, G2.Bt0, "Forward", lt < lθ)
            elseif lt == lθ
                # At τ=θ transition: B(θ,θ) = I in both conventions
                fill!(G1.Bt0, 0)
                fill!(G2.Bt0, 0)
                for j in diagind(G1.Bt0)
                    G1.Bt0[j] = 1.0
                end
                for j in diagind(G2.Bt0)
                    G2.Bt0[j] = 1.0
                end
                D .= 1 ./ D
                D_ .= 1 ./ D_
            else
                WrapB!(tmpNN, model.eK, model.eKinv, D, G1.Bt0, "Forward", lt < lθ)
                WrapB!(tmpNN, model.eK, model.eKinv, D_, G2.Bt0, "Forward", lt < lθ)
                D .= 1 ./ D
                D_ .= 1 ./ D_
            end

            # --- Propagate Lt ---
            WrapKV!(tmpN, model.eK, model.eKinv, D, G1.Lt, "Forward", "L")
            WrapKV!(tmpN, model.eK, model.eKinv, D_, G2.Lt, "Forward", "L")

            # --- Update HS field ---
            if lt == lθ
                UpdateEELayerTheta!(rng, view(ss[1], :, lt), view(ss[2], :, lt), lt, G1, G2, model, UPD, tmpN, tmpNN)
            else
                UpdateEELayer!(lt < lθ, rng, view(ss[1], :, lt), view(ss[2], :, lt), lt, G1, G2, model, UPD, tmpN, tmpNN)
            end

            # ========== VERIFICATION ==========
            ERROR = 5e-4
            Ltt1, Rtt1 = LRt(lt, model, ss[1])
            Ltt2, Rtt2 = LRt(lt, model, ss[2])
            L001, R001 = LRt(lθ, model, ss[1])
            L002, R002 = LRt(lθ, model, ss[2])
            errrr = [norm(Ltt1 - G1.Lt / norm(G1.Lt)), norm(Rtt1 - G1.Rt / norm(G1.Rt)),
                norm(Ltt2 - G2.Lt / norm(G2.Lt)), norm(Rtt2 - G2.Rt / norm(G2.Rt)),
                norm(L001 - G1.L0 / norm(G1.L0)), norm(R001 - G1.R0 / norm(G1.R0)),
                norm(L002 - G2.L0 / norm(G2.L0)), norm(R002 - G2.R0 / norm(G2.R0))]
            if !(maximum(errrr) < ERROR)
                error("lt=", lt, " lθ=", lθ, " Update error:  ", errrr)
            end

            tmpO += EE_cal(model.binoms_sq, G1.L0, G2.L0, G1.R0, G2.R0, indexA, indexAbar)
            counter += 1

            # --- Node handling ---
            if any(model.nodes .== lt)
                idx += 1
                BM_F!(tmpN, tmpNN, view(G1.BMs, :, :, idx - 1), model, ss[1], idx - 1)
                BM_F!(tmpN, tmpNN, view(G2.BMs, :, :, idx - 1), model, ss[2], idx - 1)

                mul!(view(G1.Rs, :, idx), view(G1.BMs, :, :, idx - 1), view(G1.Rs, :, idx - 1))
                mul!(view(G2.Rs, :, idx), view(G2.BMs, :, :, idx - 1), view(G2.Rs, :, idx - 1))

                # Recompute / extend Bt0s BEFORE using it
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
                # Reset Bt0 at every node to prevent numerical drift from
                # step-by-step propagation; for idx < Θidx uses init Bt0s
                # (correct since future steps beyond node_idx untouched)
                G1.Bt0 .= G1.Bt0s[:, :, idx]
                G2.Bt0 .= G2.Bt0s[:, :, idx]
            end
        end

        if record
            lock(LOCK) do
                open(file, "a") do io
                    writedlm(io, tmpO / counter, ',')
                end
            end
        end
        tmpO = 0.0
        counter = 0
    end
    TTT = round(Int, (time_ns() - TTT) / 1e9)
    hour = TTT ÷ 3600
    minite = (TTT % 3600) ÷ 60
    second = TTT % 60
    println("      acc = ", round(100 * UPD.acc / prod(size(ss[1])) / Sweeps / 4, digits=2), "%", "  $(Sweeps) Sweep finished in ", string(lpad(string(hour), 2, '0'), ":", lpad(string(minite), 2, '0'), ":", lpad(string(second), 2, '0')))
end

function UpdateEELayer!(tless0, rng, s1, s2, lt, G1, G2, model, UPD, tmpN, tmpNN)
    LR1 = dot(G1.Lt, G1.Rt)
    LR2 = dot(G2.Lt, G2.Rt)

    for i in axes(s1, 1)
        begin
            sx = rand(rng, model.samplers_vec[s1[i]])
            UPD.Δ = exp(model.αη[lt, sx] - model.αη[lt, s1[i]]) - 1
            UPD.r = abs2(1 + G1.Lt[i] * UPD.Δ * G1.Rt[i] / LR1)^model.Nb
            p = UPD.r * model.γ[sx] / model.γ[s1[i]]

            if rand(rng) < p
                s1[i] = sx
                UPD.acc += 1
                if tless0
                    mul!(tmpN, G1.Bt0, G1.Rt)
                    axpy!(UPD.Δ * G1.Rt[i], G1.Bt0[:, i], tmpN)
                    G1.R0 .= tmpN / norm(tmpN)
                else
                    mul!(tmpN, G1.Bt0', G1.Lt)
                    axpy!(UPD.Δ * G1.Lt[i], conj.(G1.Bt0[i, :]), tmpN)
                    G1.L0 .= tmpN / norm(tmpN)
                    G1.Bt0[i, :] .+= UPD.Δ * G1.Bt0[i, :]
                end
                G1.Rt[i] += UPD.Δ * G1.Rt[i]
                LR1 = dot(G1.Lt, G1.Rt)
            end
        end

        begin
            sx = rand(rng, model.samplers_vec[s2[i]])
            UPD.Δ = exp(model.αη[lt, sx] - model.αη[lt, s2[i]]) - 1
            UPD.r = abs2(1 + G2.Lt[i] * UPD.Δ * G2.Rt[i] / LR2)^model.Nb
            p = UPD.r * model.γ[sx] / model.γ[s2[i]]
            if rand(rng) < p
                s2[i] = sx
                UPD.acc += 1
                if tless0
                    mul!(tmpN, G2.Bt0, G2.Rt)
                    axpy!(UPD.Δ * G2.Rt[i], G2.Bt0[:, i], tmpN)
                    G2.R0 .= tmpN / norm(tmpN)
                else
                    mul!(tmpN, G2.Bt0', G2.Lt)
                    axpy!(UPD.Δ * G2.Lt[i], conj.(G2.Bt0[i, :]), tmpN)
                    G2.L0 .= tmpN / norm(tmpN)
                    G2.Bt0[i, :] .+= UPD.Δ * G2.Bt0[i, :]
                end
                G2.Rt[i] += UPD.Δ * G2.Rt[i]
                LR2 = dot(G2.Lt, G2.Rt)
            end
        end
    end
end

function UpdateEELayerTheta!(rng, s1, s2, lt, G1, G2, model, UPD, tmpN, tmpNN)
    # At τ=θ: Bt0=I.
    #   R(θ) = B(θ,0)|P⟩  includes B_step(θ) → affected by HS update at θ
    #   L(θ) = ⟨P|B(2θ,θ) does NOT include B_step(θ) → NOT affected
    #   B(θ,θ) = I is a fundamental identity → must remain I
    # Only update R0; leave L0 and Bt0 unchanged.
    LR1 = dot(G1.Lt, G1.Rt)
    LR2 = dot(G2.Lt, G2.Rt)

    for i in axes(s1, 1)
        begin
            sx = rand(rng, model.samplers_vec[s1[i]])
            UPD.Δ = exp(model.αη[lt, sx] - model.αη[lt, s1[i]]) - 1
            UPD.r = abs2(1 + G1.Lt[i] * UPD.Δ * G1.Rt[i] / LR1)^model.Nb
            p = UPD.r * model.γ[sx] / model.γ[s1[i]]

            if rand(rng) < p
                s1[i] = sx
                UPD.acc += 1
                # R0 update: R'(θ) = normalize(Rt + Δ*Rt[i]*e_i)
                tmpN .= G1.Rt
                axpy!(UPD.Δ * G1.Rt[i], G1.Bt0[:, i], tmpN)
                G1.R0 .= tmpN / norm(tmpN)
                # L0 and Bt0 must NOT change at τ=θ
                # Rt update
                G1.Rt[i] += UPD.Δ * G1.Rt[i]
                LR1 = dot(G1.Lt, G1.Rt)
            end
        end

        begin
            sx = rand(rng, model.samplers_vec[s2[i]])
            UPD.Δ = exp(model.αη[lt, sx] - model.αη[lt, s2[i]]) - 1
            UPD.r = abs2(1 + G2.Lt[i] * UPD.Δ * G2.Rt[i] / LR2)^model.Nb
            p = UPD.r * model.γ[sx] / model.γ[s2[i]]
            if rand(rng) < p
                s2[i] = sx
                UPD.acc += 1
                # R0 update: same as replica 1
                tmpN .= G2.Rt
                axpy!(UPD.Δ * G2.Rt[i], G2.Bt0[:, i], tmpN)
                G2.R0 .= tmpN / norm(tmpN)
                # L0 and Bt0 must NOT change at τ=θ
                G2.Rt[i] += UPD.Δ * G2.Rt[i]
                LR2 = dot(G2.Lt, G2.Rt)
            end
        end
    end
end