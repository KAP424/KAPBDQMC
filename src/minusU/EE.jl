function EE_update(path::String, model, indexA::Vector{Int64}, Sweeps::Int64, ss::Vector{Matrix{UInt8}}, record)
    ERROR = 5e-4
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
    NN = length(model.nodes)
    Θidx = div(NN, 2) + 1

    UPD = UpdateBuffer()

    G1 = PhyBuffer(Ns, NN)
    G2 = PhyBuffer(Ns, NN)

    Initial_PhyBuffer!(G1, model, ss[1])
    Initial_PhyBuffer!(G2, model, ss[2])

    indexAbar = idxbar_F(model.Lattice, model.site, indexA)

    tmpO = 0.0
    counter = 0

    # ============ SWEEP LOOP ============
    idx = 1
    for _ in 1:Sweeps
        G1.Lt .= G1.Ls[:, 1]
        G2.Lt .= G2.Ls[:, 1]
        G1.Rt .= G1.Rs[:, 1]
        G2.Rt .= G2.Rs[:, 1]

        for lt in 1:model.Nt
            G1.D .= @view model.exp_αη_pos[lt, view(ss[1], :, lt)]
            G2.D .= @view model.exp_αη_pos[lt, view(ss[2], :, lt)]

            # --- Propagate Rt ---
            WrapKV!(G1.tmpN, model.eK, model.eKinv, G1.D, G1.Rt, "Forward", "R")
            WrapKV!(G2.tmpN, model.eK, model.eKinv, G2.D, G2.Rt, "Forward", "R")

            G1.D .= 1 ./ G1.D
            G2.D .= 1 ./ G2.D

            # --- Propagate Lt ---
            WrapKV!(G1.tmpN, model.eK, model.eKinv, G1.D, G1.Lt, "Forward", "L")
            WrapKV!(G2.tmpN, model.eK, model.eKinv, G2.D, G2.Lt, "Forward", "L")


            UpdatePhyLayer!(rng, view(ss[1], :, lt), lt, G1, model, UPD)
            UpdatePhyLayer!(rng, view(ss[2], :, lt), lt, G2, model, UPD)


            # #############################################################################
            # # ========== VERIFICATION ==========
            # Ltt1, Rtt1 = LRt(lt, model, ss[1])
            # Ltt2, Rtt2 = LRt(lt, model, ss[2])
            # errrr = [norm(Ltt1 - G1.Lt / norm(G1.Lt)), norm(Rtt1 - G1.Rt / norm(G1.Rt)),
            #     norm(Ltt2 - G2.Lt / norm(G2.Lt)), norm(Rtt2 - G2.Rt / norm(G2.Rt))]
            # if !(maximum(errrr) < ERROR)
            #     error("lt=", lt, " Update error:  ", errrr)
            # end
            # #############################################################################

            # ---------------------------------------------------------------------------------------------------------
            if record && 0 <= (Θidx - idx) <= 1
                tmpO += EE_measure(model, lt, ss, G1, G2, indexA, indexAbar)
                counter += 1
            end
            # ---------------------------------------------------------------------------------------------------------

            # --- Node handling ---
            if any(model.nodes .== lt)
                idx += 1
                BM_F!(G1.tmpN, G1.tmpNN, G1.BM, model, ss[1], idx - 1)
                BM_F!(G2.tmpN, G2.tmpNN, G2.BM, model, ss[2], idx - 1)

                mul!(view(G1.Rs, :, idx), G1.BM, view(G1.Rs, :, idx - 1))
                mul!(view(G2.Rs, :, idx), G2.BM, view(G2.Rs, :, idx - 1))
                G1.Rs[:, idx] ./= norm(G1.Rs[:, idx])
                G2.Rs[:, idx] ./= norm(G2.Rs[:, idx])

                # #############################################################################
                # @assert norm(G1.Lt / norm(G1.Lt) - G1.Ls[:, idx]) < ERROR
                # @assert norm(G2.Lt / norm(G2.Lt) - G2.Ls[:, idx]) < ERROR
                # #############################################################################

                G1.Lt .= G1.Ls[:, idx]
                G2.Lt .= G2.Ls[:, idx]
                G1.Rt .= G1.Rs[:, idx]
                G2.Rt .= G2.Rs[:, idx]
            end
        end


        for lt in model.Nt:-1:1
            UpdatePhyLayer!(rng, view(ss[1], :, lt), lt, G1, model, UPD)
            UpdatePhyLayer!(rng, view(ss[2], :, lt), lt, G2, model, UPD)

            # #############################################################################
            # # ========== VERIFICATION ==========
            # Ltt1, Rtt1 = LRt(lt, model, ss[1])
            # Ltt2, Rtt2 = LRt(lt, model, ss[2])
            # errrr = [norm(Ltt1 - G1.Lt / norm(G1.Lt)), norm(Rtt1 - G1.Rt / norm(G1.Rt)),
            #     norm(Ltt2 - G2.Lt / norm(G2.Lt)), norm(Rtt2 - G2.Rt / norm(G2.Rt))]
            # if !(maximum(errrr) < ERROR)
            #     error("lt=", lt, " Update error:  ", errrr)
            # end
            # #############################################################################


            # ---------------------------------------------------------------------------------------------------------
            if record && 0 <= (Θidx - idx) <= 1
                tmpO += EE_measure(model, lt, ss, G1, G2, indexA, indexAbar)
                counter += 1
            end
            # ---------------------------------------------------------------------------------------------------------

            G1.D .= @view model.exp_αη_pos[lt, view(ss[1], :, lt)]
            G2.D .= @view model.exp_αη_pos[lt, view(ss[2], :, lt)]

            # --- Propagate Lt ---
            WrapKV!(G1.tmpN, model.eK, model.eKinv, G1.D, G1.Lt, "Backward", "L")
            WrapKV!(G2.tmpN, model.eK, model.eKinv, G2.D, G2.Lt, "Backward", "L")

            G1.D .= 1 ./ G1.D
            G2.D .= 1 ./ G2.D

            # --- Propagate Rt ---
            WrapKV!(G1.tmpN, model.eK, model.eKinv, G1.D, G1.Rt, "Backward", "R")
            WrapKV!(G2.tmpN, model.eK, model.eKinv, G2.D, G2.Rt, "Backward", "R")

            if any(model.nodes .== (lt - 1))
                idx -= 1
                BM_F!(G1.tmpN, G1.tmpNN, G1.BM, model, ss[1], idx)
                BM_F!(G2.tmpN, G2.tmpNN, G2.BM, model, ss[2], idx)

                mul!(view(G1.Ls, :, idx), G1.BM', view(G1.Ls, :, idx + 1))
                mul!(view(G2.Ls, :, idx), G2.BM', view(G2.Ls, :, idx + 1))
                G1.Ls[:, idx] ./= norm(G1.Ls[:, idx])
                G2.Ls[:, idx] ./= norm(G2.Ls[:, idx])

                # # #############################################################################
                # @assert norm(G1.Rt / norm(G1.Rt) - G1.Rs[:, idx]) < ERROR
                # @assert norm(G2.Rt / norm(G2.Rt) - G2.Rs[:, idx]) < ERROR
                # #############################################################################

                G1.Lt .= G1.Ls[:, idx]
                G2.Lt .= G2.Ls[:, idx]
                G1.Rt .= G1.Rs[:, idx]
                G2.Rt .= G2.Rs[:, idx]
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

    return ss
end

function EE_measure(model, lt, ss, G1, G2, indexA, indexAbar)
    # G1.L0 .= G1.Lt
    # G1.R0 .= G1.Rt
    # G2.L0 .= G2.Lt
    # G2.R0 .= G2.Rt

    # if lt > model.Nt / 2
    #     for t in lt:-1:div(model.Nt, 2)+1
    #         G1.D .= @view model.exp_αη_pos[lt, view(ss[1], :, t)]
    #         G2.D .= @view model.exp_αη_pos[lt, view(ss[2], :, t)]

    #         WrapKV!(G1.tmpN, model.eK, model.eKinv, G1.D, G1.L0, "Backward", "L")
    #         WrapKV!(G2.tmpN, model.eK, model.eKinv, G2.D, G2.L0, "Backward", "L")

    #         G1.D .= 1 ./ G1.D
    #         G2.D .= 1 ./ G2.D

    #         WrapKV!(G1.tmpN, model.eK, model.eKinv, G1.D, G1.R0, "Backward", "R")
    #         WrapKV!(G2.tmpN, model.eK, model.eKinv, G2.D, G2.R0, "Backward", "R")
    #     end
    # else
    #     for t in lt+1:div(model.Nt, 2)
    #         G1.D .= @view model.exp_αη_pos[lt, view(ss[1], :, t)]
    #         G2.D .= @view model.exp_αη_pos[lt, view(ss[2], :, t)]

    #         WrapKV!(G1.tmpN, model.eK, model.eKinv, G1.D, G1.R0, "Forward", "R")
    #         WrapKV!(G2.tmpN, model.eK, model.eKinv, G2.D, G2.R0, "Forward", "R")

    #         G1.D .= 1 ./ G1.D
    #         G2.D .= 1 ./ G2.D

    #         WrapKV!(G1.tmpN, model.eK, model.eKinv, G1.D, G1.L0, "Forward", "L")
    #         WrapKV!(G2.tmpN, model.eK, model.eKinv, G2.D, G2.L0, "Forward", "L")
    #     end
    # end
    # G1.L0 ./= norm(G1.L0)
    # G1.R0 ./= norm(G1.R0)
    # G2.L0 ./= norm(G2.L0)
    # G2.R0 ./= norm(G2.R0)
    WrapLR0!(model, lt, ss[1], G1)
    WrapLR0!(model, lt, ss[2], G2)


    # #############################################################################
    L001, R001 = LRt(div(model.Nt, 2), model, ss[1])
    L002, R002 = LRt(div(model.Nt, 2), model, ss[2])
    errrr = [norm(L001 - G1.L0), norm(R001 - G1.R0),
        norm(L002 - G2.L0), norm(R002 - G2.R0)]
    if !(maximum(errrr) < 1e-6)
        error("lt=", lt, " EE EE_measure Update error:  ", errrr)
    end
    # #############################################################################

    return EE_cal(model.binoms_sq, G1.L0, G2.L0, G1.R0, G2.R0, indexA, indexAbar)
end

