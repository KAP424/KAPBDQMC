function phy_update(path::String, model, s::Array{UInt8,2}, Sweeps::Int64, record::Bool=false)
    ERROR = 1e-6
    name = name_Lattice(model.Lattice)
    global LOCK = ReentrantLock()
    if model.Θquench == 0.0
        file = "$(path)/minusUphy$(name)_t$(model.Ht)U$(model.Hu1)size$(model.site)Δt$(model.Δt)Θ$(model.Θrelax)BS$(model.BatchSize).csv"
    else
        file = "$(path)/minusUphy$(name)_t$(model.Ht)U$(model.Hu1)_$(model.Hu2)size$(model.site)Δt$(model.Δt)Θ$(model.Θrelax)_$(model.Θquench)BS$(model.BatchSize).csv"
    end

    rng = MersenneTwister(Threads.threadid() + time_ns())
    TTT = time_ns()
    Ns = model.Ns
    NN = length(model.nodes)
    Θidx = div(NN, 2) + 1

    UPD = UpdateBuffer()
    G = PhyBuffer(Ns, NN)

    Initial_PhyBuffer!(G, model, s)

    Ek = Eu = CDW0 = CDW1 = SDW0 = SDW1 = 0.0
    counter = 0

    idx = 1
    for _ in 1:Sweeps
        G.Lt .= G.Ls[:, 1]
        G.Rt .= G.Rs[:, 1]

        for lt in 1:model.Nt
            G.D .= @view model.exp_αη_pos[lt, view(s, :, lt)]

            WrapKV!(G.tmpN, model.eK, model.eKinv, G.D, G.Rt, "Forward", "R")
            G.D .= 1 ./ G.D
            WrapKV!(G.tmpN, model.eK, model.eKinv, G.D, G.Lt, "Forward", "L")


            UpdatePhyLayer!(rng, view(s, :, lt), lt, G, model, UPD)

            # #############################################################################
            # ========== VERIFICATION ==========
            Ltt, Rtt = LRt(lt, model, s)
            errrr = [norm(Ltt - G.Lt / norm(G.Lt)), norm(Rtt - G.Rt / norm(G.Rt))]
            if !(maximum(errrr) < ERROR)
                error("lt=", lt, " Update error:  ", errrr)
            end
            # #############################################################################


            # ---------------------------------------------------------------------------------------------------------
            # record physical quantities
            if record && 0 <= (Θidx - idx) <= 1
                tmp = phy_measure(model, lt, s, G, tmpNN, tmpN)
                counter += 1
                Ek += tmp[1]
                Eu += tmp[2]
                CDW0 += tmp[3]
                CDW1 += tmp[4]
                SDW0 += tmp[5]
                SDW1 += tmp[6]
            end
            # ---------------------------------------------------------------------------------------------------------

            if any(model.nodes .== lt)
                idx += 1
                BM_F!(G.tmpN, G.tmpNN, G.BM, model, s, idx - 1)
                mul!(view(G.Rs, :, idx), G.BM, view(G.Rs, :, idx - 1))
                G.Rs[:, idx] ./= norm(G.Rs[:, idx])

                # #############################################################################
                @assert norm(G.Lt / norm(G.Lt) - G.Ls[:, idx]) < ERROR
                # #############################################################################

                G.Lt .= G.Ls[:, idx]
                G.Rt .= G.Rs[:, idx]
            end
        end

        for lt in model.Nt:-1:1
            UpdatePhyLayer!(rng, view(s, :, lt), lt, G, model, UPD)

            # #############################################################################
            # ========== VERIFICATION ==========
            Ltt, Rtt = LRt(lt, model, s)
            errrr = [norm(Ltt - G.Lt / norm(G.Lt)), norm(Rtt - G.Rt / norm(G.Rt))]
            if !(maximum(errrr) < ERROR)
                error("lt=", lt, " Update error:  ", errrr)
            end
            # #############################################################################


            # ---------------------------------------------------------------------------------------------------------
            # record physical quantities
            if record && 0 <= (idx - Θidx) <= 1
                tmp = phy_measure(model, lt, s, G, tmpNN, tmpN)
                counter += 1
                Ek += tmp[1]
                Eu += tmp[2]
                CDW0 += tmp[3]
                CDW1 += tmp[4]
                SDW0 += tmp[5]
                SDW1 += tmp[6]
            end
            # ---------------------------------------------------------------------------------------------------------

            G.D .= @view model.exp_αη_pos[lt, view(s, :, lt)]
            WrapKV!(G.tmpN, model.eK, model.eKinv, G.D, G.Lt, "Backward", "L")
            G.D .= 1 ./ G.D
            WrapKV!(G.tmpN, model.eK, model.eKinv, G.D, G.Rt, "Backward", "R")

            if any(model.nodes .== (lt - 1))
                idx -= 1
                BM_F!(G.tmpN, G.tmpNN, G.BM, model, s, idx)
                mul!(view(G.Ls, :, idx), G.BM', view(G.Ls, :, idx + 1))

                G.Ls[:, idx] ./= norm(G.Ls[:, idx])

                # #############################################################################
                @assert norm(G.Rt / norm(G.Rt) - G.Rs[:, idx]) < ERROR
                # #############################################################################

                G.Lt .= G.Ls[:, idx]
                G.Rt .= G.Rs[:, idx]
            end
        end
        if record
            lock(LOCK) do
                open(file, "a") do io
                    writedlm(io, [Ek Eu CDW0 CDW1 SDW0 SDW1] ./ counter, ',')
                end
            end
            Ek = Eu = CDW0 = CDW1 = SDW0 = SDW1 = 0.0
            counter = 0
        end
    end
    if record
        TTT = round(Int, (time_ns() - TTT) / 1e9)
        hour = TTT ÷ 3600
        minite = (TTT % 3600) ÷ 60
        second = TTT % 60
        println("      acc = ", round(100 * UPD.acc / prod(size(s)) / Sweeps / 2, digits=2), "%", "  $(Sweeps) Sweep finished in ", string(lpad(string(hour), 2, '0'), ":", lpad(string(minite), 2, '0'), ":", lpad(string(second), 2, '0')))
    end
    return s
end


function phy_measure(model, lt, s, G)
    WrapLR0!(model, lt, s, G)
    LR = dot(G.L0, G.R0)
    Ek = 0
    ij = [Tuple(idx) for idx in findall(!iszero, model.K)]
    for (i, j) in ij
        Ek += G.L0[i] * G.R0[j]
    end
    Ek *= 2 * modelNb * (modelNb - 1) / LR^2
     
end



