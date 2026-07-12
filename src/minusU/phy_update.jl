function phy_update(path::String, model, s::Array{UInt8,2}, Sweeps::Int64, record::Bool=false)
    global LOCK = ReentrantLock()
    TTT = time_ns()
    ERROR = 1e-6

    NN = length(model.nodes)
    Θidx = div(NN, 2) + 1
    name = name_Lattice(model.Lattice)

    UPD = UpdateBuffer()
    Phy = PhyBuffer(model.Ns, NN)
    PLs, PRs, BM, F, tmpN, tmpNN = Phy.PLs, Phy.PRs, Phy.BM, Phy.F, Phy.tmpN, Phy.tmpNN


    if model.Θquench == 0.0
        file = "$(path)/minusUphy$(name)_t$(model.Ht)U$(model.Hu1)size$(model.site)Δt$(model.Δt)Θ$(model.Θrelax)BS$(model.BatchSize).csv"
    else
        file = "$(path)/minusUphy$(name)_t$(model.Ht)U$(model.Hu1)_$(model.Hu2)size$(model.site)Δt$(model.Δt)Θ$(model.Θrelax)_$(model.Θquench)BS$(model.BatchSize).csv"
    end

    Ns = model.Ns
    ns = div(Ns, 2)
    rng = MersenneTwister(Threads.threadid() + time_ns())

    PRs[:, 1] .= model.HalfeKinv * model.Pt
    PLs[:, NN] .= model.HalfeK * model.Pt

    for idx in NN-1:-1:1
        BM_F!(tmpN, tmpNN, BM, model, s, idx)
        mul!(view(PLs, :, idx), BM', view(PLs, :, idx + 1))
        PLs[:, idx] ./= norm(PLs[:, idx])
    end

    idx = 1
    get_F!(model.Nb, view(PLs, :, idx), view(PRs, :, idx), F)
    for _ in 1:Sweeps
        for lt in 1:model.Nt
            @inbounds @simd for iii in 1:Ns
                tmpN[iii] = @fastmath model.exp_αη_pos[lt, s[iii, lt]]
            end
            WrapKV!(tmpNN, model.eK, model.eKinv, tmpN, F, "Forward", "B")

            UpdatePhyLayer!(rng, view(s, :, lt), lt, model, UPD, Phy)
            # ---------------------------------------------------------------------------------------------------------
            # record physical quantities
            # if record && 0 <= (Θidx - idx) <= 1
            #     tmp = phy_measure(model, lt, s, G, tmpNN, tmpN)
            #     counter += 1
            #     Ek += tmp[1]
            #     Eu += tmp[2]
            #     CDW0 += tmp[3]
            #     CDW1 += tmp[4]
            #     SDW0 += tmp[5]
            #     SDW1 += tmp[6]
            # end
            # ---------------------------------------------------------------------------------------------------------

            if any(model.nodes .== lt)
                println(idx)
                BM_F!(tmpN, tmpNN, BM, model, s, idx)
                idx += 1
                mul!(view(PRs, :, idx), BM, view(PRs, :, idx - 1))

                copyto!(tmpNN, F)

                get_F!(model.Nb, view(PLs, :, idx), view(PRs, :, idx), F)
                #####################################################################
                # axpy!(-1.0, G, tmpNN)
                # if norm(tmpNN) > ERROR
                #     println("Warning for Batchsize Wrap Error : $(norm(tmpNN))")
                # end
                #####################################################################
            end
        end

        for lt in model.Nt:-1:1
            #####################################################################
            # # print("-")
            # if norm(G - Gτ(model, s, lt)) > ERROR
            #     error(lt, " Wrap error:  ", norm(G - Gτ(model, s, lt)))
            # end
            #####################################################################

            UpdatePhyLayer!(rng, view(s, :, lt), lt, model, UPD, Phy)
            # ---------------------------------------------------------------------------------------------------------
            # record physical quantities
            # if record && 0 <= (idx - Θidx) <= 1
            #     tmp = phy_measure(model, lt, s, G, tmpNN, tmpN)
            #     counter += 1
            #     Ek += tmp[1]
            #     Eu += tmp[2]
            #     CDW0 += tmp[3]
            #     CDW1 += tmp[4]
            #     SDW0 += tmp[5]
            #     SDW1 += tmp[6]
            # end
            # ---------------------------------------------------------------------------------------------------------
            @inbounds @simd for iii in 1:Ns
                tmpN[iii] = @fastmath model.exp_αη_neg[lt, s[iii, lt]]
            end
            WrapKV!(tmpNN, model.eK, model.eKinv, tmpN, F, "Backward", "B")

            if any(model.nodes .== (lt - 1))
                # println("idx=",idx," lt=",lt-1)
                idx -= 1
                BM_F!(tmpN, tmpNN, BM, model, s, idx)
                mul!(view(PLs, :, idx), BM', view(PLs, :, idx + 1))
                get_F!(model.Nb, view(PLs, :, idx), view(PRs, :, idx), F)

            end
        end

    end

end

function UpdatePhyLayer!(rng, s, lt, model, UPD, Phy::PhyBuffer_)
    for i in eachindex(s)
        sx = rand(rng, model.samplers_vec[s[i]])
        UPD.Δ = exp(model.αη[lt, sx] - model.αη[lt, s[i]]) - 1
        UPD.r = 1 + UPD.Δ / model.Ns * Phy.F[i, i]
        p = UPD.r * model.γ[sx] / model.γ[s[i]]
        if rand(rng) < p
            UPD.acc += 1
            Phy.F .*= (1 + UPD.Δ) / UPD.r
            # Gupdate!(Phy, UPD)
            s[i] = sx
        end
    end
end
