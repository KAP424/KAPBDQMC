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

    # #############################################################################
    # @assert norm(model.eK - model.eK') < 1e-10
    # for idx in 1:NN
    #     Ltt1, Rtt1 = LRt(model.nodes[idx], model, ss[1])
    #     Ltt2, Rtt2 = LRt(model.nodes[idx], model, ss[2])
    #     errrr = [norm(Ltt1 - G1.Ls[:, idx]), norm(Rtt1 - G1.Rs[:, idx]),
    #         norm(Ltt2 - G2.Ls[:, idx]), norm(Rtt2 - G2.Rs[:, idx])]
    #     println("idx=$(idx)  errrr=$((errrr))")
    # end
    # #############################################################################


    G1.Bt0s[:, :, Θidx] .= I(Ns)
    G2.Bt0s[:, :, Θidx] .= I(Ns)
    for idx in 1:div(NN, 2)
        # G1.Bt0s[:, :, Θidx-idx] = G1.Bt0s[:, :, Θidx-idx+1] * G1.BMs[:, :, Θidx-idx]
        # G2.Bt0s[:, :, Θidx-idx] = G2.Bt0s[:, :, Θidx-idx+1] * G2.BMs[:, :, Θidx-idx]
        mul!(view(G1.Bt0s, :, :, Θidx - idx), view(G1.Bt0s, :, :, Θidx - idx + 1), view(G1.BMs, :, :, Θidx - idx))
        mul!(view(G2.Bt0s, :, :, Θidx - idx), view(G2.Bt0s, :, :, Θidx - idx + 1), view(G2.BMs, :, :, Θidx - idx))
    end


    idx = 1
    for loop in 1:Sweeps
        G1.Lt .= G1.Ls[:, 1]
        G2.Lt .= G2.Ls[:, 1]
        G1.Rt .= G1.Rs[:, 1]
        G2.Rt .= G2.Rs[:, 1]

        G1.L0 .= G1.Ls[:, Θidx]
        G1.R0 .= G1.Rs[:, Θidx]
        G2.L0 .= G2.Ls[:, Θidx]
        G2.R0 .= G2.Rs[:, Θidx]

        for lt in 1:model.Nt
            D .= @view model.exp_αη_pos[lt, view(ss[1], :, lt)]
            D_ .= @view model.exp_αη_pos[lt, view(ss[2], :, lt)]

            WrapKV!(tmpN, model.eK, model.eKinv, D, G1.Rt, "Forward", "R")
            WrapKV!(tmpN, model.eK, model.eKinv, D_, G2.Rt, "Forward", "R")
            WrapKV!(tmpN, model.eK, model.eKinv, D, G1.Lt, "Forward", "L")
            WrapKV!(tmpN, model.eK, model.eKinv, D_, G2.Lt, "Forward", "L")

            UpdateEELayer!(lt < lθ, rng, view(ss[1], :, lt), view(ss[2], :, lt), lt, G1, G2, model, UPD)

            ##############################################################
            # ERROR = 1e-5
            # Ltt1, Rtt1 = LRt(lt, model, ss[1])
            # Ltt2, Rtt2 = LRt(lt, model, ss[2])
            # L001, R001 = LRt(lθ, model, ss[1])
            # L002, R002 = LRt(lθ, model, ss[2])
            # errrr = [norm(Ltt1 - G1.Lt / norm(G1.Lt)), norm(Rtt1 - G1.Rt / norm(G1.Rt)),
            #     norm(Ltt2 - G2.Lt / norm(G2.Lt)), norm(Rtt2 - G2.Rt / norm(G2.Rt)),
            #     norm(L001 - G1.L0 / norm(G1.L0)), norm(R001 - G1.R0 / norm(G1.R0)),
            #     norm(L002 - G2.L0 / norm(G2.L0)), norm(R002 - G2.R0 / norm(G2.R0))]
            # # println("lt=$(lt)  errrr=$((errrr))")
            # if sum(errrr) > ERROR
            #     println(lt, " Update error:  ", sum(errrr[1:2:end]), "  ", sum(errrr[2:2:end]))
            # end
            ##############################################################

            tmpO += EE_cal(model.binoms_sq, G1.L0, G2.L0, G1.R0, G2.R0, indexA, indexAbar)
            counter += 1

            if any(model.nodes .== lt)
                idx += 1
                BM_F!(tmpN, tmpNN, view(G1.BMs, :, :, idx - 1), model, ss[1], idx - 1)
                BM_F!(tmpN, tmpNN, view(G2.BMs, :, :, idx - 1), model, ss[2], idx - 1)

                # LRs[]=...
                mul!(view(G1.Rs, :, idx), view(G1.BMs, :, :, idx - 1), view(G1.Rs, :, idx - 1))
                mul!(view(G2.Rs, :, idx), view(G2.BMs, :, :, idx - 1), view(G2.Rs, :, idx - 1))
                # G1.Rs[:, idx] .= G1.BMs[:, :, idx-1] * G1.Rs[:, idx-1]
                # G2.Rs[:, idx] .= G2.BMs[:, :, idx-1] * G2.Rs[:, idx-1]

                G1.Lt .= G1.Ls[:, idx]
                G1.Rt .= G1.Rs[:, idx]
                G2.Lt .= G2.Ls[:, idx]
                G2.Rt .= G2.Rs[:, idx]

                # 过半后开始计算 Bt0s
                if idx == Θidx
                    # recalculate Bt0s befor Θidx_idx 
                    for i in Θidx-1:-1:1
                        mul!(view(G1.Bt0s, :, :, i), view(G1.Bt0s, :, :, i + 1), view(G1.BMs, :, :, i))
                        mul!(view(G2.Bt0s, :, :, i), view(G2.Bt0s, :, :, i + 1), view(G2.BMs, :, :, i))
                    end
                elseif idx > Θidx
                    # update Bt0s after Θidx_idx
                    mul!(view(G1.Bt0s, :, :, idx), view(G1.BMs, :, :, idx - 1), view(G1.Bt0s, :, :, idx - 1))
                    mul!(view(G2.Bt0s, :, :, idx), view(G2.BMs, :, :, idx - 1), view(G2.Bt0s, :, :, idx - 1))
                end
            end
        end

        for lt in model.Nt:-1:1
            UpdateEELayer!(lt < lθ, rng, view(ss[1], :, lt), view(ss[2], :, lt), lt, G1, G2, model, UPD)

            ##############################################################
            # ERROR = 1e-5
            # Ltt1, Rtt1 = LRt(lt, model, ss[1])
            # Ltt2, Rtt2 = LRt(lt, model, ss[2])
            # L001, R001 = LRt(lθ, model, ss[1])
            # L002, R002 = LRt(lθ, model, ss[2])
            # errrr = [norm(Ltt1 - G1.Lt / norm(G1.Lt)), norm(Rtt1 - G1.Rt / norm(G1.Rt)),
            #     norm(Ltt2 - G2.Lt / norm(G2.Lt)), norm(Rtt2 - G2.Rt / norm(G2.Rt)),
            #     norm(L001 - G1.L0 / norm(G1.L0)), norm(R001 - G1.R0 / norm(G1.R0)),
            #     norm(L002 - G2.L0 / norm(G2.L0)), norm(R002 - G2.R0 / norm(G2.R0))]
            # # println("lt=$(lt)  errrr=$((errrr))")
            # if sum(errrr) > ERROR
            #     println(lt, " Update error:  ", sum(errrr[1:2:end]), "  ", sum(errrr[2:2:end]))
            # end
            ##############################################################

            tmpO += EE_cal(model.binoms_sq, G1.L0, G2.L0, G1.R0, G2.R0, indexA, indexAbar)
            counter += 1

            if any(model.nodes .== (lt - 1))
                idx -= 1
                BM_F!(tmpN, tmpNN, view(G1.BMs, :, :, idx), model, ss[1], idx)
                BM_F!(tmpN, tmpNN, view(G2.BMs, :, :, idx), model, ss[2], idx)

                mul!(view(G1.Ls, :, idx), view(G1.BMs, :, :, idx)', view(G1.Ls, :, idx + 1))
                mul!(view(G2.Ls, :, idx), view(G2.BMs, :, :, idx)', view(G2.Ls, :, idx + 1))

                G1.Lt .= G1.Ls[:, idx]
                G1.Rt .= G1.Rs[:, idx]
                G2.Lt .= G2.Ls[:, idx]
                G2.Rt .= G2.Rs[:, idx]

                # 过半后开始计算 Bt0s
                if idx == Θidx
                    # recalculate Bt0s after Θidx_idx 
                    for i in Θidx+1:NN
                        mul!(view(G1.Bt0s, :, :, i), view(G1.BMs, :, :, i - 1), view(G1.Bt0s, :, :, i - 1))
                        mul!(view(G2.Bt0s, :, :, i), view(G2.BMs, :, :, i - 1), view(G2.Bt0s, :, :, i - 1))
                        # G1.Bt0s[:, :, i] = G1.BMs[:, :, i-1] * G1.Bt0s[:, :, i-1]
                        # G2.Bt0s[:, :, i] = G2.BMs[:, :, i-1] * G2.Bt0s[:, :, i-1]
                    end
                elseif idx < Θidx
                    # update Bt0s before Θidx_idx
                    mul!(view(G1.Bt0s, :, :, idx), view(G1.Bt0s, :, :, idx + 1), view(G1.BMs, :, :, idx))
                    mul!(view(G2.Bt0s, :, :, idx), view(G2.Bt0s, :, :, idx + 1), view(G2.BMs, :, :, idx))
                    # G1.Bt0s[:, :, idx] = G1.Bt0s[:, :, idx+1] * G1.BMs[:, :, idx]
                    # G2.Bt0s[:, :, idx] = G2.Bt0s[:, :, idx+1] * G2.BMs[:, :, idx]
                end
            else
                D .= @view model.exp_αη_pos[lt, view(ss[1], :, lt)]
                D_ .= @view model.exp_αη_pos[lt, view(ss[2], :, lt)]

                WrapKV!(tmpN, model.eK, model.eKinv, D, G1.Lt, "Backward", "L")
                WrapKV!(tmpN, model.eK, model.eKinv, D_, G2.Lt, "Backward", "L")
                WrapKV!(tmpN, model.eK, model.eKinv, D, G1.Rt, "Backward", "R")
                WrapKV!(tmpN, model.eK, model.eKinv, D_, G2.Rt, "Backward", "R")

            end

        end

        if record
            lock(LOCK) do
                open(file, "a") do io
                    writedlm(io, tmpO / counter, ',')
                end
            end
        end
        # 输出 O
        tmpO = 0.0
        counter = 0
    end
    TTT = round(Int, (time_ns() - TTT) / 1e9)
    hour = TTT ÷ 3600
    minite = (TTT % 3600) ÷ 60
    second = TTT % 60
    println("      acc = ", round(100 * UPD.acc / prod(size(ss[1])) / Sweeps / 4, digits=2), "%", "  $(Sweeps) Sweep finished in ", string(lpad(string(hour), 2, '0'), ":", lpad(string(minite), 2, '0'), ":", lpad(string(second), 2, '0')))
end

function UpdateEELayer!(tless0, rng, s1, s2, lt, G1, G2, model, UPD)
    LR1 = dot(G1.Lt, G1.Rt)
    LR2 = dot(G2.Lt, G2.Rt)

    for i in axes(s1, 1)
        begin
            sx = rand(rng, model.samplers_vec[s1[i]])
            UPD.Δ = exp(model.αη[lt, sx] - model.αη[lt, s1[i]]) - 1
            UPD.r = abs2(1 + G1.Lt[i] * UPD.Δ * G1.Rt[i] / dot(G1.Lt, G1.Rt))^model.Nb
            p = UPD.r * model.γ[sx] / model.γ[s1[i]]

            # if p < 0
            #     println("warnin for negative p:  ", p)
            # end
            if rand(rng) < p
                s1[i] = sx
                UPD.acc += 1
                G1.Rt[i] += UPD.Δ * G1.Rt[i]
                LR1 = dot(G1.Lt, G1.Rt)
                # update L0 R0
                if tless0
                    G1.R0 .+= UPD.Δ .* G1.Rt[i] .* G1.Bt0[:, i]
                else
                    G1.L0 .+= UPD.Δ .* G1.Lt[i] .* conj.(G1.Bt0[i, :])
                end
            end
        end

        begin
            sx = rand(rng, model.samplers_vec[s2[i]])
            UPD.Δ = exp(model.αη[lt, sx] - model.αη[lt, s2[i]]) - 1
            UPD.r = abs2(1 + G2.Lt[i] * UPD.Δ * G2.Rt[i] / dot(G2.Lt, G2.Rt))^model.Nb
            p = UPD.r * model.γ[sx] / model.γ[s2[i]]
            # if p < 0
            #     println("warnin for negative p:  ", p)
            # end
            if rand(rng) < p
                s2[i] = sx
                UPD.acc += 1
                G2.Rt[i] += UPD.Δ * G2.Rt[i]
                # update L0 R0

                if tless0
                    G2.R0 .+= UPD.Δ .* G2.Rt[i] .* G2.Bt0[:, i]
                else
                    G2.L0 .+= UPD.Δ .* G2.Lt[i] .* conj.(G2.Bt0[i, :])
                end
            end
        end
    end
end


# function ctrl_EEicr(path::String, model, indexA::Vector{Int64}, indexB::Vector{Int64}, Sweeps::Int64, λ::Float64, Nλ::Int64, ss::Vector{Matrix{UInt8}}, record)
#     ERROR = 1e-6
#     global LOCK = ReentrantLock()
#     TTT = time_ns()

#     Ns = model.Ns
#     ns = div(Ns, 2)
#     NN = length(model.nodes)
#     Θidx = div(NN, 2) + 1

#     UPD = UpdateBuffer()
#     SCEE = SCEEBuffer(model.Ns)
#     A = AreaBuffer(indexA)
#     B = AreaBuffer(indexB)
#     G1 = G4Buffer(model.Ns, NN)
#     G2 = G4Buffer(model.Ns, NN)

#     name = name_Lattice(model.Lattice)

#     if model.Θquench == 0.0
#         file = "$(path)/tUSCEE$(name)_t$(model.Ht)U$(model.Hu1)size$(model.site)Δt$(model.Δt)Θ$(model.Θrelax)N$(Nλ)BS$(model.BatchSize).csv"
#     else
#         file = "$(path)/tUSCEE$(name)_t$(model.Ht)U$(model.Hu1)_$(model.Hu2)size$(model.site)Δt$(model.Δt)Θ$(model.Θrelax)_$(model.Θquench)N$(Nλ)BS$(model.BatchSize).csv"
#     end

#     rng = MersenneTwister(Threads.threadid() + time_ns())


#     Gt1, Gt01, G0t1, BLMs1, BRMs1, BMs1, BMsinv1 =
#         G1.Gt, G1.Gt0, G1.G0t, G1.BLMs, G1.BRMs, G1.BMs, G1.BMinvs
#     Gt2, Gt02, G0t2, BLMs2, BRMs2, BMs2, BMsinv2 =
#         G2.Gt, G2.Gt0, G2.G0t, G2.BLMs, G2.BRMs, G2.BMs, G2.BMinvs

#     tmpN, tmpN_, tmpNN, tmpNn, tmpnN, tau = SCEE.N, SCEE.N_, SCEE.NN, SCEE.Nn, SCEE.nN, SCEE.tau


#     tmpO = 0.0
#     counter = 0
#     O = zeros(Float64, Sweeps + 1)
#     O[1] = λ

#     for idx in 1:NN-1
#         BM_F!(tmpN, tmpNN, view(BMs1, :, :, idx), model, ss[1], idx)
#         BM_F!(tmpN, tmpNN, view(BMs2, :, :, idx), model, ss[2], idx)
#         BMinv_F!(tmpN, tmpNN, view(BMsinv1, :, :, idx), model, ss[1], idx)
#         BMinv_F!(tmpN, tmpNN, view(BMsinv2, :, :, idx), model, ss[2], idx)
#     end

#     BLMs1[:, :, NN] .= model.Pt'
#     BRMs1[:, :, 1] .= model.Pt

#     BLMs2[:, :, NN] .= model.Pt'
#     BRMs2[:, :, 1] .= model.Pt

#     for i in 1:NN-1
#         mul!(tmpnN, view(BLMs1, :, :, NN - i + 1), view(BMs1, :, :, NN - i))
#         LAPACK.gerqf!(tmpnN, tau)
#         LAPACK.orgrq!(tmpnN, tau, ns)
#         copyto!(view(BLMs1, :, :, NN - i), tmpnN)

#         mul!(tmpNn, view(BMs1, :, :, i), view(BRMs1, :, :, i))
#         LAPACK.geqrf!(tmpNn, tau)
#         LAPACK.orgqr!(tmpNn, tau, ns)
#         copyto!(view(BRMs1, :, :, i + 1), tmpNn)
#         # ---------------------------------------------------------------
#         mul!(tmpnN, view(BLMs2, :, :, NN - i + 1), view(BMs2, :, :, NN - i))
#         LAPACK.gerqf!(tmpnN, tau)
#         LAPACK.orgrq!(tmpnN, tau, ns)
#         copyto!(view(BLMs2, :, :, NN - i), tmpnN)

#         mul!(tmpNn, view(BMs2, :, :, i), view(BRMs2, :, :, i))
#         LAPACK.geqrf!(tmpNn, tau)
#         LAPACK.orgqr!(tmpNn, tau, ns)
#         copyto!(view(BRMs2, :, :, i + 1), tmpNn)

#     end

#     idx = 1
#     get_ABGM!(G1, G2, A, B, SCEE, model.nodes, idx, "Forward")
#     for loop in 1:Sweeps
#         # println("\n ====== Sweep $loop / $Sweeps ======")
#         for lt in 1:model.Nt
#             @inbounds @simd for iii in 1:Ns
#                 @fastmath tmpN[iii] = model.exp_αη_pos[lt, ss[1][iii, lt]]
#                 @fastmath tmpN_[iii] = model.exp_αη_pos[lt, ss[2][iii, lt]]
#             end

#             WrapKV!(tmpN, model.eK, model.eKinv, tmpN, Gt01, "Forward", "L")
#             WrapKV!(tmpN, model.eK, model.eKinv, tmpN_, Gt02, "Forward", "L")
#             WrapKV!(tmpN, model.eK, model.eKinv, tmpN, Gt1, "Forward", "B")
#             WrapKV!(tmpN, model.eK, model.eKinv, tmpN_, Gt2, "Forward", "B")
#             WrapKV!(tmpN, model.eK, model.eKinv, tmpN, G0t1, "Forward", "R")
#             WrapKV!(tmpN, model.eK, model.eKinv, tmpN_, G0t2, "Forward", "R")

#             #####################################################################
#             # Gt1_, G01_, Gt01_, G0t1_ = G4(model, ss[1], lt, div(model.Nt, 2))
#             # Gt2_, G02_, Gt02_, G0t2_ = G4(model, ss[2], lt, div(model.Nt, 2))
#             # Gtt = zeros(ComplexF64, Ns, Ns)
#             # get_G!(SCEE.nn, tmpnN, SCEE.ipiv, view(BLMs1, :, :, idx), view(BRMs1, :, :, idx), Gtt)
#             # WrapKV!(tmpN, model.eK, model.eKinv, tmpN, Gtt, "Forward", "B")
#             # # println(norm(Gtt - Gt1_))
#             # # # Gτ 和 G4 一致
#             # # Gttt = Gτ(model, ss[1], lt)
#             # # println(norm(Gttt - Gt1_))
#             # if norm(Gt1 - Gt1_) + norm(Gt2 - Gt2_) + norm(Gt01 - Gt01_) + norm(Gt02 - Gt02_) + norm(G0t1 - G0t1_) + norm(G0t2 - G0t2_) > ERROR
#             #     println(norm(Gt1 - Gt1_), '\n', norm(Gt2 - Gt2_), '\n', norm(Gt01 - Gt01_), '\n', norm(Gt02 - Gt02_), '\n', norm(G0t1 - G0t1_), '\n', norm(G0t2 - G0t2_))
#             #     error("WrapTime=$lt ")
#             # end
#             # GM_A_ = GroverMatrix(G01_[indexA[:], indexA[:]], G02_[indexA[:], indexA[:]])
#             # gmInv_A_ = inv(GM_A_)
#             # GM_B_ = GroverMatrix(G01_[indexB[:], indexB[:]], G02_[indexB[:], indexB[:]])
#             # gmInv_B_ = inv(GM_B_)
#             # detg_A_ = abs2(det(GM_A_))
#             # detg_B_ = abs2(det(GM_B_))
#             # if norm(gmInv_A_ - A.gmInv) + norm(B.gmInv - gmInv_B_) + abs(A.detg - detg_A_) + abs(B.detg - detg_B_) > ERROR
#             #     println(norm(gmInv_A_ - A.gmInv), " ", norm(B.gmInv - gmInv_B_), " ", abs(A.detg - detg_A_), " ", abs(B.detg - detg_B_))
#             #     error("s2:  $lt : WrapTime")
#             # end
#             #####################################################################

#             UpdateSCEELayer!(rng, view(ss[1], :, lt), view(ss[2], :, lt), lt, G1, G2, A, B, model, UPD, SCEE, λ)

#             ##------------------------------------------------------------------------
#             tmpO += (A.detg / B.detg)^(1 / Nλ)
#             counter += 1
#             ##------------------------------------------------------------------------

#             if any(model.nodes .== lt)
#                 idx += 1
#                 BM_F!(tmpN, tmpNN, view(BMs1, :, :, idx - 1), model, ss[1], idx - 1)
#                 BMinv_F!(tmpN, tmpNN, view(BMsinv1, :, :, idx - 1), model, ss[1], idx - 1)
#                 BM_F!(tmpN, tmpNN, view(BMs2, :, :, idx - 1), model, ss[2], idx - 1)
#                 BMinv_F!(tmpN, tmpNN, view(BMsinv2, :, :, idx - 1), model, ss[2], idx - 1)
#                 for i in idx:max(Θidx, idx)
#                     # println("update BR i=",i)
#                     mul!(tmpNn, view(BMs1, :, :, i - 1), view(BRMs1, :, :, i - 1))
#                     LAPACK.geqrf!(tmpNn, tau)
#                     LAPACK.orgqr!(tmpNn, tau, ns)
#                     copyto!(view(BRMs1, :, :, i), tmpNn)
#                     # ---------------------------------------------------------------
#                     mul!(tmpNn, view(BMs2, :, :, i - 1), view(BRMs2, :, :, i - 1))
#                     LAPACK.geqrf!(tmpNn, tau)
#                     LAPACK.orgqr!(tmpNn, tau, ns)
#                     copyto!(view(BRMs2, :, :, i), tmpNn)
#                 end

#                 for i in idx-1:-1:min(Θidx, idx)
#                     # println("update BL i=",i)
#                     mul!(tmpnN, view(BLMs1, :, :, i + 1), view(BMs1, :, :, i))
#                     LAPACK.gerqf!(tmpnN, tau)
#                     LAPACK.orgrq!(tmpnN, tau, ns)
#                     copyto!(view(BLMs1, :, :, i), tmpnN)
#                     # ---------------------------------------------------------------
#                     mul!(tmpnN, view(BLMs2, :, :, i + 1), view(BMs2, :, :, i))
#                     LAPACK.gerqf!(tmpnN, tau)
#                     LAPACK.orgrq!(tmpnN, tau, ns)
#                     copyto!(view(BLMs2, :, :, i), tmpnN)
#                 end
#                 get_ABGM!(G1, G2, A, B, SCEE, model.nodes, idx, "Forward")
#             end

#         end

#         # println("\n ----------------reverse update ----------------")

#         for lt in model.Nt:-1:1

#             #####################################################################
#             # Gt1_, G01_, Gt01_, G0t1_ = G4(model, ss[1], lt, div(model.Nt, 2))
#             # Gt2_, G02_, Gt02_, G0t2_ = G4(model, ss[2], lt, div(model.Nt, 2))
#             # if norm(Gt1 - Gt1_) + norm(Gt2 - Gt2_) + norm(Gt01 - Gt01_) + norm(Gt02 - Gt02_) + norm(G0t1 - G0t1_) + norm(G0t2 - G0t2_) > ERROR
#             #     println(norm(Gt1 - Gt1_), '\n', norm(Gt2 - Gt2_), '\n', norm(Gt01 - Gt01_), '\n', norm(Gt02 - Gt02_), '\n', norm(G0t1 - G0t1_), '\n', norm(G0t2 - G0t2_))
#             #     error("WrapTime=$lt ")
#             # end
#             # GM_A_ = GroverMatrix(G01_[indexA[:], indexA[:]], G02_[indexA[:], indexA[:]])
#             # gmInv_A_ = inv(GM_A_)
#             # GM_B_ = GroverMatrix(G01_[indexB[:], indexB[:]], G02_[indexB[:], indexB[:]])
#             # gmInv_B_ = inv(GM_B_)
#             # detg_A_ = abs2(det(GM_A_))
#             # detg_B_ = abs2(det(GM_B_))
#             # if norm(gmInv_A_ - A.gmInv) + norm(B.gmInv - gmInv_B_) + abs(A.detg - detg_A_) + abs(B.detg - detg_B_) > ERROR
#             #     println(norm(gmInv_A_ - A.gmInv), " ", norm(B.gmInv - gmInv_B_), " ", abs(A.detg - detg_A_), " ", abs(B.detg - detg_B_))
#             #     error("s2:  $lt : WrapTime")
#             # end
#             #####################################################################

#             UpdateSCEELayer!(rng, view(ss[1], :, lt), view(ss[2], :, lt), lt, G1, G2, A, B, model, UPD, SCEE, λ)

#             ##------------------------------------------------------------------------
#             tmpO += (A.detg / B.detg)^(1 / Nλ)
#             counter += 1
#             ##------------------------------------------------------------------------

#             if any(model.nodes .== (lt - 1))
#                 idx -= 1
#                 BM_F!(tmpN, tmpNN, view(BMs1, :, :, idx), model, ss[1], idx)
#                 BM_F!(tmpN, tmpNN, view(BMs2, :, :, idx), model, ss[2], idx)
#                 BMinv_F!(tmpN, tmpNN, view(BMsinv1, :, :, idx), model, ss[1], idx)
#                 BMinv_F!(tmpN, tmpNN, view(BMsinv2, :, :, idx), model, ss[2], idx)
#                 for i in idx:-1:min(Θidx, idx)
#                     # println("update BL i=",i)
#                     mul!(tmpnN, view(BLMs1, :, :, i + 1), view(BMs1, :, :, i))
#                     LAPACK.gerqf!(tmpnN, tau)
#                     LAPACK.orgrq!(tmpnN, tau, ns)
#                     copyto!(view(BLMs1, :, :, i), tmpnN)

#                     mul!(tmpnN, view(BLMs2, :, :, i + 1), view(BMs2, :, :, i))
#                     LAPACK.gerqf!(tmpnN, tau)
#                     LAPACK.orgrq!(tmpnN, tau, ns)
#                     copyto!(view(BLMs2, :, :, i), tmpnN)
#                 end
#                 for i in idx+1:max(Θidx, idx)
#                     # println("update BR i=",i)
#                     mul!(tmpNn, view(BMs1, :, :, i - 1), view(BRMs1, :, :, i - 1))
#                     LAPACK.geqrf!(tmpNn, tau)
#                     LAPACK.orgqr!(tmpNn, tau, ns)
#                     copyto!(view(BRMs1, :, :, i), tmpNn)

#                     mul!(tmpNn, view(BMs2, :, :, i - 1), view(BRMs2, :, :, i - 1))
#                     LAPACK.geqrf!(tmpNn, tau)
#                     LAPACK.orgqr!(tmpNn, tau, ns)
#                     copyto!(view(BRMs2, :, :, i), tmpNn)
#                 end
#                 get_ABGM!(G1, G2, A, B, SCEE, model.nodes, idx, "Backward")
#             else
#                 @inbounds @simd for iii in 1:Ns
#                     @fastmath tmpN[iii] = model.exp_αη_neg[lt, ss[1][iii, lt]]
#                     @fastmath tmpN_[iii] = model.exp_αη_neg[lt, ss[2][iii, lt]]
#                 end

#                 WrapKV!(tmpN, model.eK, model.eKinv, tmpN, Gt01, "Backward", "L")
#                 WrapKV!(tmpN, model.eK, model.eKinv, tmpN_, Gt02, "Backward", "L")
#                 WrapKV!(tmpN, model.eK, model.eKinv, tmpN, Gt1, "Backward", "B")
#                 WrapKV!(tmpN, model.eK, model.eKinv, tmpN_, Gt2, "Backward", "B")
#                 WrapKV!(tmpN, model.eK, model.eKinv, tmpN, G0t1, "Backward", "R")
#                 WrapKV!(tmpN, model.eK, model.eKinv, tmpN_, G0t2, "Backward", "R")
#             end

#         end

#         O[loop+1] = tmpO / counter
#         tmpO = 0.0
#         counter = 0
#     end

#     if record
#         TTT = round(Int, (time_ns() - TTT) / 1e9)
#         hour = TTT ÷ 3600
#         minite = (TTT % 3600) ÷ 60
#         second = TTT % 60
#         println("      λ=$λ  acc = ", round(100 * UPD.acc / prod(size(ss[1])) / Sweeps / 4, digits=2), "%", "  $(Sweeps) Sweep finished in ", string(lpad(string(hour), 2, '0'), ":", lpad(string(minite), 2, '0'), ":", lpad(string(second), 2, '0')))
#         lock(LOCK) do
#             open(file, "a") do io
#                 writedlm(io, O', ',')
#             end
#         end
#     end

#     return ss
# end

# function get_ABGM!(G1::G4Buffer_, G2::G4Buffer_, A::AreaBuffer_, B::AreaBuffer_, SCEE::SCEEBuffer_, nodes, idx, direction::String="Backward")
#     G4!(SCEE, G1, nodes, idx, direction)
#     G4!(SCEE, G2, nodes, idx, direction)
#     GroverMatrix!(A.gmInv, view(G1.G0, A.index, A.index), view(G2.G0, A.index, A.index))
#     A.detg = abs2(det(A.gmInv))
#     LAPACK.getrf!(A.gmInv, A.ipiv)
#     LAPACK.getri!(A.gmInv, A.ipiv)

#     GroverMatrix!(B.gmInv, view(G1.G0, B.index, B.index), view(G2.G0, B.index, B.index))
#     B.detg = abs2(det(B.gmInv))
#     LAPACK.getrf!(B.gmInv, B.ipiv)
#     LAPACK.getri!(B.gmInv, B.ipiv)
# end

