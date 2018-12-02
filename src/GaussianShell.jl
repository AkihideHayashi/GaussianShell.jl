module GaussianShell
using GaussianBasis
using StaticArrays
using LinearAlgebra
export Shell, suboneelectrons, suboneelectrons!, suberi!, direct_rhf!
struct Shell
    l ::Int64
    size ::Int64
    nami ::Vector{Int64}  # nami
    ζ ::Vector{Float64}  # nζ
    CN ::Matrix{Float64}  # nζ * nami
    R  ::SVector{3, Float64}  # 3
end

function Shell(ls::Vector{Int64}, ζC::Matrix{Float64}, R::Vector{Float64})
    nami = [i for l in ls for i in lens[l+1]+1:lens[l+2]]
    ζ = ζC[:, 1]
    C = hcat([ζC[:, i+1] for (i, l) in enumerate(ls) for _ in lens[l+1]:(lens[l+2]-1)]...)
    N = reshape([normalization_constant(z, amis[i]) for l in ls for i in (lens[l+1]+1):lens[l+2] for z in ζ], size(C))
    Shell(maximum(ls), length(nami), nami, ζ, N .* C, SVector{3, Float64}(R))
end
Shell(sh::Shell, R::Vector{Float64}) = Shell(sh.l, sh.size, sh.nami, sh.ζ, sh.CN, SVector{3, Float64}(R))


function suboneelectrons!(a::Shell, b::Shell, N::Vector{Float64}, R::Matrix{Float64}, S::SubArray{Float64, 2}, T::SubArray{Float64, 2}, V::SubArray{Float64, 2}, work1::Matrix{Float64}, work2::Matrix{Float64})
    S[:, :] .= 0
    T[:, :] .= 0
    V[:, :] .= 0
    for i in 1:length(a.ζ)
        for j in 1:length(b.ζ)
            g = STV(a.l, b.l, a.ζ[i], b.ζ[j], a.R, b.R)
            CN = a.CN[i, :] * b.CN[j, :]'
            unnormalized_overlap!(g, work1)
            unnormalized_kinetic!(g, work1, work2)
            S[:, :] .+= CN .* work1[a.nami, b.nami]
            T[:, :] .+= CN .* work2[a.nami, b.nami]
            for k in 1:size(R, 1)
                work1.=0
                STV!(g, R[k, :])
                unnormalized_nuclear!(g, work1)
                V[:, :] .-= CN .* work1[a.nami, b.nami, 1] * N[k]
            end
        end
    end
end

function oneelectrons!(basis::Vector{Shell}, N::Vector{Float64}, R::Matrix{Float64}, S::Matrix{Float64}, T::Matrix{Float64}, V::Matrix{Float64}, work1::Matrix{Float64}, work2::Matrix{Float64})
    si = 1
    for i in 1:length(basis)
        ei = si + length(basis[i].nami) - 1
        sj = 1
        for j in 1:length(basis)
            ej = sj + length(basis[j].nami) - 1
            ri = si:ei
            rj = sj:ej
            suboneelectrons!(basis[i], basis[j], N, R, @view(S[ri, rj]), @view(T[ri, rj]), @view(V[ri, rj]), work1, work2)
            sj += length(basis[j].nami)
        end
        si += length(basis[i].nami)
    end
end

function oneelectrons(basis::Vector{Shell}, N::Vector{Float64}, R::Matrix{Float64})
    n_basis = sum([length(b.nami) for b in basis])
    n_work = maximum([length(b.nami) for b in basis])
    S = zeros((n_basis, n_basis))
    T = zeros((n_basis, n_basis))
    V = zeros((n_basis, n_basis))
    work1 = zeros((n_work, n_work))
    work2 = zeros((n_work, n_work))
    oneelectrons!(basis, N, R, S, T, V, work1, work2)
    (S, T, V)
end


function direct_degree(i::Int64, j::Int64, k::Int64, l::Int64)
    s12_deg = i == j ? 1.0 : 2.0
    s34_deg = k == l ? 1.0 : 2.0
    s12_34_deg = i == k && j == l ? 1.0 : 2.0
    s12_deg * s34_deg * s12_34_deg
end

function direct_rhf!(basis::Vector{Shell}, D::Matrix{Float64}, G::Matrix{Float64})
    G[:, :] .= 0.0
    n_basis = sum([length(b.nami) for b in basis])
    n_work = maximum([length(b.nami) for b in basis])
    eri = zeros((n_basis, n_basis, n_basis, n_basis))
    work1 = zeros((n_work, n_work, n_work, n_work))
    work2 = zeros((n_work, n_work, n_work, n_work))
    heads = [0, [sum([length(b.nami) for b in basis[1:i]]) for i in 1:length(basis)]...][1:end-1]
    l_max = maximum(b.l for b in basis)
    g = Eri(l_max)
    for i in 1:length(basis)
        for j in 1:i
            for k in 1:i
                for l in 1:(i == k ? j : k)
                    deg = direct_degree(i, j, k, l)
                    suberi!(basis[i], basis[j], basis[k], basis[l], work2, work1, g)
                    for ii = 1:length(basis[i].nami)
                        bf1 = heads[i] + ii
                        for jj = 1:length(basis[j].nami)
                            bf2 = heads[j] + jj
                            for kk = 1:length(basis[k].nami)
                                bf3 = heads[k] + kk
                                for ll = 1:length(basis[l].nami)
                                    bf4 = heads[l] + ll
                                    val = work1[ii, jj, kk, ll] * deg
                                    G[bf1,bf2] = G[bf1,bf2] + D[bf3,bf4] * val
                                    G[bf3,bf4] = G[bf3,bf4] + D[bf1,bf2] * val
                                    G[bf1,bf3] = G[bf1,bf3] - 0.25 * D[bf2,bf4] * val
                                    G[bf2,bf4] = G[bf2,bf4] - 0.25 * D[bf1,bf3] * val
                                    G[bf1,bf4] = G[bf1,bf4] - 0.25 * D[bf2,bf3] * val
                                    G[bf2,bf3] = G[bf2,bf3] - 0.25 * D[bf1,bf4] * val
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    G .= (G .+ G') ./ 2
    return
end

function suberi!(a::Shell, b::Shell, c::Shell, d::Shell, work::Array{Float64, 4}, eri::Array{Float64, 4}, g::Eri)
    eri[:, :, :, :] .= 0
    for i in 1:length(a.ζ)
        for j in 1:length(b.ζ)
            for k in 1:length(c.ζ)
                for l in 1:length(d.ζ)
                    Eri!(a.l, b.l, c.l, d.l, a.ζ[i], b.ζ[j], c.ζ[k], d.ζ[l], a.R, b.R, c.R, d.R, g)
                    unnormalized_eri!(g, work)
                    for (ii, ni) in enumerate(a.nami)
                        cni = a.CN[i, ii]
                        for (jj, nj) in enumerate(b.nami)
                            cnj = cni * b.CN[j, jj]
                            for (kk, nk) in enumerate(c.nami)
                                cnk = cnj * c.CN[k, kk]
                                for (ll, nl) in enumerate(d.nami)
                                    cnl = cnk * d.CN[l, ll]
                                    # eri[ii, jj, kk, ll] += a.CN[i, ii] * b.CN[j,  jj] * c.CN[k, kk] * d.CN[l, ll] * work[ni, nj, nk, nl]
                                    eri[ii, jj, kk, ll] += cnl * work[ni, nj, nk, nl]
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end



end