module GaussianBasis
export oneelectrons, oneelectrons!
export nuclear_repulsion
export Eri, Eri!, suberi!, unnormalized_eri!
export STV, STV!, unnormalized_overlap!, unnormalized_kinetic!, unnormalized_nuclear!
export boys
export lens, amis
export normalization_constant

const lens = [0, 1, 4, 10]

function factorial2(n::Int64)
    f = 1
    a = n
    while a > 0
        f *= a
        a -= 2
    end
    f
end


using StaticArrays
using LinearAlgebra
normalization_constant(ζ::Float64, n::Vector{Int64}) = (2ζ / π)^(3/4) * (4ζ)^(sum(n) / 2) / sqrt(prod(factorial2.(2n.-1)))
overlap_ss(pioz::Float64, xi::Float64, AB2::Float64) = pioz * sqrt(pioz) * exp(-xi * AB2)
kinetic_ss(xi::Float64, AB2::Float64, Sss::Float64) = xi * (3 - 2 * xi * AB2) * Sss

mutable struct STV
    la    ::Int64
    lb    ::Int64
    oo2za ::Float64
    oo2zb ::Float64
    zeta  ::Float64
    xi    ::Float64
    oo2z  ::Float64
    P     ::MVector{3, Float64}
    AP    ::MVector{3, Float64}
    BP    ::MVector{3, Float64}
    CP    ::MVector{3, Float64}
    overlap_ss   ::Float64
    kinetic_ss   ::Float64
    nuclear_ss   ::Vector{Float64}
end

function STV(la::Int64, lb::Int64, za::Float64, zb::Float64, A::SVector{3, Float64}, B::SVector{3, Float64})
    oo2za = 0.5 / za
    oo2zb = 0.5 / zb
    zeta = za + zb
    xi = za * zb / zeta
    oo2z = 0.5 / zeta
    pioz = pi / zeta
    P = (za * A + zb * B) / zeta
    AP = P - A
    BP = P - B
    AB2 = (A - B)' * (A - B)
    Sss = overlap_ss(pioz, xi, AB2)
    Tss = kinetic_ss(xi, AB2, Sss)
    STV(la, lb, oo2za, oo2zb, zeta, xi, oo2z, P, AP, BP, @SVector([0.0, 0.0, 0.0]), Sss, Tss, [0.0 for m in 0:la+lb])
end

function STV!(stv::STV, C::Vector{Float64})
    stv.CP = stv.P - SVector{3, Float64}(C)
    U = stv.zeta * stv.CP' * stv.CP
    coef = 2 * (stv.zeta / pi)^0.5 * stv.overlap_ss
    boys!(stv.la+stv.lb, U, stv.nuclear_ss)
    stv.nuclear_ss .*= coef
end


mutable struct Eri
    la    ::Int64
    lb    ::Int64
    lc    ::Int64
    ld    ::Int64
    oo2z  ::Float64
    oo2e  ::Float64
    oo2ze ::Float64
    rhoz  ::Float64
    rhoe  ::Float64
    AP    ::MVector{3, Float64}
    BP    ::MVector{3, Float64}
    CQ    ::MVector{3, Float64}
    DQ    ::MVector{3, Float64}
    PW    ::MVector{3, Float64}
    QW    ::MVector{3, Float64}
    ssss  ::Vector{Float64}
end
function Eri(l_max::Integer)
    Eri([0 for _ in 1:4]..., [0.0 for _ in 1:5]..., [@MVector(zeros(3)) for _ in 1:6]..., [0.0 for _ in 0:4l_max])
end

function Eri!(la::Int64, lb::Int64, lc::Int64, ld::Int64, za::Float64, zb::Float64, zc::Float64, zd::Float64, A::SVector{3, Float64}, B::SVector{3, Float64}, C::SVector{3, Float64}, D::SVector{3, Float64}, g::Eri)
    zeta = za + zb
    eta = zc + zd
    xi = za * zb / zeta
    ix = zc * zd / eta
    rho = zeta * eta / (zeta + eta)
    rhoz = rho / zeta
    rhoe = rho / eta
    oo2z = 0.5 / zeta
    oo2e = 0.5 / eta
    oo2ze = 0.5 / (zeta + eta)
    P = (za * A + zb * B) / zeta
    Q = (zc * C + zd * D) / eta
    W = (zeta * P + eta * Q) / (zeta + eta)
    AP = P - A
    BP = P - B
    CQ = Q - C
    DQ = Q - D
    PW = W - P
    QW = W - Q
    AB2 = (A - B)' * (A - B)
    CD2 = (C - D)' * (C - D)
    T = rho * (P - Q)' * (P - Q)
    g.la = la
    g.lb = lb
    g.lc = lc
    g.ld = ld
    g.oo2z = oo2z
    g.oo2e = oo2e
    g.oo2ze = oo2ze
    g.rhoz = rhoz
    g.rhoe = rhoe
    g.AP .= AP
    g.BP .= BP
    g.CQ .= CQ
    g.DQ .= DQ
    g.PW .= PW
    g.QW .= QW
    coef_ssss = 2 * sqrt(rho / pi) * overlap_ss(pi/zeta, xi, AB2) * overlap_ss(pi/eta, ix, CD2)
    boys!(la+lb+lc+ld, T, g.ssss)
    g.ssss .*= coef_ssss
end

include("Boys.jl")
include("GaussianIntegral.jl")

end