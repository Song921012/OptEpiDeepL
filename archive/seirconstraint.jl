using Flux, DiffEqFlux, DifferentialEquations, Optimization, OptimizationOptimJL, Random, Plots, Ipopt
using OptimizationMOI
#rng = Random.default_rng()
tspan = (0.0f0, 50.0f0)
ann = FastChain(FastDense(1, 32, relu), FastDense(32, 1), (x, p) -> 0.6 * sigmoid.(x) .+ 0.2)
# Set the SEIR parameters
γ = 0.303
β = 0.727
N = 1e4
ξ = 0.2


# Set the domain information

i_max = 0.004
t0 = 0
tf = 50

# Set the intial condition values
e0 = 1 / N
i0 = 0
r0 = 0
s0 = 1 - 1 / N
function sir_nn!(du, u, p, t)
    S, E, I, R, z = u
    du[1] = -(1 - ann([t], p)[1]) * β * S * I
    du[2] = (1 - ann([t], p)[1]) * β * S * I - ξ * E
    du[3] = ξ * E - γ * I
    du[4] = γ * I
    du[5] = (ann([t], p)[1])^2
end
u0 = Float32[s0, e0, i0, r0, 0, 0]
ts = Float32.(collect(0.0:1:tspan[2]))
θ = initial_params(ann)
prob = ODEProblem(sir_nn!, u0, tspan, θ)
sol_init = solve(prob, Vern9())
plot(sol_init[2, :])

function predict_adjoint(θ)
    Array(solve(prob, Vern9(), p=θ, saveat=ts))
end
function loss_adjoint(θ)
    pred = predict_adjoint(θ)
    # I_barrier_loss = map(y -> relaxed_log_barrier(i_max - y; δ = 100 / N), x[3, :])
    state = pred[2, :]
    loss = pred[5, end]
    return loss, state
end
@show l = loss_adjoint(θ)[1]
cb = function (θ, l)
    println(l)
    p = plot(Array(solveeq(remake(prob, p=θ), Tsit5(), saveat=0.5))[3, :], lw=3)
    p1 = plot(ts, [first(ann([t], θ)) for t in ts], label="u(t)", lw=3)
    display(p)
    display(p1)
    return false
end

# use Optimization.jl to solve the problem
lcons = 0.0 * ones(length(ts))
ucons = i_max * ones(length(ts))
optf = Optimization.OptimizationFunction((x, p) -> loss_adjoint(x)[1], Optimization.AutoForwardDiff(), cons=(res, x, p) -> (res .= loss_adjoint(x)[2]))
optprob = OptimizationProblem(optf, θ, lcons=lcons, ucons=ucons)
sol = solve(optprob, Ipopt.Optimizer())#, callback=cb)
