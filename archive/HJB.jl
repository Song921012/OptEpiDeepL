using NeuralPDE
using Flux
using DifferentialEquations
using LinearAlgebra
d = 100 # number of dimensions
X0 = fill(0.0f0, d) # initial value of stochastic control process
tspan = (0.0f0, 1.0f0)
λ = 1.0f0

g(X) = log(0.5f0 + 0.5f0 * sum(X .^ 2))
f(X, u, σᵀ∇u, p, t) = -λ * sum(σᵀ∇u .^ 2)
μ_f(X, p, t) = zero(X)  # Vector d x 1 λ
σ_f(X, p, t) = Diagonal(sqrt(2.0f0) * ones(Float32, d)) # Matrix d x d
prob = TerminalPDEProblem(g, f, μ_f, σ_f, X0, tspan)
hls = 10 + d # hidden layer size
opt = Flux.ADAM(0.01)  # optimizer
# sub-neural network approximating solutions at the desired point
u0 = Flux.Chain(Dense(d, hls, relu),
    Dense(hls, hls, relu),
    Dense(hls, 1))
# sub-neural network approximating the spatial gradients at time point
σᵀ∇u = Flux.Chain(Dense(d + 1, hls, relu),
    Dense(hls, hls, relu),
    Dense(hls, hls, relu),
    Dense(hls, d))
pdealg = NNPDENS(u0, σᵀ∇u, opt=opt)
@time ans = solve(prob, pdealg, verbose=true, maxiters=100, trajectories=100,
    alg=EM(), dt=1.2, pabstol=1.0f-2)
using OrdinaryDiffEq
function lorenz(du, u, p, t)
    du[1] = 10.0(u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - (8 / 3) * u[3]
end
u0 = [1.0; 0.0; 0.0]
tspan = (0.0, 100.0)
prob = ODEProblem(lorenz, u0, tspan)
sol = solve(prob, Tsit5())
sol = solve(prob, Tsit5())

using Profile
using ProfileView
# Profile 1000 runs
ProfileView.@profview for i in 1:1000
    sol = solve(prob, Tsit5())
end
vscode.profiler()