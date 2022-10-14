using InfiniteOpt, Ipopt, Plots
using Ipopt

# Parmeters
# Time span
t0 = 0
tf = 1
# Initial values
x1 = 1


# Model
model = InfiniteModel(KNITRO.Optimizer)

# infinite_parameter
@infinite_parameter(model, t ∈ [t0, tf], num_supports = 101)

# variable
@variable(model, x ≥ 0, Infinite(t))

@variable(model, u, Infinite(t), start = 0)

# objective
@objective(model, Min, ∫(u^2, t))

# constraint
@constraint(model, x(0) == x1)
@constraint(model, x_constr, ∂(x, t) == x + u)


# Optimization

print(model)

optimize!(model)

# Visulization
ts = value(t)
u = value(u)
x = value(x)

display(plot(ts, u))
display(plot(ts, x))


using DiffEqFlux, DifferentialEquations
const solveeq = DifferentialEquations.solve

tspan = (0.0f0, 20.0f0)
ann = FastChain(FastDense(1, 32, tanh), FastDense(32, 1), (x, p) -> 0.9 * sigmoid.(x))
θ = initial_params(ann)
function sir_nn(du, u, p, t)
    S, E, I, N = u
    du[1] = b * N - d * S - c * S * I - ann([t], p)[1] * S
    du[2] = c * S * I - (e + d) * E
    du[3] = e * E - (g + a + d) * I
    du[4] = (b - d) * N - a * I
end
u0 = [s0, e0, i0, n0]
ts = Float32.(collect(0.0:0.2:tspan[2]))
prob = ODEProblem(sir_nn, u0, tspan, θ)
sol_init = solveeq(prob, Vern9(), abstol = 1e-10, reltol = 1e-10)
plot(sol_init)
function predict_adjoint(θ)
    Array(solveeq(prob, Vern9(), p = θ, saveat = ts, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))
end
function loss_adjoint(θ)
    x = predict_adjoint(θ)
    mean(abs, A .* x[3, :]) + mean(abs2, [first(ann([t], θ)) for t in ts])
end
l = loss_adjoint(θ)
cb = function (θ, l)
    println(l)
    p = plot(solveeq(remake(prob, p = θ), Tsit5(), saveat = 0.2), lw = 3)
    #plot!(p, ts, [first(ann([t], θ)) for t in ts], label = "u(t)", lw = 3)
    display(p)
    return false
end
# Display the ODE with the current parameter values.
cb(θ, l)
loss1 = loss_adjoint(θ)
res1 = DiffEqFlux.sciml_train(loss_adjoint, θ, ADAM(0.05), cb = cb, maxiters = 100)
res2 = DiffEqFlux.sciml_train(loss_adjoint, res1.u,
    BFGS(initial_stepnorm = 0.01), cb = cb, maxiters = 100,
    allow_f_increases = false)

# Loss saving 

opt_method_loss = 13.90369069230782
nn_method_loss = 13.711496454176164