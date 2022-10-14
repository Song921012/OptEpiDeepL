# Part One: using direct collection method
using InfiniteOpt, Ipopt, Distributions, Plots

# Set the SEIR parameters
b = 0.525
d = 0.5
c = 0.0001
e = 0.5
g = 0.1
a = 0.2
A = 0.1

# Set the domain information
t0 = 0
tf = 20


# Set the intial condition values
s0 = 1000
e0 = 100
i0 = 50
n0 = 1165

model = InfiniteModel(Ipopt.Optimizer)

@infinite_parameter(model, t ∈ [t0, tf], num_supports = 101,
    derivative_method = OrthogonalCollocation(2))

@variable(model, S ≥ 0, Infinite(t))
@variable(model, E ≥ 0, Infinite(t))
@variable(model, I ≥ 0, Infinite(t))
@variable(model, N ≥ 0, Infinite(t))
@variable(model, 0 ≤ u ≤ 0.9, Infinite(t), start = 0.2)


@objective(model, Min, ∫(A * I + u^2, t))



# Define the initial conditions
@constraint(model, S(0) == s0)
@constraint(model, E(0) == e0)
@constraint(model, I(0) == i0)
@constraint(model, N(0) == n0)

# Define the SEIR equations
@constraint(model, s_constr, ∂(S, t) == b * N - d * S - c * S * I - u * S)
@constraint(model, e_constr, ∂(E, t) == c * S * I - (e + d) * E)
@constraint(model, i_constr, ∂(I, t) == e * E - (g + a + d) * I)
@constraint(model, n_constr, ∂(N, t) == (b - d) * N - a * I)

# Define the infection rate limit



print(model)

optimize!(model)


I_value = value(I)
u_value = value(u)
tspan = value(t)
plot(tspan, I_value)
plot(tspan, u_value)


# Part Two: using Deep Learning Method
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


# Part Three: Saving the results
using DataFrames
using CSV
results_saving_opt = DataFrame()
results_saving_opt.t = tspan
results_saving_opt.I = I_value
results_saving_opt.u = u_value
results_saving_opt.I_nn = Array(solveeq(remake(prob, p = res2.u), Tsit5(), saveat = 0.2))[3, :]
results_saving_opt.u_nn = [first(ann([t], re2.u)) for t in ts]
CSV.write("results_saving_SEINmodel.csv", results_saving_opt)

plot(results_saving_opt.t, results_saving_opt.I, lw = 2, label = "I(t) by direct collection")
plot!(results_saving_opt.t, results_saving_opt.I_nn, lw = 2,foreground_color_legend=nothing, label = "I(t) by deep learning")
xlabel!("t(day)")
ylabel!("I(t)")
savefig("sein_I.png")

plot(results_saving_opt.t[2:end-1], results_saving_opt.u[2:end-1], lw = 2, label = "u(t) by direct collection")
plot!(results_saving_opt.t[2:end-1], results_saving_opt.u_nn[2:end-1], lw = 2,foreground_color_legend=nothing, label = "u(t) by deep learning")
xlabel!("t(day)")
ylabel!("u(t)")
savefig("sein_control.png")


# Loss saving 

opt_method_loss = 13.90369069230782
nn_method_loss = 13.711496454176164
