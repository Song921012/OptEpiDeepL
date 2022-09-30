using InfiniteOpt, Ipopt, Plots
# Parmeters
# Time span
t0 = 0
tf = 4
# Initial values
x1 = 0
# Model
model = InfiniteModel(Ipopt.Optimizer)
# infinite_parameter
@infinite_parameter(model, t ∈ [t0, tf], num_supports = 101)
# variable
@variable(model, x, Infinite(t))
@variable(model, u ≤ 5, Infinite(t))
# objective
@objective(model, Max, x(4) - 3 * ∫(u^2, t))
# constraint
@constraint(model, x(0) == x1)
@constraint(model, x_constr, ∂(x, t) == x + u)
# Optimization
print(model)
optimize!(model)
# Visulization
ts = value(t)
u_value = value(u)
x_value = value(x)
display(plot(ts, u_value))
display(plot(ts, x_value))

using DiffEqFlux, DifferentialEquations
const solveeq = DifferentialEquations.solve

tspan = (t0, tf)
ann = FastChain(FastDense(1, 32, tanh), FastDense(32, 32, tanh), FastDense(32, 1), (x, p) -> 5 * sigmoid.(x))
θ = initial_params(ann)
function sir_nn(du, u, p, t)
    du[1] = u[1] + ann([t], p)[1]
end
u0 = [x1]
ts = Float32.(collect(0.0:0.01:tspan[2]))
prob = ODEProblem(sir_nn, u0, tspan, θ)
function predict_adjoint(θ)
    Array(solveeq(prob, Vern9(), p = θ, saveat = ts, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))
end
function loss_adjoint_pre(θ)
    x = predict_adjoint(θ)
    -x[1, end] + sum(abs2, [first(ann([t], θ)) for t in ts])
end
function loss_adjoint(θ)
    x = predict_adjoint(θ)
    -x[1, end] + 3 * 0.01 * sum(abs2, [first(ann([t], θ)) for t in ts])
end
l = loss_adjoint(θ)
cb = function (θ, l)
    println(l)
    p = plot(solveeq(remake(prob, p = θ), Tsit5(), saveat = ts), lw = 3)
    #plot!(p, ts, [first(ann([t], θ)) for t in ts], label = "u(t)", lw = 3)
    display(p)
    return false
end
# Display the ODE with the current parameter values.
cb(θ, l)
loss1 = loss_adjoint(θ)
res1 = DiffEqFlux.sciml_train(loss_adjoint_pre, θ, ADAM(0.05), cb = cb, maxiters = 100)
res1 = DiffEqFlux.sciml_train(loss_adjoint, res1.u, ADAM(0.05), cb = cb, maxiters = 100)
res2 = DiffEqFlux.sciml_train(loss_adjoint, res1.u,
    BFGS(initial_stepnorm = 0.01), cb = cb, maxiters = 100,
    allow_f_increases = false)


using DataFrames
using CSV
p_value = res2.u
ts = Float32.(collect(0.0:0.04:tspan[2]))
results_saving_opt = DataFrame()
results_saving_opt.t = ts
results_saving_opt.I = x_value
results_saving_opt.u = u_value
results_saving_opt.I_nn = Array(solveeq(remake(prob, p = p_value), Tsit5(), saveat = ts))[1, :]
results_saving_opt.u_nn = [first(ann([t], p_value)) for t in ts]
CSV.write("results_saving_Example8_3.csv", results_saving_opt)

plot(results_saving_opt.t, results_saving_opt.I, lw = 2, label = "I(t) by direct collection")
plot!(results_saving_opt.t, results_saving_opt.I_nn, lw = 2, label = "I(t) by deep learning")
xlabel!("t(day)")
ylabel!("I(t)")
savefig("Example8_3_I.png")

plot(results_saving_opt.t[2:end-1], results_saving_opt.u[2:end-1], lw = 2, label = "u(t) by direct collection")
plot!(results_saving_opt.t[2:end-1], results_saving_opt.u_nn[2:end-1], lw = 2, label = "u(t) by deep learning")
xlabel!("t(day)")
ylabel!("u(t)")
savefig("Example8_3_control.png")


# 
opt_loss = objective_value(model)
# opt_loss = 1.75
opt_loss_nn = loss_adjoint(p_value)