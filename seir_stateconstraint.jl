using InfiniteOpt, Ipopt, Plots
#using KNITRO

exponential_relaxation(z, δ) = exp(one(typeof(z)) - z / δ) - one(typeof(z)) - log(δ)
function relaxed_log_barrier(z; δ=0.3f0)
    return max(z > δ ? -log(z) : exponential_relaxation(z, δ), zero(typeof(z)))
end
# Set the SEIR parameters
γ = 0.303
β = 0.727
N = 1e4
ξ = 0.2


# Set the domain information
i_max = 0.001
t0 = 0
tf = 50

# Set the intial condition values
e0 = 1 / N
i0 = 0
r0 = 0
s0 = 1 - 1 / N

model = InfiniteModel(Ipopt.Optimizer)

@infinite_parameter(model, t ∈ [t0, tf], num_supports = 201,
    derivative_method = OrthogonalCollocation(2))

@variable(model, s ≥ 0, Infinite(t))
@variable(model, e ≥ 0, Infinite(t))
@variable(model, i ≥ 0, Infinite(t))
@variable(model, r ≥ 0, Infinite(t))
@variable(model, 0.2 ≤ u ≤ 0.8, Infinite(t), start = 0.2)


@objective(model, Min, ∫(u^2, t))



# Define the initial conditions
@constraint(model, s(0) == s0)
@constraint(model, e(0) == e0)
@constraint(model, i(0) == i0)
@constraint(model, r(0) == r0)

# Define the SEIR equations
@constraint(model, s_constr, ∂(s, t) == -(1 - u) * β * s * i)
@constraint(model, e_constr, ∂(e, t) == (1 - u) * β * s * i - ξ * e)
@constraint(model, i_constr, ∂(i, t) == ξ * e - γ * i)
@constraint(model, r_constr, ∂(r, t) == γ * i)

# Define the infection rate limit
@constraint(model, imax_constr, i ≤ i_max)



print(model)

optimize!(model)


r_opt = value(r, ndarray=true) * 100 # make the population fractions into percentages
s_opt = value(s, ndarray=true) * 100
i_opt = value(i, ndarray=true) * 100
e_opt = value(e, ndarray=true) * 100
u_opt = value(u)
@show obj_opt = objective_value(model)
ts = value(t)

p = plot(ts, i_opt, label="i(t)")
plot!(ts, e_opt, label="e(t)")
ylabel!("Pop. (%)")
xlabel!("Time (Days)")
display(p)

p1 = plot(ts, u_opt, label="u(t)")
xlabel!("Time (Days)")
ylabel!("Distancing Ratio")
display(p1)


using DiffEqFlux, DifferentialEquations
const solveeq = DifferentialEquations.solve


tspan = (0.0f0, 50.0f0)
ann = FastChain(FastDense(1, 32, relu), FastDense(32, 1), (x, p) -> 0.6 * sigmoid.(x) .+ 0.2)
θ = initial_params(ann)
function sir_nn(du, u, p, t)
    S, E, I, R, z = u
    du[1] = -(1 - ann([t], p)[1]) * β * S * I
    du[2] = (1 - ann([t], p)[1]) * β * S * I - ξ * E
    du[3] = ξ * E - γ * I
    du[4] = γ * I
    du[5] = (ann([t], p)[1])^2
    du[6] = (relu(I - i_max))^2
end
u0 = [s0, e0, i0, r0, 0, 0]
ts = Float32.(collect(0.0:1:tspan[2]))
prob = ODEProblem(sir_nn, u0, tspan, θ)
sol_init = solveeq(prob, Vern9())
plot(sol_init)
function predict_adjoint(θ)
    Array(solveeq(prob, Vern9(), p=θ, saveat=ts, sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
end


# function quadratic_relaxation(z, δ)
#     return one(typeof(z)) / 2 * (((z - 2δ) / δ)^2 - one(typeof(z))) - log(δ)
# end
#
λ = 10000
for ii = 1:2
    λ = 10* λ
    function loss_adjoint(θ)
        x = predict_adjoint(θ)
        # I_barrier_loss = map(y -> relaxed_log_barrier(i_max - y; δ = 100 / N), x[3, :])
        λ * x[6, end] + x[5, end]
    end
    @show l = loss_adjoint(θ)
    cb = function (θ, l)
        println(l)
        p = plot(Array(solveeq(remake(prob, p=θ), Tsit5(), saveat=0.5))[3, :], lw=3)
        p1 = plot(ts, [first(ann([t], θ)) for t in ts], label="u(t)", lw=3)
        display(p)
        display(p1)
        return false
    end
    # Display the ODE with the current parameter values.
    #cb(θ, l)
    #loss1 = loss_adjoint(θ)
    res1 = DiffEqFlux.sciml_train(loss_adjoint, θ, ADAM(0.05), cb=cb, maxiters=100)
    θ = res1.u
    res2 = DiffEqFlux.sciml_train(loss_adjoint, θ,
        BFGS(initial_stepnorm=0.01), cb=cb, maxiters=100,
        allow_f_increases=false)
    θ = res2.u
end
p = plot(Array(solveeq(remake(prob, p=θ), Tsit5(), saveat=0.5))[3, :], lw=3)
# Part Three: Saving the results
display(p)
using DataFrames
using CSV
p_value = res1.u
results_saving_opt = DataFrame()
results_saving_opt.t = tspan
results_saving_opt.I = I_value
results_saving_opt.u = u_value
results_saving_opt.I_nn = Array(solveeq(remake(prob, p=p_value), Tsit5(), saveat=0.2))[3, :]
results_saving_opt.u_nn = [first(ann([t], p_value)) for t in ts]
CSV.write("results_saving_SEIRconstraintmodel.csv", results_saving_opt)

plot(results_saving_opt.t, results_saving_opt.I, lw=2, label="I(t) by direct collection")
plot!(results_saving_opt.t, results_saving_opt.I_nn, lw=2, label="I(t) by deep learning")
xlabel!("t(day)")
ylabel!("I(t)")
savefig("SEIRconstraint_I.png")

plot(results_saving_opt.t[2:end-1], results_saving_opt.u[2:end-1], lw=2, label="u(t) by direct collection")
plot!(results_saving_opt.t[2:end-1], results_saving_opt.u_nn[2:end-1], lw=2, label="u(t) by deep learning")
xlabel!("t(day)")
ylabel!("u(t)")
savefig("SEIRconstraint_control.png")


# Loss saving 

opt_method_loss = 13.90369069230782
nn_method_loss = 13.711496454176164