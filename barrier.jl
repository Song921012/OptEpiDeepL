using InfiniteOpt, Ipopt, Plots
using KNITRO

exponential_relaxation(z, δ) = exp(one(typeof(z)) - z / δ) - one(typeof(z)) - log(δ)
function relaxed_log_barrier(z; δ = 0.3f0)
    return max(z > δ ? -log(z) : exponential_relaxation(z, δ), zero(typeof(z)))
end
# Set the SEIR parameters
γ = 0.303
β = 0.727
N = 100
ξ = 0.2


# Set the domain information
i_max = 0.06
t0 = 0
tf = 50

# Set the intial condition values
e0 = 1 / N
i0 = 0
r0 = 0
s0 = 1 - 1 / N

model = InfiniteModel(KNITRO.Optimizer)

@infinite_parameter(model, t ∈ [t0, tf], num_supports = 201,
    derivative_method = OrthogonalCollocation(2))

@variable(model, s ≥ 0, Infinite(t))
@variable(model, e ≥ 0, Infinite(t))
@variable(model, i ≥ 0, Infinite(t))
@variable(model, r ≥ 0, Infinite(t))
@variable(model, 0.2 ≤ u ≤ 0.8, Infinite(t), start = 0.2)


@objective(model, Min, ∫(u, t))



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


r_opt = value(r, ndarray = true) * 100 # make the population fractions into percentages
s_opt = value(s, ndarray = true) * 100
i_opt = value(i, ndarray = true) * 100
e_opt = value(e, ndarray = true) * 100
u_opt = value(u)
@show obj_opt = objective_value(model)
ts = value(t)

p = plot(ts, i_opt, label = "i(t)")
plot!(ts, e_opt, label = "e(t)")
ylabel!("Pop. (%)")
xlabel!("Time (Days)")
display(p)

p1 = plot(ts, u_opt, label = "u(t)")
xlabel!("Time (Days)")
ylabel!("Distancing Ratio")
display(p1)