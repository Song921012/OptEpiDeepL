using InfiniteOpt, Ipopt, Plots

# Parmeters
β = 0.3
γ = 0.1

i_max = 1000
# Time span
t0 = 0
tf = 50
# Initial values
i0 = 1

# Model
model = InfiniteModel(Ipopt.Optimizer)

# infinite_parameter
@infinite_parameter(model, t ∈ [t0, tf], num_supports = 101,
    derivative_method = OrthogonalCollocation(2))

# variable
@variable(model, i ≥ 0, Infinite(t))

@variable(model, 0 ≤ u ≤ 1, Infinite(t), start = 0)

# objective
@objective(model, Min, ∫(u, t))

# constraint
@constraint(model, i(0) == i0)
@constraint(model, i_constr, ∂(i, t) == (1 - u) * β * i - γ * i )
@constraint(model, imax_constr, i ≤ i_max)

# Optimization

print(model)

optimize!(model)

# Visulization
ts = value(t)
u = value(u)
I = value(i)

plot(ts,u)
plot(ts,I)

