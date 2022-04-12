using InfiniteOpt, Ipopt, Plots
using KNITRO

# Parmeters
# Time span
t0 = 0
tf = 5
# Initial values
x10 = 0.7

x20 = 0.4

# Model
model = InfiniteModel(KNITRO.Optimizer)

# infinite_parameter
@infinite_parameter(model, t ∈ [t0, tf], num_supports = 101)

# variable
@variable(model, x1 ≥ 0, Infinite(t))
@variable(model, x2 ≥ 0, Infinite(t))

@variable(model, 0 ≤ u ≤ 1, Infinite(t), start = 0)

# objective
@objective(model, Max, ∫(log(x2), t))

# constraint
@constraint(model, x1(0) == x10)
@constraint(model, x2(0) == x20)
@constraint(model, x1_constr, ∂(x1, t) == u * x1)
@constraint(model, x2_constr, ∂(x2, t) == (1 - u) * x2)
@constraint(model, x1_min, x1 >= 0)
@constraint(model, x2_min, x2 >= 0)

# Optimization

print(model)

optimize!(model)

# Visulization
ts = value(t)
u = value(u)
x1 = value(x1)
x2 = value(x2)

plot(ts, u)
plot(ts, x1)
plot!(ts, x2)

