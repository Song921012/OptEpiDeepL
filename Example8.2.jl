using InfiniteOpt, Ipopt, Plots
using KNITRO

# Parmeters
# Time span
t0 = 0
tf = 2
# Initial values
x1 = 5


# Model
model = InfiniteModel(KNITRO.Optimizer)

# infinite_parameter
@infinite_parameter(model, t ∈ [t0, tf], num_supports = 101)

# variable
@variable(model, x, Infinite(t))

@variable(model, 0 ≤ u ≤ 2, Infinite(t), start = 0)

# objective
@objective(model, Max, ∫(2*x -3*u - u^2, t))

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


