using InfiniteOpt, Ipopt, Plots
using KNITRO

# Parmeters
# Time span
t0 = 0
tf = 2
# Initial values
x1 = 1


# Model
model = InfiniteModel(Ipopt.Optimizer)

# infinite_parameter
@infinite_parameter(model, t ∈ [t0, tf], num_supports = 101)

# variable
@variable(model, x , Infinite(t))

@variable(model, 0 ≤ u ≤ 4, Infinite(t))

# objective
@objective(model, Min, ∫((x-t^2)^2, t))

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

@show plot(ts, u)
@show plot(ts, x)


