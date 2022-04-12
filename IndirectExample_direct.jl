using InfiniteOpt, Ipopt, Plots
using KNITRO

# Parmeters
A = 1
B = 1
C = 4
# Time span
t0 = 0
tf = 1
# Initial values
x1 = 1



# Model
model = InfiniteModel(Ipopt.Optimizer)

# infinite_parameter
@infinite_parameter(model, t ∈ [t0, tf], num_supports = 101)

# variable
@variable(model, x, Infinite(t))

@variable(model, u, Infinite(t), start = 0.2)

# objective
@objective(model, Max, ∫(A * x - B * u^2, t))

# constraint
@constraint(model, x(0) == x1)
@constraint(model, x_constr, ∂(x, t) == -0.5 * x^2 + C * u)


# Optimization

print(model)

optimize!(model)

# Visulization
ts = value(t)
u = value(u)
x = value(x)

display(plot(ts, u))
display(plot(ts, x))


