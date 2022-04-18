using DifferentialEquations, Plots
A = 1
B = 1
C = 4
x0 = 1
function f(du, u, p, t)
    du[1] = -0.5 * u[1]^2 + 0.5 * C^2 * u[2] / B
    du[2] = -A + u[1] * u[2]
end

function bc1!(residual, u, p, t)
    residual[1] = u[1][1] - x0 
    residual[2] = u[end][2]
end
tspan = (0,1)
bvp1 = BVProblem(f, bc1!, [0, 0], tspan)
sol1 = solve(bvp1, GeneralMIRK4(), dt = 0.05)
plot(sol1)