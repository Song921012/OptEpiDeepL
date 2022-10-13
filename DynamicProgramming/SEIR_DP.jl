using RobotDynamics
using ForwardDiff
using FiniteDiff
using StaticArrays

# Define the model struct with parameters
RobotDynamics.@autodiff struct SEIR <: RobotDynamics.ContinuousDynamics
    β::Float64
    ξ::Float64
    γ::Float64
end
γ0 = 0.303
β0 = 0.727
N = 1e4
ξ0 = 0.2
SEIR() = SEIR(β0, ξ0, γ0)

RobotDynamics.state_dim(::SEIR) = 4
RobotDynamics.control_dim(::SEIR) = 1

function RobotDynamics.dynamics(model::SEIR, x, u)
    β = model.β   # infection rate
    ξ = model.ξ   # 1/incubation period
    γ = model.γ     # 1/infection period
    s,e,i,r=x[SA[1,2,3,4]]
    return [-(1 - u[1]) * β * s * i,(1 - u[1]) * β * s * i - ξ * e, ξ * e - γ * i,γ * i]
end
function RobotDynamics.dynamics!(model::SEIR, xdot, x, u)
    xstatic = SA[x[1], x[2], x[3], x[4]]
    ustatic = SA[u[1]]
    xdot .= RobotDynamics.dynamics(model, xstatic, ustatic)
    return nothing
end
model = SEIR()
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
n,m = RD.dims(model)
tf = 50.0        # final time (sec)
N = 201          # number of knot points
dt = tf / (N-1)  # time step (sec)