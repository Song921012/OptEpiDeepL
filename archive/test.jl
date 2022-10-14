using Flux, OrdinaryDiffEq, SciMLSensitivity, Optimization, OptimizationOptimisers, OptimizationOptimJL, Plots

u0 = Float32[2.; 0.]
datasize = 30
tspan = (0.0f0,1.5f0)

function trueODEfunc(du,u,p,t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end
t = range(tspan[1],tspan[2],length=datasize)
prob = ODEProblem(trueODEfunc,u0,tspan)
ode_data = Array(solve(prob,Tsit5(),saveat=t))

dudt2 = Flux.Chain(x -> x.^3,
             Flux.Dense(2,50,tanh),
             Flux.Dense(50,2))
p,re = Flux.destructure(dudt2) # use this p as the initial condition!
dudt(u,p,t) = re(p)(u) # need to restrcture for backprop!
prob = ODEProblem(dudt,u0,tspan)

θ = [u0;p] # the parameter vector to optimize

function predict_n_ode(θ)
  Array(solve(prob,Tsit5(),u0=θ[1:2],p=θ[3:end],saveat=t))
end

function loss_n_ode(θ)
    pred = predict_n_ode(θ)
    loss = sum(abs2,ode_data .- pred)
    loss,pred
end

loss_n_ode(θ)

callback1 = function (θ,l,pred;doplot=false) #callback function to observe training
  display(l)
  # plot current prediction against data
  pl = scatter(t,ode_data[1,:],label="data")
  scatter!(pl,t,pred[1,:],label="prediction")
  display(plot(pl))
  return false
end

# Display the ODE with the initial parameter values.
callback1(θ,loss_n_ode(θ)...)

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((p,_)->loss_n_ode(p), adtype)
optprob = Optimization.OptimizationProblem(optf, θ)

result_neuralode = Optimization.solve(optprob,
                                       OptimizationOptimisers.Adam(0.05),
                                       callback = callback1,
                                       maxiters = 300)

optprob2 = remake(optprob,u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2,
                                        LBFGS(),
                                        callback = callback1,
                                        allow_f_increases = false)