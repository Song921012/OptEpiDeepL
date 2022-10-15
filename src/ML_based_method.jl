# loading the Julia packages  needed.
using DifferentialEquations
using DiffEqFlux, Flux
using Plots
using CSV
using DataFrames
source_data = DataFrame(CSV.File("./Source_Data/Provincial_Daily_Totals.csv"))
data_on = source_data[source_data.Province.=="ONTARIO", :]
n = 30
nday = 59
m = nday
data_acc = data_on.TotalCases[(n+1):n+m+1]
data_daily = data_on.TotalCases[(n+1):n+m+1] - data_on.TotalCases[n:n+m]
display(plot(data_daily, label="Daily Confirmed Cases", lw=2))
display(plot(data_acc, label="Accumulated Confirmed Cases", lw=2))
data_daily[1]
println(length(data_acc))

# Model generation
ann = FastChain(FastDense(1, 32, tanh), FastDense(32, 1))
p_0 = initial_params(ann)
function SIR_nn(du, u, p, t)
    I, H = u
    du[1] = 0.1 * min(5, abs(ann(t, p)[1])) * I - 0.1 * I
    du[2] = 0.1 * min(5, abs(ann(t, p)[1])) * I
end
u_0 = Float32[1, data_acc[1]]
tspan_data = (0.0f0, Float32(nday))
prob_nn = ODEProblem(SIR_nn, u_0, tspan_data, p_0)
function train(θ)
    solvedata = Array(concrete_solve(prob_nn, Vern7(), u_0, θ, saveat=1))
end
function loss(θ)
    pred = train(θ)
    sum(abs2, (log.(data_acc) .- log.(pred[2, 1:(nday+1)]))), pred # + 1e-5*sum(sum.(abs, params(ann)))
end
println(loss(p_0))
const losses = []
callback(θ, l, pred) = begin
    push!(losses, l)
    if length(losses) % 50 == 0
        println(losses[end])
    end
    false
end

@time res1_node = DiffEqFlux.sciml_train(loss, p_0, ADAM(0.06), cb=callback, maxiters=1000)
@time res2_node = DiffEqFlux.sciml_train(loss, res1_node.minimizer, BFGS(initial_stepnorm=0.01), cb=callback, maxiters=300)
ann_param = res2_node.minimizer


using BSON: @save
@save "./output/Saving_Data/ann_nn_ir$nday.bason" ann
@save "./output/Saving_Data/ann_para_irbfgs100$nday.bason" ann_param
#@time res3_node = DiffEqFlux.sciml_train(loss, res2_node.minimizer, BFGS(initial_stepnorm = 0.01), cb = callback, maxiters = 200)
#ann_param = res3_node.minimizer
#@save "./output/Saving_Data/ann_para_irbfgs500$nday.#bason" ann_param

##
p_min = ann_param
npredict = nday
tspan_predict = (0.0, npredict)
data_acc_predict = data_on.TotalCases[(n+1):n+npredict+1]
data_daily_predict = data_on.TotalCases[(n+1):n+npredict+1] - data_on.TotalCases[n:n+npredict]
prob_prediction = ODEProblem(SIR_nn, u_0, tspan_predict, p_min)
data_prediction = Array(solve(prob_prediction, Tsit5(), saveat=1))
data_prediction_acc = data_prediction[2, :]
data_prediction_daily = zeros(length(data_prediction_acc))
data_prediction_daily[2:end] = data_prediction_acc[2:end] - data_prediction_acc[1:end-1]
scatter(data_acc_predict, label="Real accumulated cases")
plot!(data_prediction[2, :], label="Fit accumulated cases", foreground_color_legend=nothing)
xlabel!("Days after Feb 25")
title!("Data fitting ($(nday+1) days training data)")
savefig("output/epiforecastpredictacc$nday.png")
scatter(data_daily_predict, label="Real daily cases")
plot!(data_prediction_daily, label="Fit daily cases", foreground_color_legend=nothing)
xlabel!("Days after Feb 25")
title!("Data fitting ($(nday+1) days training data)")
savefig("output/epiforecastfit$nday.png")
##
p_min = ann_param
npredict = 29
tspan_predict = (0.0, npredict + nday + 1)
data_acc_predict = data_on.TotalCases[(n+1):n+npredict+1]
data_daily_predict = data_on.TotalCases[(n+1):n+npredict+1] - data_on.TotalCases[n:n+npredict]
scatter(data_acc_predict, label="Real accumulated cases")
prob_prediction = ODEProblem(SIR_nn, u_0, tspan_predict, p_min)
data_prediction = Array(solve(prob_prediction, Tsit5(), saveat=1))
data_prediction_acc = data_prediction[2, :]
data_prediction_daily = zeros(length(data_prediction_acc))
data_prediction_daily[2:end] = data_prediction_acc[2:end] - data_prediction_acc[1:end-1]
plot!(data_prediction[2, :], label="Predicted accumulated cases", foreground_color_legend=nothing)
xlabel!("Days after Feb 25")
title!("$(nday+1) days training data to predict 30 days cases")
savefig("output/epiforecastpredictacc$nday.png")
scatter(data_daily_predict, label="Real daily cases")
plot!(data_prediction_daily, label="Predicted daily cases", foreground_color_legend=nothing)
xlabel!("Days after Feb 25")
title!("$(nday+1) days training data to predict 30 days cases")
savefig("output/epiforecastpredictdaily$nday.png")
