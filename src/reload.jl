# loading the Julia packages  needed.
using DifferentialEquations
using DiffEqFlux, Flux
using Plots
using Random
Random.seed!(14);
source_data = DataFrame(CSV.File("./Source_Data/Provincial_Daily_Totals.csv"))
data_on = source_data[source_data.Province.=="ONTARIO", :]
data_on
n = 30
m = 149
data_acc = data_on.TotalCases[(n+1):n+m+1]
data_daily = data_on.TotalCases[(n+1):n+m+1] - data_on.TotalCases[n:n+m]
display(plot(data_daily, label = "Daily Confirmed Cases", lw = 2))
display(plot(data_acc, label = "Accumulated Confirmed Cases", lw = 2))
data_daily[1]
println(length(data_acc))

# export data
data_export = data_on[(n+1):n+m+1, [:SummaryDate, :TotalCases]]
ontario_data_export = DataFrame()
ontario_data_export.date = data_export.SummaryDate
ontario_data_export.case = data_daily
ontario_data_export.Accase = data_acc


# Model generation
ann = FastChain(FastDense(1, 32, tanh), FastDense(32, 1))
p_0 = initial_params(ann)
function SIR_nn(du, u, p, t)
    I, H = u
    du[1] = 0.1 * min(5, abs(ann(t, p)[1])) * I - 0.1 * I
    du[2] = 0.1 * min(5, abs(ann(t, p)[1])) * I
end
u_0 = Float32[1, data_acc[1]]
tspan_data = (0.0f0, 149.0f0)
prob_nn = ODEProblem(SIR_nn, u_0, tspan_data, p_0)
function train(θ)
    solvedata = Array(concrete_solve(prob_nn, Vern7(), u_0, θ, saveat = 1,
        abstol = 1e-6, reltol = 1e-6,
        sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP())))
end


using BSON: @load
@load "./DeepLearningEffectiveReproductionNumber/Saving_Data/ann_para_irbfgs100.bason" ann_param

p_min = ann_param
tspan_predict = (0.0, 149)
scatter(data_acc, label = "Real accumulated cases")
prob_prediction = ODEProblem(SIR_nn, u_0, tspan_data, p_min)
data_prediction = Array(solve(prob_prediction, Tsit5(), saveat = 1))
plot!(data_prediction[2, :], label = "Fit accumulated cases")
xlabel!("Days after Feb 25")

tspan = collect(0:1:149)'
ann_value = abs.(ann(tspan, p_min))'
plot(ann_value, label = "Effective reproduction number")
xlabel!("Days after Feb 25")

ontario_data_export.DLRt = abs.(ann(tspan, p_min))[1, :]
ontario_data_export.DLAccases = data_prediction[2, :]
A = zeros(150)
A[1] =4
A[2:150] = data_prediction[2, 1:end-1]
ontario_data_export.DLdaily = data_prediction[2, :] -A

CSV.write("ontario_DL_export.csv", ontario_data_export)