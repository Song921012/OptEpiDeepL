##
# Activate and set up the project environment
using Pkg
Pkg.activate(".")
Pkg.instantiate() # install

##
# Precompile to accelerate the computation
using PackageCompiler
@time create_sysimage([:DifferentialEquations, :DiffEqFlux, :Plots, :DataFrames, :Optimization, :SciMLSensitivity, :InfiniteOpt], sysimage_path="JuliaSysimage.so", precompile_execution_file="./src/noconstraint_mayer_Example3_2.jl")

