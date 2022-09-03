using BenchmarkTools
using LinearAlgebra
x = rand(1000, 1000);
y = rand(1000, 1000);
z = rand(1000, 1000);

f(x, y, z) = x - y - z;
g(x, y, z) = @. x - y - z;
h(x, y, z) = x .- y .- z;

@btime f($x, $y, $z); # 3.487 ms (4 allocations: 15.26 MiB)
@btime g($x, $y, $z); # 1.941 ms (2 allocations: 7.63 MiB)
@btime h($x, $y, $z); #  1.929 ms (2 allocations: 7.63 MiB)
w = zeros(1000, 1000);
@btime mul!(w, x, y); # 13.778 ms (0 allocations: 0 bytes)
@btime w = x * y; # 15.708 ms (2 allocations: 7.63 MiB)
@btime w .= x .+ y .+ z # 1.259 ms (4 allocations: 128 bytes)
@btime @. w = x + y + z # 1.267 ms (2 allocations: 64 bytes)

A = rand(100, 100)
B = rand(100, 100)
C = rand(100, 100)
using BenchmarkTools

# 按列循环
function inner_rows!(C, A, B)
    for i in 1:100, j in 1:100
        C[i, j] = A[i, j] + B[i, j]
    end
end
@btime inner_rows!(C, A, B)

function inner_cols!(C, A, B)
    for j in 1:100, i in 1:100
        C[i, j] = A[i, j] + B[i, j]
    end
end
@btime inner_cols!(C, A, B)

# mutation防止内存到堆里，堆是慢的，栈是快的
function inner_noalloc!(C, A, B)
    for j in 1:100, i in 1:100
        val = A[i, j] + B[i, j]
        C[i, j] = val[1]
    end
end
@btime inner_noalloc!(C, A, B) # 到栈

function inner_alloc(A, B)
    C = similar(A)
    for j in 1:100, i in 1:100
        val = A[i, j] + B[i, j]
        C[i, j] = val[1]
    end
end
@btime inner_alloc(A, B) # 到堆

function inner_alloc!(A, B)
    C = similar(A)
    for j in 1:100, i in 1:100
        val = A[i, j] + B[i, j]
        C[i, j] = val[1]
    end
end
@btime inner_alloc!(A, B) # 也是到堆

# 如果另外函数内部直接用全部变量，更是到堆

using StaticArrays
C = @SMatrix rand(100, 100)
function inner_alloc_s!(A, B)
    #C = @SVector similar(A)
    for j in 1:100, i in 1:100
        val = A[i, j] + B[i, j]
        C[i, j] = val[1]
    end
end
@btime inner_alloc_s!(A, B)


s = 0.0
@time for i = 1:10_000_000
    global s
    s += sqrt(rand())
end

s = 0.0
@time for i = 1:10_000_000
    s += sqrt(rand())
end

using BenchmarkTools
function f()
    A = rand(100, 100)
    B = rand(100, 100)
    C = A + B
end
function f!()
    A = rand(100, 100)
    B = rand(100, 100)
    C = A + B
end
function f1!()
    A = rand(100, 100)
    B = rand(100, 100)
    C = similar(A)
    C .= A .+ B
end
function g(A, B)
    C = A + B
    return C
end
function g!(A, B)
    C = A + B
    return C
end
function h(A, B)
    C = similar(A)
    C .= A .+ B
    return C
end
function h!(C, A, B)
    C = similar(A)
    C = A + B
end
@btime f() # 19.100 μs (6 allocations: 234.61 KiB)
@btime f!() #同上
@btime f1!()

A = rand(100, 100)
B = rand(100, 100)
@btime g(A, B)
@btime g!(A, B)
@btime h(A, B)