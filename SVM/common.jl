# Some basic functions to interact with the data.

function gencov(d::Integer)
  A = rand(d, d)
  eye(d) + (A + A')/2
end

function plotclouds(cloudA, cloudB)
  scatter(cloudA[1,:], cloudA[2,:], label="-1")
  scatter!(cloudB[1,:], cloudB[2,:], label="1")
end

function errorrate(classA::MvNormal,
                   classB::MvNormal,
                   w::AbstractVector, n::Integer)
  testA = rand(classA, n)
  testB = rand(classB, n)
  X = [[testA testB]' ones(2*n)]
  Y = [-1.*ones(n); ones(n)]
  sum(sign.(X*w).≠Y)/n
end

function drawborder(f)
  xs = linspace(xlims()..., 100)
  ys = linspace(ylims()..., 100)
  contour!(xs, ys, f, levels=[0], colorbar=false, ls=:dash)
end

function plotdualitygap(m::Integer,
                        μ::Real,
                        numstepsarray::AbstractVector{T}) where T<:Integer
  dualgaps = [m/μ^i for i=0:length(numstepsarray)]
  numstepssumsarray = reduce(
    (arr, n) -> [arr; arr[end]+n] ,[0], numstepsarray)
  dualgaps2 = repeat(dualgaps, inner=2)[1:end-1]
  numstepssumsarray2 = repeat(numstepssumsarray, inner=2)[2:end]
  plot(numstepssumsarray2, dualgaps2, yscale=:log10,
       xlabel="Newton iterations", ylabel="duality gap")
end
