## Julia pre-requisites 
1. Install Julia to your environment:
	wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.2-linux-x86_64.tar.gz
	tar zxvf julia-1.9.2-linux-x86_64.tar.gz

2. Export the path: export PATH="$PATH:/data/shared/burch/julia-1.9.2/bin"

3. Launch Julia: julia
	      using Pkg
	      Pkg.add.(["Cumulants", "NPZ", "LinearAlgebra", "Random", "Statistics"])

4. Additional documentation: https://github.com/iitis/Cumulants.jl
