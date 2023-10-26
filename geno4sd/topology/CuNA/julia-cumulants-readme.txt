## Julia pre-requisites 
Install Julia to your environment:
	wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.2-linux-x86_64.tar.gz
	tar zxvf julia-1.9.2-linux-x86_64.tar.gz

Export the path: export PATH="$PATH:/data/shared/burch/julia-1.9.2/bin"

Install Julia package in Python: pip install julia

Install Python package in Julia:
Launch Julia: julia
	      using Pkg
	      Pkg.add.(["IJulia", "PyCall", "Cumulants", "NPZ", "LinearAlgebra", "Random", "Statistics"])


## Move files to appropriate directories

CuNA.ipynb: move to 'tutorials/'
cumulants.py: move to 'topology/CuNA/'
cumulants.jl: move to 'topology/CuNA/'

