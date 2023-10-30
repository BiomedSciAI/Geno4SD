_author__ = "Myson Burch"
__copyright__ = "Copyright 2023, IBM Research"
__version__ = "0.1"
__maintainer__ = "Myson Burch"
__email__ = "myson.burch@ibm.com"
__status__ = "Development"

import Base.Threads.@threads

using Cumulants, NPZ, LinearAlgebra, Random, Statistics

function upper_tri(A)
    """
    Generate upper triangular entries of residuals
    """

    # Get the dimensions of the matrix
    n, m = size(A)

    # Initialize an empty vector to store the upper triangular elements
    upper_triangular_vector = Vector{Float64}(undef, n*(n-1)÷2)

    # Extract the upper triangular elements and arrange them in order by row
    index = 1
    for i in 1:n
        for j in i+1:n
            upper_triangular_vector[index] = A[i, j]
            index += 1
        end
    end

    # upper triangular vector
    return upper_triangular_vector
end

function parse_third_order(x, n)
    """
    Parse third order cumulant labels
    """

    third = Vector{Float64}()
    for i in 1:n
        curr_sheet = x[i,:,:]
        sheet_without_dupes = curr_sheet[i+1:end,i+1:end]
        third_res = upper_tri(sheet_without_dupes) # sheet_without_dupes[triu(trues(size(sheet_without_dupes)), 1)]
        third_res[isnan.(third_res)] .= 0.0
        append!(third, third_res)
    end
    return third
end

function parse_fourth_order(x, n)
    """
    Parse fourth order cumulant labels
    """

    fourth = Vector{Float64}()
    for j in 1:n
        for i in 1:n
            if i <= j 
                # pass
            else
                curr_sheet = x[j,i,:,:]
                sheet_without_dupes = curr_sheet[i+1:end,i+1:end]
                fourth_res = upper_tri(sheet_without_dupes) # sheet_without_dupes[triu(trues(size(sheet_without_dupes)), 1)]
                fourth_res[isnan.(fourth_res)] .= 0.0
                append!(fourth, fourth_res)
            end
        end
    end
    return fourth
end

function permute_dat(x)
    """
    Permute columns
    """

    # Get the number of rows and columns in the matrix
    num_rows, num_cols = size(x)

    y = rand(num_rows, num_cols)

    # Loop through each column and shuffle its elements
    for col in 1:num_cols
        y[:, col] = shuffle!(x[:, col])
    end

    return y

end

function run_cumulants(x, order)
    """
    Compute cumulants
    """

    # cumulants(data::Matrix{T}, order::Int = 4, block::Int = 2)
    c = cumulants(x, parse(Int,order), 4)

    res = Vector{Float64}()

    first = Array(c[1])
    first[isnan.(first)] .= 0.0
    append!(res, first)

    second = Array(c[2])
    second = upper_tri(second) # second[triu(trues(size(second)), 1)]
    second[isnan.(second)] .= 0.0
    append!(res, second)

    third = parse_third_order(Array(c[3]), size(first)[1])
    append!(res, third)

    if order == "4"
        fourth = parse_fourth_order(Array(c[4]), size(first)[1])
        append!(res, fourth)
    end

    return res

end

## LOAD IN DATA 
x = npzread(ARGS[1]*"julia_dat.npy")

order = ARGS[2]

## RUN CUMULANTS
res = run_cumulants(x, order)

## RUN PERMUTATIONS 
n = size(res)[1]; m = 50;
dat_perms = rand(n,m);

@threads for i in 1:m
    y = permute_dat(x)
    dat_perms[:,i] = run_cumulants(y, order)
end

# Calculate the mean by rows (axis=2) and standard deviation by rows (axis=2)
mean_by_rows = mean(dat_perms, dims=2)
std_by_rows = std(dat_perms, dims=2)
std_by_rows[std_by_rows .< 1e-12] .= 0
z = res - mean_by_rows
z = ifelse.(denominator .== 0.0, 0.0, z ./ std_by_rows)
z = ifelse.(isinf.(z), 0.0, z)

## WRITE TO OUTPUT
npzwrite(ARGS[1]*"julia_cumulants.npz", res, mean_by_rows, std_by_rows, z)