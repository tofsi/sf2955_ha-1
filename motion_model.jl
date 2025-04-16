using LinearAlgebra
using Distributions
using Random
using ProgressMeter
using Plots
using MAT
using StaticArrays # Because it makes matmul faster


P = 1 / 20.0 .* (15.0 .* Matrix{Float64}(I, 5, 5) .+ ones(Float64, 5, 5))
P = SMatrix{5,5,Float64}(P) # Transition matrix for Z states
Δt = 0.5 # seconds
α = 0.6 # Correlation between subsequent acceleration values
σ_squared = 0.5^2 # Variance of the process noise
Φ_tilde = @SMatrix [1.0 Δt Δt^2/2.0; 0.0 1.0 Δt; 0.0 0.0 α]
# soo annoying to have to write the entire thing out 
# but SMatrix is super annoying about input types
Φ = SMatrix{6,6}([
    1.0 Δt Δt^2/2.0 0.0 0.0 0.0;
    0.0 1.0 Δt 0.0 0.0 0.0;
    0.0 0.0 α 0.0 0.0 0.0;
    0.0 0.0 0.0 1.0 Δt Δt^2/2.0;
    0.0 0.0 0.0 0.0 1.0 Δt;
    0.0 0.0 0.0 0.0 0.0 α
])
Ψ_z = @SMatrix ([
    Δt^2/2.0 0.0;
    Δt 0.0;
    0.0 0.0;
    0.0 Δt^2/2.0;
    0.0 Δt;
    0.0 0.0])
Ψ_w = @SMatrix ([
    Δt^2/2.0 0.0;
    Δt 0.0;
    1.0 0.0;
    0.0 Δt^2/2.0;
    0.0 Δt;
    0.0 1.0])
Z_states = @SMatrix [0.0 0.0; 3.5 0.0; 0.0 3.5; 0.0 -3.5; -3.5 0.0]
ς_squared = 1.5^2 # Variance of the measurement noise
p_y_normalizing_constant = (1 / sqrt(2 * π * ς_squared))^6
v = 90 # dB
η = 3 # Slope index

function update_particles(x::Matrix, z::Vector, w::Matrix)
    """
    Update positions of particles based on the motion model
    
    ## Arguments:
    x : n_particles x 6 matrix of particles
    z : n_particles vector of states of Z
    w : n_particles x 2 matrix of noise

    ## Returns:
    n_particles x 6 matrix of updated particles
    """
    return x * Φ' .+ Z_states[z, :] * Ψ_z' .+ w * Ψ_w'
end

function simulate(n_simulations::Int, steps::Int)
    """
    # Simulation of the Motion model

    ## Arguments:
    n_simulations : number of particles to simulate
    steps : number of time steps to simulate

    ## Returns:
    Matrix{n_simulations, steps, 6}
        n_simulations x steps x 6 matrix of simulated particles
    """
    states = zeros(Float64, n_simulations, steps, 6)
    initial_distribution = MvNormal(zeros(6), Diagonal([500, 5, 5, 200, 5, 5]))
    states[:, 1, :] = rand(initial_distribution, n_simulations)'
    Z = rand(1:5, n_simulations) # Randomly select the initial Z state
    noise_distribution = MvNormal(zeros(2), σ_squared .* Matrix{Float64}(I, 2, 2))
    progress = Progress(n_simulations * (steps - 1), 1, "Simulating trajectories")
    transition_distributions = [Categorical(P[i, :]) for i = 1:5]
    for j = 2:steps
        # Generate the next state based on the previous state and the motion model
        states[:, j, :] = update_particles(states[:, j-1, :], Z, Matrix(rand(noise_distribution, n_simulations)'))
        # Generate the next Z state based on the previous Z state and the transition matrix
        Z = [rand(transition_distributions[Z[i]]) for i = 1:n_simulations] # Couldn't avoid this for loop ;-;
        # Update the progress bar
        next!(progress)
    end
    return states
end

function p_y_given_x(X::Matrix{Float64}, y::Vector{Float64}, station_positions::SMatrix{2,6,Float64})
    """
    Calculate the probability of measurements y given position x

    # Arguments:
    X : n_particles x 6 matrix of particles
    y : 6 x 1 vector of measurements
    station_positions : 2 x 6 matrix of station positions

    # Returns:
    Vector{n_particles, Float64}
        Vector of particle-wise likelihoods for y 
    """
    n_particles = size(X, 1)
    μs = similar(y, n_particles, 6)
    for j = 1:6
        station = station_positions[:, j]
        μs[:, j] .= v .- 10 * η * log10.(
                             sqrt.((X[:, 1] .- station[1]) .^ 2 .+ (X[:, 4] .- station[2]) .^ 2)
                         )
    end
    return exp.(-0.5 .* sum((μs .- y') .^ 2, dims=2)[:, 1] ./ ς_squared) ./ p_y_normalizing_constant
end

function sis(y::Matrix, n_particles::Int, stop_time::Int, station_positions::SMatrix{2,6,Float64})
    """
    # Sequential Importance Sampling (SIS) algorithm

    ## Arguments:
    y : 6 x # measurements matrix of measurements
    n_particles : number of particles to use in the algorithm
    stop_time : number of time steps to simulate
    station_positions : 2 x 6 matrix of station positions
    
    ## Returns:
    Matrix{stop_time, 6, Float64}
        process mean estimates
    """
    @assert stop_time <= size(y, 2) "stop_time exceeds the number of measurements"
    X = Matrix(rand(MvNormal(zeros(6), Diagonal([500, 5, 5, 200, 5, 5])), n_particles)')
    ω = [p_y_given_x(X[i, :], y[:, 1], station_positions) for i = 1:n_particles]  # Weights for the particles
    noise_distribution = MvNormal(zeros(2), σ_squared .* Matrix{Float64}(I, 2, 2))
    Z = rand(1:5, n_particles)  # Randomly select the initial Z state for each particle
    τ = zeros(Float64, stop_time, 6)  # Process mean estimate
    for i = 1:6
        τ[1, i] = sum(ω .* X[:, i]) / sum(ω)  # Process mean estimate
    end
    progress = Progress((stop_time - 1), "u better work sis <3")
    transition_distributions = [Categorical(P[i, :]) for i = 1:5]
    for t in 2:stop_time
        # Simulation of the motion model
        X = update_particles(X, Z, Matrix(rand(noise_distribution, n_particles)'))
        Z = [rand(transition_distributions[Z[i]]) for i = 1:n_particles] # Couldn't avoid this for loop ;-;
        ω .*= p_y_given_x(X, y[:, t], station_positions)
        for i = 1:6
            τ[t, i] = sum(ω .* X[:, i]) / sum(ω)  # Process mean estimate
        end
        next!(progress)
    end
    return τ
end

function sisr(y::Matrix, n_particles::Int, stop_time::Int, station_positions::SMatrix{2,6,Float64}, functions_to_estimate::AbstractVector{<:Function})
    """
    # Sequential Importance Sampling with Resampling (SISR) algorithm

    ## Arguments:
    y : 6 x # measurements matrix of measurements
    n_particles : number of particles to use in the algorithm
    stop_time : number of time steps to simulate
    station_positions : 2 x 6 matrix of station positions
    functions_to_estimate : array of functions for which to estimate the process mean
    
    ## Returns:
    τ : stop_time x 6 matrix of process mean estimates
    """
    @assert stop_time <= size(y, 2) "stop_time exceeds the number of measurements"
    X = Matrix(rand(MvNormal(zeros(6), Diagonal([500, 5, 5, 200, 5, 5])), n_particles)')
    ω = [p_y_given_x(X[i, :], y[:, 1], station_positions) for i = 1:n_particles]  # Weights for the particles
    noise_distribution = MvNormal(zeros(2), σ_squared .* Matrix{Float64}(I, 2, 2))
    Z = rand(1:5, n_particles)  # Randomly select the initial Z state for each particle
    n_functions = size(functions_to_estimate, 1)
    τ = zeros(Float64, n_functions, stop_time, 6)  # Process mean estimate
    for i = 1:n_functions
        ϕ_X = functions_to_estimate[i](X)  # Apply the function ϕ to the resampled particles
        for j = 1:6
            τ[i, 1, j] = sum(ϕ_X[:, j]) / n_particles  # Process mean estimate
        end
    end
    progress = Progress((stop_time - 1), "u better work sisr <3")
    transition_distributions = [Categorical(P[i, :]) for i = 1:5]
    for t in 2:stop_time
        # Simulation of the motion model
        X = update_particles(X, Z, Matrix(rand(noise_distribution, n_particles)'))
        Z = [rand(transition_distributions[Z[i]]) for i = 1:n_particles] # Couldn't avoid this for loop ;-;
        ω = p_y_given_x(X, y[:, t], station_positions)
        # Estimate process mean
        resampling_indices = rand(Categorical(ω ./ sum(ω)), n_particles)  # Resample indices based on weights
        X = X[resampling_indices, :]  # Resample the particles
        for i = 1:n_functions
            ϕ_X = functions_to_estimate[i](X)  # Apply the function ϕ to the resampled particles
            for j = 1:6
                τ[i, t, j] = sum(ϕ_X[:, j]) / n_particles  # Process mean estimate
            end
        end
        #= ϕ_X = ϕ(X)  # Apply the function ϕ to the resampled particles

        for i = 1:6
            τ[t, i] = sum(ϕ_X[:, i]) / n_particles  # Process mean estimate
        end =#
        # Resample the particles based on the weights
        next!(progress)
    end
    return τ
end

function ς_grid_search(y::Matrix, n_particles::Int, stop_time::Int, station_positions::SMatrix{2,6,Float64}, ς_values::Vector{Float64})
    """
    # Grid search for the optimal ς value

    ## Arguments:
    y : 6 x # measurements matrix of measurements
    n_particles : number of particles to use in the algorithm
    stop_time : number of time steps to simulate
    station_positions : 2 x 6 matrix of station positions
    ς_values : vector of ς values to test
    
    ## Returns:
    Float64 : Estimate of the standard deviation ς of the measurement noise
    Matrix{stop_time, Float64} : process mean for each ς value

    """
    log_likelihoods = zeros(Float64, length(ς_values))
    for i in 1:length(ς_values)
        ς_squared = ς_values[i]^2
        log_likelihoods[i] = sum(log.(p_y_given_x(X, y[:, t], station_positions)) for t in 1:stop_time) / n_particles
    end
    return log_likelihoods
end
#simulate(1000, 100) # Simulate 1000 particles for 100 time steps
#= y = matread("RSSI-measurements.mat")["Y"] # SMatrix bad for large arrays
station_positions = SMatrix{2,6}(matread("stations.mat")["pos_vec"])
n_particles = 1000
stop_time = size(y, 2) # number of time steps
estimated_movement_sis = sis(y, n_particles, stop_time, station_positions) =#
#= y = matread("RSSI-measurements.mat")["Y"] # SMatrix bad for large arrays
station_positions = SMatrix{2,6}(matread("stations.mat")["pos_vec"])
n_particles = 10000
stop_time = size(y, 2)
a = sis(y, n_particles, stop_time, station_positions)
simulate(1000, 100) # Simulate 1000 particles for 100 time steps
print(unique(a)) =#