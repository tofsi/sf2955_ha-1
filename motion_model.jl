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
Ψ_z_tilde = @SVector [Δt^2 / 2.0; Δt; 0]
Ψ_w_tilde = @SVector [Δt^2 / 2.0; Δt; 1]
Ψ_z = @SMatrix [Δt^2/2.0 0.0;
    Δt 0.0;
    0.0 0.0;
    0.0 Δt^2/2.0;
    0.0 Δt;
    0.0 0.0]
Ψ_w = @SMatrix [Δt^2/2.0 0.0;
    Δt 0.0;
    1.0 0.0;
    0.0 Δt^2/2.0;
    0.0 Δt;
    0.0 1.0]
Z_states = @SMatrix [0.0 0.0; 3.5 0.0; 0.0 3.5; 0.0 -3.5; -3.5 0.0]
ζ_squared = 1.5^2 # Variance of the measurement noise
p_y_normalizing_constant = 1 / sqrt(2 * π * ζ_squared)
v = 90 # dB
η = 30 # Slope index


function update_particle(x, z, w)
    return Φ * x + Ψ_z * Z_states[z, :] + Ψ_w * w
end

function update_particles(x, z, w)
    # Update the particles based on the motion model
    # x is a n_particles x 6 matrix of particles
    # z is a n_particles vector of Z states
    # w is a n_particles x 2 matrix of noise
    return x * Φ .+ Z_states[z, :] * Ψ_z' .+ w * Ψ_w'
    #return Φ * x + Ψ_z * Z_states[z, :] + Ψ_w * w
end

function simulate(n_simulations::Int, steps::Int)

    # Simylate a trajectory for the motion model
    # Generate initial states from a multivariate normal distribution
    states = zeros(Float64, n_simulations, steps, 6)
    initial_distribution = MvNormal(zeros(6), Diagonal([500, 5, 5, 200, 5, 5]))
    states[:, 1, :] = rand(initial_distribution, n_simulations)'
    Z = rand(1:5, n_simulations) # Randomly select the initial Z state
    noise_distribution = MvNormal(zeros(2), σ_squared .* Matrix{Float64}(I, 2, 2))
    progress = Progress(n_simulations * (steps - 1), 1, "Simulating trajectories")
    for i = 1:n_simulations
        for j = 2:steps
            # Generate the next state based on the previous state and the motion model
            states[i, j, :] = update_particle(states[i, j-1, :], Z[i], rand(noise_distribution))
            # Generate the next Z state based on the previous Z state and the transition matrix
            Z[i] = rand(Categorical(P[Z[i], :]))
            # Update the progress bar
            next!(progress)
        end
    end
    return states
end

function p_y_given_x(x, y, station_position)
    # Calculate the probability of measurements y given position x
    μ = v - 10 * η * log10(norm(x[[1, 4]] - station_position))
    return exp(-0.5 * (y - μ)^2 / ζ_squared) / p_y_normalizing_constant
end

function sis(y, n_particles, stop_time, x0, P0, station_position)
    @assert stop_time <= size(y, 1) "stop_time exceeds the number of measurements"
    # Initialize the state and covariance matrices
    X = zeros(Float64, n_particles, 6)  # State vector
    Λ = zeros(M, M)  # Covariance matrix

    # Initialize the state with the initial value
    X = x0
    Λ = P0
    ω = ones(n_particles) .* p_y_given_x(X, y[:, t], station_position)  # Weights for the particles
    noise_distribution = MvNormal(zeros(2), σ_squared .* Matrix{Float64}(I, 2, 2))
    Z = rand(1:5, n_particles)  # Randomly select the initial Z state for each particle
    tau = zeros(Float64, stop_time, 6)  # Process mean estimate
    tau[1, :] = sum(X .* ω) / sum(ω)

    for t in 2:stop_time
        # Simulation of the motion model
        for i = 1:n_particles
            # Generate the next state based on the previous state and the motion model
            X[i, :] = update_particle(X[i, :], Z[i], rand(noise_distribution))
            # Generate the next Z state based on the previous Z state and the transition matrix
            Z[i] = rand(Categorical(P[Z[i], :]))
        end
        # Estimate process mean
        ω .*= p_y_given_x.(eachrow(X), y[:, t], station_position)  # Update weights based on the measurement likelihood
        for i = 1:6
            τ[t, i] = sum(ω .* X[:, i]) / sum(ω)  # Process mean estimate
        end
    end
    return τ
end

# simulate(1000, 100) # Simulate 1000 particles for 100 time steps