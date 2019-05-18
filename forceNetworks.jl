module ForceNetwork
import ProgressMeter

export Network, NetworkParameters, NetworkMatrices, NetworkState, findSpikes, latentVariables, step!, run!

struct NetworkParameters
    τ_synapse::Float64
    τ_membrane::Float64
    τ_r::Float64
    t_refactory::Float64
    dt::Float64
end

function NetworkParameters(;τ_synapse=2e-2, τ_membrane=1e-2, τ_r=2e-3, t_refactory=2e-3, dt=5e-5)
    NetworkParameters(τ_synapse, τ_membrane, τ_r, t_refactory, dt)
end

mutable struct NetworkMatrices
    Ω::Matrix{Float64}
    P::Matrix{Float64}
    η::Matrix{Float64}
    Φ::Matrix{Float64}
end

function NetworkMatrices(N::Int64, K::Int64; sparsity::Float64=0.1, Q::Float64=100.0, G::Float64=10.0, α::Float64=2*5e-5)
    Ω = G*randn(N,N) .* (rand(N,N).<sparsity) ./ (sqrt(N)*sparsity)
    P = α * eye(N);
    η = Q .* (1 .- 2.*rand(N, K));
    for i=1:N
        mask = abs.(Ω[i,:]).>0
        Ω[i, mask] -= sum(Ω[i,:]) / count(mask)
    end
    Φ = zeros(K, N)
    NetworkMatrices(Ω, P, η, Φ)
end

mutable struct NetworkState
    time_since_last_spike::Vector{Float64}
    I::Vector{Float64}
    V::Vector{Float64}
    x::Vector{Float64}
    spiking::Vector{Bool}
    time::Float64
end

NetworkState(N::Int64, K::Int64) = NetworkState(zeros(N), randn(N), rand(N)-1, zeros(K), zeros(Bool, N), 0.0)

struct Network
    parameters::NetworkParameters
    matrices::NetworkMatrices
    state::NetworkState
end

Network(N::Int64, K::Int64) = Network(NetworkParameters(), NetworkMatrices(N, K), NetworkState(N, K))

function step!(network::Network, target::Union{Void, Vector{Float64}} = nothing, input::Union{Void, Vector{Float64}} = nothing, recurrent::Float64=1.0)
    s = network.state; p = network.parameters; m=network.matrices
    N, K = size(m.η)
    s.time += p.dt   
    s.x = m.Φ*s.I
        
    if target != nothing #Do learning
        @assert length(target) == K
        error = target - s.x
        PI::Vector{Float64} = m.P*s.I
        denom = 1 + (s.I'*PI)
        m.P -= PI*PI' ./ denom
        m.Φ += (PI * error')'
    end
    
    feedback::Vector{Float64} = recurrent*s.x + (1-recurrent)*input # recurrent*s.x + (1-recurrent)*target + input
    s.V = s.V * (1-p.dt/p.τ_membrane) + p.dt*(m.Ω*s.I + m.η*(feedback))
    s.V[s.time_since_last_spike .<= p.t_refactory] = -1
    s.time_since_last_spike += p.dt
    s.spiking = s.V .> 0
    s.time_since_last_spike[s.spiking] = 0
    s.V[s.spiking] = 3
    s.I = s.I * (1-p.dt/p.τ_synapse) + s.spiking
    
    

end

function run!(network::Network, target::Array{Float64, 2}, train::Vector{Bool}, input::Matrix{Float64}, recurrent::Float64=1.0)
    T = size(target, 2)
    @assert size(train, 1) == T
    @assert size(input) == size(target)
    states = Vector{NetworkState}()
    @ProgressMeter.showprogress 1 "Simulating..." for t = 1:T
        if train[t]
            step!(network, target[:,t], input[:,t], recurrent)
        else
            step!(network, nothing, input[:,t], recurrent)
        end
        push!(states, deepcopy(network.state))
    end
    return states
end

function findSpikes(states::Vector{NetworkState})
    spikes = Vector{Tuple{Float64, Int64}}()
    for state in states
        for ind in find(state.spiking)
            push!(spikes, (state.time, ind))
        end
    end
    return spikes
end

function latentVariables(states::Vector{NetworkState})
    res = Matrix{Float64}(length(states), length(states[1].x))
    for i=1:length(states)
        res[i, :] = states[i].x
    end
    return res
end
end
