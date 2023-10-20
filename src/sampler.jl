mutable struct Hyperparameters
    ϵ::Float64
    N::Int
    sigma
end

Hyperparameters(;kwargs...) = begin
   ϵ = get(kwargs, :ϵ, 0.25)
   N = get(kwargs, :N, 25)
   sigma = get(kwargs, :sigma, [0.0])
   Hyperparameters(ϵ, N, sigma)
end

mutable struct Settings
    nchains::Int
    integrator::String
end

Settings(;kwargs...) = begin
    kwargs = Dict(kwargs)
    nchains = get(kwargs, :nchains, 1)
    integrator = get(kwargs, :integrator, "LF")
    Settings(nchains, integrator)
end

struct Sampler <: AbstractMCMC.AbstractSampler
   settings::Settings
   hyperparameters::Hyperparameters
   hamiltonian_dynamics::Function
end

function HMC(N::Int, ϵ::Float64; kwargs...)
   """HMC sampler"""
   sett = Settings(;kwargs...)
   hyperparameters = Hyperparameters(;N=N, ϵ=ϵ, kwargs...)

   ### integrator ###
   if sett.integrator == "LF"  # leapfrog
       hamiltonian_dynamics = Leapfrog
       grad_evals_per_step = 1.0
   elseif sett.integrator == "MN"  # minimal norm integrator
       hamiltonian_dynamics = Minimal_norm
       grad_evals_per_step = 2.0
   else
       println(string("integrator = ", integrator, "is not a valid option."))
   end

   return Sampler(sett, hyperparameters, hamiltonian_dynamics)
end

function Random_unit_vector(target::Target; _normalize=true)
    """Generates a random (isotropic) unit vector."""    
    return Random_unit_vector(target.rng, target.d; _normalize=_normalize)
end    

function Random_unit_vector(rng::MersenneTwister, d::Int; _normalize=true)
    """Generates a random (isotropic) unit vector."""
    u = randn(rng, d)
    if _normalize
        u = normalize(u)
    end        
    return u
end
    
struct State
    x
    u
    l
    g
    dE::Float64    
end  

function Init(sampler::Sampler, target::Target; kwargs...)
    sett = sampler.settings
    kwargs = Dict(kwargs)
    d = target.d
    ### initial conditions ###
    if :initial_x ∈ keys(kwargs)
        x = target.transform(kwargs[:initial_x])  
    else
        x = target.prior_draw()
    end 
    l, g = target.nlogp_grad_nlogp(x)
    u = Random_unit_vector(target, _normalize=false)
    return State(x, u, l, g, 0.0)
end 

function Step(sampler::Sampler, target::Target, state::State; kwargs...)
    local xx, uu, ll, gg
    """One step of the Langevin-like dynamics."""
    dialog = get(kwargs, :dialog, false)    
    N = sampler.hyperparameters.N 
    x, u, l, g, dE = state.x, state.u, state.l, state.g, state.dE
    # Hamiltonian step#
    for i in 1:N
        xx, uu, ll, gg = sampler.hamiltonian_dynamics(sampler, target, state)
    end
    #Metropolis Adjustment
    dEE =  (l - ll) - (dot(uu,uu) - dot(u,u))
    accept = log(rand()) < dEE
    xx = @.(accept * x + (1 - accept) * xx)
    ll = @.(accept * l + (1 - accept) * ll)
    gg = @.(accept * g + (1 - accept) * gg)
    dEE = @.(accept * dE + (1 - accept) * dEE)
    # Resample energy
    uuu = Random_unit_vector(target; _normalize=false)
    return State(xx, uuu, ll, gg, dEE)   
end
    
function _make_sample(sampler::Sampler, target::Target, state::State)
    return  Array([target.inv_transform(state.x)[:]; state.x[:]; state.dE; -state.l])
end        
    

"""
    $(TYPEDSIGNATURES)

Sample from the target distribution using the provided sampler.

Keyword arguments:
* `file_name` — if provided, save chain to disk (in HDF5 format)
* `file_chunk` — write to disk only once every `file_chunk` steps
  (default: 10)
 
Returns: a vector of samples
"""        
function Sample(sampler::Sampler, target::Target, num_steps::Int;
                thinning::Int=1, file_name=nothing, file_chunk=10, progress=true, kwargs...)

    state = Init(sampler, target; kwargs...)
            
    sample = _make_sample(sampler, target, state)
    samples = similar(sample, (length(sample), Int(floor(num_steps/thinning))))   
            
    pbar = Progress(num_steps, (progress ? 0 : Inf), "MCHMC: ")

    write_chain(file_name, size(samples)..., eltype(sample), file_chunk) do chain_file
        for i in 1:num_steps
            state = Step(sampler, target, state; kwargs...)
            if mod(i, thinning)==0
                j = Int(floor(i/thinning))
                samples[:,j] = sample = _make_sample(sampler, target, state)
                if chain_file !== nothing      
                    push!(chain_file, sample)
                end
            end
            ProgressMeter.next!(pbar, showvalues = [
                ("ϵ", sampler.hyperparameters.eps),
                ("dE/d", state.dE / target.d)
            ])
        end
    end

    ProgressMeter.finish!(pbar)

    return samples
end