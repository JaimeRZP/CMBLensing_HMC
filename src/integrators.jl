function Leapfrog(sampler::Sampler, target::Target, state::State)
    ϵ = sampler.hyperparameters.ϵ
    sigma = sampler.hyperparameters.sigma
    return Leapfrog(target, ϵ, sigma, state.x, state.u, state.l, state.g)
end

function Leapfrog(target::Target,
    ϵ::Number, sigma::AbstractVector,
                  x::AbstractVector, u::AbstractVector,
                  l::Number, g::AbstractVector)
    """leapfrog"""
    # go to the latent space
    z = x ./ sigma 
    
    #half step in momentum
    uu =  u .+ ((ϵ * 0.5).* (g .* sigma)) 
    #full step in x
    zz = z .+ (ϵ .* uu)
    xx = zz .* sigma # rotate back to parameter space
    ll, gg = target.nlogp_grad_nlogp(xx)

    #half step in momentum
    uu = uu .+ ((ϵ * 0.5).* (gg .* sigma)) 

    return xx, uu, ll, gg
end
