function Leapfrog(sampler::Sampler, target::Target, state::State)
    eps = sampler.hyperparameters.eps
    sigma = sampler.hyperparameters.sigma
    return Leapfrog(target, eps, sigma, state.x, state.u, state.l, state.g)
end

function Leapfrog(target::Target,
                  eps::Number, sigma::AbstractVector,
                  x::AbstractVector, u::AbstractVector,
                  l::Number, g::AbstractVector)
    """leapfrog"""
    # go to the latent space
    z = x ./ sigma 
    
    #half step in momentum
    uu =  u .+ ((eps * 0.5).* (g .* sigma)) 
    #full step in x
    zz = z .+ eps .* uu
    xx = zz .* sigma # rotate back to parameter space
    ll, gg = target.nlogp_grad_nlogp(xx)

    #half step in momentum
    uu =  u .+ ((eps * 0.5).* (g .* sigma)) 

    return xx, uu, ll, gg, kinetic_change
end
