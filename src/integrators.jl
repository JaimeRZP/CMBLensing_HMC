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
    #full step in x
    zz = z .- (ϵ .* u .- ((ϵ * 0.5).* (g .* sigma)))
    xx = zz .* sigma # rotate back to parameter space
    ll, gg = h.∂lπ∂θ(xx)
    #half step in momentum
    uu = u .- ((ϵ * 0.5).* ((g+gg) .* sigma)) 
    HH = ll - dot(uu,uu)/2
    return xx, uu, ll, gg, HH
end
