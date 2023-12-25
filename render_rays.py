import torch

def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1),
                                 device="cpu"),
                                 accumulated_transmittance[:, :-1]), dim=1)

def render_rays(nerf_model, rays_origins, ray_directions, hn=0, hf=0.5, nb_bins=192):
    device = "cpu"

    t = torch.linspace(hn, hf, nb_bins, device = device).expand(rays_origins.shape[0], nb_bins)

    mid = (t[:,:-1] + t[:, 1:]) /2
    lower = torch.cat((t[:,:1], mid), -1)
    upper = torch.cat((mid, t[:,-1:]), -1)
    u = torch.rand(t.shape, device = device)
    t = lower + (upper - lower) * u 
    delta = torch.cat((t[:,1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(rays_origins.shape[0],1)), -1)
    
    #Compute the 3d points along each ray
    x = rays_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)

    #Expand the ray_directions tensor to match shape of x
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0,1)

    colors, sigma = nerf_model(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])

    alpha = 1 - torch.exp(-sigma * delta)

    weights = compute_accumulated_transmittance(1-alpha).unsqueeze(2) * alpha.unsqueeze(2)

    #Compute the pixel values as a weighted sum of colors along each ray
    c = (weights * colors).sum(dim=1)

    # Regularization for white background
    weight_sum = weights.sum(-1).sum(-1)

    return c + 1 - weight_sum.unsqueeze(-1)




