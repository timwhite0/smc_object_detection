import torch
from scipy.optimize import brentq

class SMCsampler(object):
    def __init__(self,
                 image,
                 tile_dim,
                 Prior,
                 ImageModel,
                 MutationKernel,
                 num_catalogs_per_count,
                 max_smc_iters):
        self.image = image
        self.image_dim = image.shape[0]
        
        self.tile_dim = tile_dim
        self.num_tiles_per_side = self.image_dim // self.tile_dim
        self.tiled_image = image.unfold(0,
                                        self.tile_dim,
                                        self.tile_dim).unfold(1,
                                                              self.tile_dim,
                                                              self.tile_dim)
        
        self.Prior = Prior
        self.ImageModel = ImageModel
        self.MutationKernel = MutationKernel
        self.MutationKernel.locs_lower = 0 - self.Prior.pad
        self.MutationKernel.locs_upper = self.tile_dim + self.Prior.pad
        self.MutationKernel.features_lower = 0.0 # for Pareto fluxes: self.Prior.feature_prior.scale
        self.MutationKernel.features_upper = torch.inf
        
        self.max_objects = self.Prior.max_objects
        self.num_counts = self.max_objects + 1  # num_counts = |{0,1,2,...,max_objects}|
        self.num_catalogs_per_count = num_catalogs_per_count
        self.num_catalogs = self.num_counts * self.num_catalogs_per_count
        
        self.max_smc_iters = max_smc_iters
        
        # initialize catalogs
        cats = self.Prior.sample(num_tiles_per_side = self.num_tiles_per_side,
                                 stratify_by_count = True,
                                 num_catalogs_per_count = self.num_catalogs_per_count)
        self.counts, self.locs, self.features = cats
        
        # initialize temperature
        self.temperature_prev = torch.zeros(1)
        self.temperature = torch.zeros(1)
        self.backup_temperature_seq = (2 ** (torch.arange(1, 21) / 20)) - 1
        self.use_backup_temperature_seq = False
        
        # cache loglikelihood for tempering step
        self.loglik = self.ImageModel.loglikelihood(self.tiled_image, self.locs, self.features)
        
        # initialize weights
        self.weights_log_unnorm = torch.zeros(self.num_tiles_per_side,
                                              self.num_tiles_per_side,
                                              self.num_catalogs)
        self.weights_intracount = torch.stack(torch.split(self.weights_log_unnorm,
                                                          self.num_catalogs_per_count,
                                                          dim = 2), dim = 2).softmax(3)
        self.weights_intercount = self.weights_log_unnorm.softmax(2)
        self.log_normalizing_constant = (self.weights_log_unnorm.exp().mean(2)).log()
        
        # set ESS thresholds
        self.ESS = 1 / (self.weights_intracount ** 2).sum(3)
        self.ESS_threshold_resampling = 0.5 * self.num_catalogs_per_count
        self.ESS_threshold_tempering = 0.5 * self.num_catalogs_per_count
        
        self.has_run = False


    def log_target(self, counts, locs, features, temperature):
        logprior = self.Prior.log_prob(counts, locs, features)
        loglik = self.ImageModel.loglikelihood(self.tiled_image, locs, features)
        
        return logprior + temperature * loglik


    def tempering_objective(self, loglikelihood, delta):
        log_numerator = 2 * ((delta * loglikelihood).logsumexp(0))
        log_denominator = (2 * delta * loglikelihood).logsumexp(0)

        return (log_numerator - log_denominator).exp() - self.ESS_threshold_tempering


    def temper(self):
        self.temperature_prev = self.temperature
        
        if self.use_backup_temperature_seq is False:
            self.loglik = self.ImageModel.loglikelihood(self.tiled_image, self.locs, self.features)
            solutions = torch.zeros(self.num_tiles_per_side, self.num_tiles_per_side)
            
            for h in range(self.num_tiles_per_side):
                for w in range(self.num_tiles_per_side):
                    def func(delta):
                        return self.tempering_objective(self.loglik[h,w], delta)
                    
                    if func(1 - self.temperature_prev.item()) < 0:
                        solutions[h,w] = brentq(func, 0.0, 1 - self.temperature_prev.item())
                    else:
                        solutions[h,w] = 1 - self.temperature_prev.item()
            
            delta = solutions.min()
            self.temperature = self.temperature_prev + delta
        elif self.use_backup_temperature_seq is True:
            self.temperature = self.backup_temperature_seq[self.iter]
        
        if self.iter == 0 and self.temperature > 0.5:
            self.use_backup_temperature_seq = True
            self.temperature = self.backup_temperature_seq[self.iter]
    
    
    def resample(self, method = "multinomial"):
        for count_num in range(self.num_counts):
            weights = self.weights_intracount[:,:,count_num,:]
            
            if method == "multinomial":
                weights_intracount_flat = weights.flatten(0,1)
                resampled_index_flat = weights_intracount_flat.multinomial(self.num_catalogs_per_count, replacement = True)
                resampled_index = resampled_index_flat.unflatten(0, (self.num_tiles_per_side, self.num_tiles_per_side))
            elif method == "systematic":
                resampled_index = torch.zeros_like(weights)
                for h in range(self.num_tiles_per_side):
                    for w in range(self.num_tiles_per_side):
                        u = (torch.arange(self.num_catalogs_per_count) + torch.rand([1])) / self.num_catalogs_per_count
                        bins = weights[h,w].cumsum(0)
                        resampled_index[h,w] = torch.bucketize(u, bins)
            
            resampled_index = resampled_index.int().clamp(min = 0, max = self.num_catalogs_per_count - 1)
            
            lower = count_num * self.num_catalogs_per_count
            upper = (count_num + 1) * self.num_catalogs_per_count
            
            for h in range(self.num_tiles_per_side):
                for w in range(self.num_tiles_per_side):
                    l = self.locs[h,w,lower:upper,:,:]
                    f = self.features[h,w,lower:upper,:]
                    self.locs[h,w,lower:upper,:,:] = l[resampled_index[h,w,:],:,:]
                    self.features[h,w,lower:upper,:] = f[resampled_index[h,w,:],:]
                    
            self.weights_intracount[:,:,count_num,:] = 1/self.num_catalogs_per_count
            tmp_weights_intercount = self.weights_intercount[:,:,lower:upper].sum(2) / self.num_catalogs_per_count
            self.weights_intercount[:,:,lower:upper] = tmp_weights_intercount.unsqueeze(2).repeat(1, 1, self.num_catalogs_per_count)
    
    
    def mutate(self):
        self.locs, self.features = self.MutationKernel.run(self.counts, self.locs, self.features,
                                                           self.temperature_prev, self.log_target)
    
    
    def update_weights(self):
        weights_log_incremental = (self.temperature - self.temperature_prev) * self.loglik
        
        self.weights_log_unnorm = self.weights_intercount.log() + weights_log_incremental
        self.weights_log_unnorm = torch.nan_to_num(self.weights_log_unnorm, -torch.inf)
        
        self.weights_intracount = torch.stack(torch.split(self.weights_log_unnorm,
                                                          self.num_catalogs_per_count,
                                                          dim = 2), dim = 2).softmax(3)
        self.weights_intercount = self.weights_log_unnorm.softmax(2)
        
        m = self.weights_log_unnorm.max(2).values
        w = (self.weights_log_unnorm - m.unsqueeze(2)).exp()
        s = w.sum(2)
        self.log_normalizing_constant = self.log_normalizing_constant + m + (s/self.num_catalogs).log()
        
        self.ESS = 1/(self.weights_intracount ** 2).sum(3)


    def resample_intercount(self, method = "multinomial"):
        if method == "multinomial":
            weights_intercount_flat = self.weights_intercount.flatten(0,1)
            resampled_index_flat = weights_intercount_flat.multinomial(self.num_catalogs, replacement = True)
            resampled_index = resampled_index_flat.unflatten(0, (self.num_tiles_per_side, self.num_tiles_per_side))
        elif method == "systematic":
            resampled_index = torch.zeros_like(self.weights_intercount)
            for h in range(self.num_tiles_per_side):
                for w in range(self.num_tiles_per_side):
                    u = (torch.arange(self.num_catalogs) + torch.rand([1])) / self.num_catalogs
                    bins = self.weights_intercount[h,w].cumsum(0)
                    resampled_index[h,w] = torch.bucketize(u, bins)
        
        resampled_index = resampled_index.int().clamp(min = 0, max = self.num_catalogs - 1)
            
        for h in range(self.num_tiles_per_side):
            for w in range(self.num_tiles_per_side):
                self.counts[h,w,:] = self.counts[h,w,resampled_index[h,w,:]]
                self.locs[h,w,:] = self.locs[h,w,resampled_index[h,w,:]]
                self.features[h,w,:] = self.features[h,w,resampled_index[h,w,:]]
                self.weights_intercount[h,w,:] = 1 / self.num_catalogs
        
        
    def run(self, print_progress = True):
        self.iter = 0
        
        if print_progress is True:
            print("Starting the tile samplers...")
        
        self.temper()
        self.update_weights()
        
        while self.temperature < 1 and self.iter <= self.max_smc_iters:
            self.iter += 1
            
            if print_progress is True and self.iter % 5 == 0:
                print(f"iteration {self.iter}, temperature = {self.temperature.item()}")
            
            self.resample()
            self.mutate()
            self.temper()
            self.update_weights()
        
        self.resample_intercount()
        
        self.has_run = True
        
        if print_progress is True:
            print("Done!\n")
    
    
    @property
    def posterior_mean_counts(self):
        return (self.weights_intercount * self.counts).sum(-1)
    
    
    def summarize(self):
        if self.has_run == False:
            raise ValueError("Sampler hasn't been run yet.")
        
        print(f"summary\nnumber of SMC iterations: {self.iter}")
        print(f"posterior mean count by tile:\n{self.posterior_mean_counts}")
