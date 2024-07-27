import torch
from torch.distributions import Uniform, Categorical
from smc.distributions import TruncatedDiagonalMVN

class PointProcessPrior(object):
    def __init__(self,
                 max_objects,
                 image_height,
                 image_width,
                 pad = 0):
        self.max_objects = max_objects
        self.image_height = image_height
        self.image_width = image_width
        self.pad = pad
        self.update_attrs()
    
    
    def update_attrs(self):
        self.num_counts = self.max_objects + 1    
        self.count_prior = Categorical((1 / self.num_counts) * torch.ones(self.num_counts))
        self.loc_prior = Uniform((0 - self.pad) * torch.ones(2),
                                 torch.tensor((self.image_height + self.pad,
                                               self.image_width + self.pad)))
    
    
    def sample(self,
               num_catalogs = 1,
               num_tiles_per_side = 1,
               stratify_by_count = False,
               num_catalogs_per_count = None):
        if stratify_by_count is True and num_catalogs_per_count is None:
            raise ValueError("If stratify_by_count is True, need to specify catalogs_per_count.")
        elif stratify_by_count is False and num_catalogs_per_count is not None:
            raise ValueError("If stratify_by_count is False, do not specify catalogs_per_count.")
        
        if stratify_by_count is False:
            self.num = num_catalogs
            counts = self.count_prior.sample([num_tiles_per_side, num_tiles_per_side, self.num])
        elif stratify_by_count is True:
            self.num = self.num_counts * num_catalogs_per_count
            strata = torch.arange(self.num_counts).repeat_interleave(num_catalogs_per_count)
            counts = strata * torch.ones(num_tiles_per_side, num_tiles_per_side, self.num)
        
        self.count_indicator = torch.arange(1, self.num_counts).unsqueeze(0) <= counts.unsqueeze(3)
        locs = self.loc_prior.sample([num_tiles_per_side, num_tiles_per_side, self.num, self.max_objects])
        locs *= self.count_indicator.unsqueeze(4)
        
        return [counts, locs]
    
    # we define log_prob for stratify_by_count = True, to be used within SMCsampler
    def log_prob(self, counts, locs):
        self.count_indicator = torch.arange(1, self.num_counts).unsqueeze(0) <= counts.unsqueeze(-1)

        log_prior = self.count_prior.log_prob(counts)
        log_prior += (self.loc_prior.log_prob(locs) * self.count_indicator.unsqueeze(-1)).sum([-1,-2])
        
        return log_prior


# TODO: Move each subclass of PointProcessPrior to its own case_studies directory
class StarPrior(PointProcessPrior):
    def __init__(self,
                 *args,
                 flux_mean,
                 flux_stdev,
                 flux_min,
                 flux_max,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.flux_mean = flux_mean * torch.ones(1)
        self.flux_stdev = flux_stdev * torch.ones(1)
        self.flux_min = flux_min * torch.ones(1)
        self.flux_max = flux_max * torch.ones(1)
        self.feature_prior = TruncatedDiagonalMVN(self.flux_mean, self.flux_stdev,
                                                  self.flux_min, self.flux_max)
    
    def sample(self,
               num_catalogs = 1,
               num_tiles_per_side = 1,
               stratify_by_count = False,
               num_catalogs_per_count = None):
        counts, locs = super().sample(num_catalogs, num_tiles_per_side,
                                      stratify_by_count, num_catalogs_per_count)
        
        features = self.feature_prior.sample([num_tiles_per_side, num_tiles_per_side,
                                              self.num, self.max_objects])
        features *= self.count_indicator
        
        return [counts, locs, features]
    
    # we define log_prob for stratify_by_count = True, to be used within SMCsampler
    def log_prob(self, counts, locs, features):
        log_prior = super().log_prob(counts, locs)
        
        # add fudge to dummy vals before log_prob because the empty slots in features are zeros
        fudge = self.feature_prior.base_dist.mean
        return log_prior + (self.feature_prior.log_prob(features + fudge * (features == 0)) * self.count_indicator).sum(-1)
