import torch
from torch.distributions import Uniform
from smc.distributions import TruncatedDiagonalMVN

class MetropolisHastings(object):
    def __init__(self,
                 num_iters,
                 locs_stdev,
                 features_stdev):
        self.num_iters = num_iters
        
        self.locs_lower = None  # defined automatically within SMCsampler
        self.locs_upper = None  # defined automatically within SMCsampler
        self.locs_stdev = torch.tensor(locs_stdev)
        
        self.features_lower = None  # defined automatically within SMCsampler
        self.features_upper = None  # defined automatically within SMCsampler
        self.features_stdev = features_stdev * torch.ones(1)
    
    def run(self, counts, locs, features, temperature, log_target):
        count_indicator = torch.arange(1, counts.max().item() + 1).unsqueeze(0) <= counts.unsqueeze(3)
        
        locs_prev = locs
        features_prev = features
        
        for iter in range(self.num_iters):
            locs_proposed = TruncatedDiagonalMVN(locs_prev, self.locs_stdev, self.locs_lower,
                                                 self.locs_upper).sample() * count_indicator.unsqueeze(4)
            features_proposed = TruncatedDiagonalMVN(features_prev, self.features_stdev, self.features_lower,
                                                     self.features_upper).sample() * count_indicator
            
            log_numerator = log_target(counts, locs_proposed, features_proposed, temperature)
            log_numerator += (TruncatedDiagonalMVN(locs_proposed, self.locs_stdev, self.locs_lower,
                                                   self.locs_upper).log_prob(locs_prev) * count_indicator.unsqueeze(4)).sum([3,4])
            log_numerator += (TruncatedDiagonalMVN(features_proposed, self.features_stdev, self.features_lower,
                                                   self.features_upper).log_prob(features_prev + self.features_lower * (features_prev == 0)).nan_to_num() * count_indicator).sum(3)

            if iter == 0:
                log_denominator = log_target(counts, locs_prev, features_prev, temperature)
                log_denominator += (TruncatedDiagonalMVN(locs_prev, self.locs_stdev, self.locs_lower,
                                                         self.locs_upper).log_prob(locs_proposed) * count_indicator.unsqueeze(4)).sum([3,4])
                log_denominator += (TruncatedDiagonalMVN(features_prev, self.features_stdev, self.features_lower,
                                                         self.features_upper).log_prob(features_proposed + self.features_lower * (features_proposed == 0)).nan_to_num() * count_indicator).sum(3)

            alpha = (log_numerator - log_denominator).exp().clamp(max = 1)
            prob = Uniform(torch.zeros_like(counts), torch.ones_like(counts)).sample()
            accept = prob <= alpha
            
            locs_new = locs_proposed * (accept).unsqueeze(3).unsqueeze(4) + locs_prev * (~accept).unsqueeze(3).unsqueeze(4)
            features_new = features_proposed * (accept).unsqueeze(3) + features_prev * (~accept).unsqueeze(3)
            
            # Cache log_denominator for next iteration
            log_denominator = log_numerator * (accept) + log_denominator * (~accept)
            
            locs_prev = locs_new
            features_prev = features_new

        return [locs_new, features_new]
