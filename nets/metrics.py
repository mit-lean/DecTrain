'''
MIT License

Copyright (c) 2019 Diana Wofk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import torch
import math
import numpy as np

def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)

class Result(object):
    def __init__(self):
        self.irmse, self.imae = 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel, self.lg10 = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.delta1_lit, self.delta2_lit, self.delta3_lit = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0
        self.loss = 0 
        self.nll, self.aleatoric_nll, self.epistemic_nll = 0, 0, 0
        self.avg_depth_unmasked, self.avg_depth_masked, self.avg_depth_gt_masked, self.avg_aleatoric_unc, self.avg_epistemic_unc = 0, 0, 0, 0, 0
        self.depth_unmasked_median, self.depth_masked_median, self.depth_gt_masked_median, self.aleatoric_unc_unmasked_median, self.epistemic_unc_unmasked_median = 0, 0, 0, 0, 0
        self.depth_unmasked_max, self.depth_masked_max, self.depth_gt_masked_max, self.aleatoric_unc_unmasked_max, self.epistemic_unc_unmasked_max = 0, 0, 0, 0, 0
        self.depth_unmasked_min, self.depth_masked_min, self.depth_gt_masked_min, self.aleatoric_unc_unmasked_min, self.epistemic_unc_unmasked_min = 0, 0, 0, 0, 0
        self.depth_unmasked_var, self.depth_masked_var, self.depth_gt_masked_var, self.aleatoric_unc_unmasked_var, self.epistemic_unc_unmasked_var = 0, 0, 0, 0, 0
    def set_to_worst(self):
        self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.absrel, self.lg10 = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.delta1_lit, self.delta2_lit, self.delta3_lit = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0
        self.loss = np.inf
        self.nll, self.aleatoric_nll, self.epistemic_nll = np.inf, np.inf, np.inf
        self.avg_aleatoric_unc, self.avg_epistemic_unc = np.inf, np.inf
        self.avg_depth = np.inf
    def update(self, irmse, imae, mse, rmse, mae, absrel, lg10, delta1, delta2, delta3, delta1_lit, delta2_lit, delta3_lit, gpu_time, data_time, loss, nll, aleatoric_nll, epistemic_nll, avg_aleatoric_unc, avg_epistemic_unc, avg_depth_unmasked, avg_depth_masked, avg_depth_gt_masked, depth_unmasked_median, depth_masked_median, depth_gt_masked_median, aleatoric_unc_unmasked_median, epistemic_unc_unmasked_median, depth_unmasked_max, depth_masked_max, depth_gt_masked_max, aleatoric_unc_unmasked_max, epistemic_unc_unmasked_max, depth_unmasked_min, depth_masked_min, depth_gt_masked_min, aleatoric_unc_unmasked_min, epistemic_unc_unmasked_min, depth_unmasked_var, depth_masked_var, depth_gt_masked_var, aleatoric_unc_unmasked_var, epistemic_unc_unmasked_var):
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.absrel, self.lg10 = absrel, lg10
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3
        self.delta1_lit, self.delta2_lit, self.delta3_lit = delta1_lit, delta2_lit, delta3_lit
        self.data_time, self.gpu_time = data_time, gpu_time
        self.loss = loss
        self.nll = nll 
        self.aleatoric_nll = aleatoric_nll
        self.epistemic_nll = epistemic_nll
        self.avg_aleatoric_unc = avg_aleatoric_unc
        self.avg_epistemic_unc = avg_epistemic_unc 
        self.avg_depth_unmasked = avg_depth_unmasked
        self.avg_depth_masked = avg_depth_masked
        self.avg_depth_gt_masked = avg_depth_gt_masked

        self.depth_unmasked_median = depth_unmasked_median
        self.depth_masked_median = depth_masked_median
        self.depth_gt_masked_median = depth_gt_masked_median 
        self.aleatoric_unc_unmasked_median = aleatoric_unc_unmasked_median
        self.epistemic_unc_unmasked_median = epistemic_unc_unmasked_median

        self.depth_unmasked_max = depth_unmasked_max
        self.depth_masked_max = depth_masked_max
        self.depth_gt_masked_max = depth_gt_masked_max 
        self.aleatoric_unc_unmasked_max = aleatoric_unc_unmasked_max
        self.epistemic_unc_unmasked_max = epistemic_unc_unmasked_max

        self.depth_unmasked_min = depth_unmasked_min
        self.depth_masked_min = depth_masked_min
        self.depth_gt_masked_min = depth_gt_masked_min 
        self.aleatoric_unc_unmasked_min = aleatoric_unc_unmasked_min
        self.epistemic_unc_unmasked_min = epistemic_unc_unmasked_min

        self.depth_unmasked_var = depth_unmasked_var
        self.depth_masked_var = depth_masked_var
        self.depth_gt_masked_var = depth_gt_masked_var 
        self.aleatoric_unc_unmasked_var = aleatoric_unc_unmasked_var
        self.epistemic_unc_unmasked_var = epistemic_unc_unmasked_var
    def evaluate(self, output, target, loss, aleatoric_unc = None, epistemic_unc = None):
        if aleatoric_unc is None:
            aleatoric_unc = torch.full(target.shape, np.inf, device=target.device)
        if epistemic_unc is None:
            epistemic_unc = torch.full(target.shape, np.inf, device=target.device)
        # print('result.evaluate called')
        output = torch.squeeze(output)
        target = torch.squeeze(target)
        aleatoric_unc = torch.squeeze(aleatoric_unc)
        epistemic_unc = torch.squeeze(epistemic_unc)
        valid_mask = target>0
        # compute summary statistics for unmasked depth 
        self.avg_depth_unmasked = torch.mean(output).item() # before mask
        self.depth_unmasked_median = torch.median(output).item()
        self.depth_unmasked_min = torch.min(output).item()
        self.depth_unmasked_max = torch.max(output).item()
        self.depth_unmasked_var = torch.var(output).item()
        # mask target and output
        output = output[valid_mask]
        target = target[valid_mask]
        # compute summary statistics for masked depth 
        self.avg_depth_masked = torch.mean(output).item() # after mask
        self.depth_masked_median = torch.median(output).item()
        self.depth_masked_min = torch.min(output).item()
        self.depth_masked_max = torch.max(output).item()
        self.depth_masked_var = torch.var(output).item()
        # summary statistics for masked gt depth 
        self.avg_depth_gt_masked = torch.mean(target).item() # after mask
        self.depth_gt_masked_median = torch.median(target).item()
        self.depth_gt_masked_min = torch.min(target).item()
        self.depth_gt_masked_max = torch.max(target).item()
        self.depth_gt_masked_var = torch.var(target).item()
        # summary statistics for unmaksed aleatoric unc
        self.avg_aleatoric_unc = torch.mean(aleatoric_unc).item() # before mask (not masked during active learning acquisition)
        self.aleatoric_unc_unmasked_median = torch.median(aleatoric_unc).item()
        self.aleatoric_unc_unmasked_min = torch.min(aleatoric_unc).item()
        self.aleatoric_unc_unmasked_max = torch.max(aleatoric_unc).item()
        self.aleatoric_unc_unmasked_var = torch.var(aleatoric_unc).item()
        # summary statistics for unmaksed aleatoric unc
        self.avg_epistemic_unc = torch.mean(epistemic_unc).item() # before mask (not masked during active learning acquisition)
        self.epistemic_unc_unmasked_median = torch.median(epistemic_unc).item()
        self.epistemic_unc_unmasked_min = torch.min(epistemic_unc).item()
        self.epistemic_unc_unmasked_max = torch.max(epistemic_unc).item()
        self.epistemic_unc_unmasked_var = torch.var(epistemic_unc).item()
        # mask uncertainties
        aleatoric_unc = aleatoric_unc[valid_mask]
        epistemic_unc = epistemic_unc[valid_mask]

        abs_diff = (output - target).abs()

        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output) - log10(target)).abs().mean())
        self.absrel = float((abs_diff / target).mean())
        # new right way of calculating delta-1
        upper_bound_delta1 = (1.25)*target
        lower_bound_delta1 = (0.75)*target
        upper_bound_delta2 = (1.25**2)*target
        lower_bound_delta2 = (1-((1.25**2)-1))*target
        upper_bound_delta3 = (1.25**3)*target
        lower_bound_delta3 = (1-((1.25**3)-1))*target
        self.delta1 = torch.logical_and((output >= lower_bound_delta1),(output <= upper_bound_delta1)).float().mean().item()
        self.delta2 = torch.logical_and((output >= lower_bound_delta2),(output <= upper_bound_delta2)).float().mean().item()
        self.delta3 = torch.logical_and((output >= lower_bound_delta3),(output <= upper_bound_delta3)).float().mean().item()
        # old way of calculating delta-1 (used in the literature)
        max_ratio = torch.max(output / target, target / output)
        self.delta1_lit = (max_ratio < 1.25).float().mean().item()
        self.delta2_lit = (max_ratio < 1.25 ** 2).float().mean().item()
        self.delta3_lit = (max_ratio < 1.25 ** 3).float().mean().item()

        self.data_time = 0
        self.gpu_time = 0

        inv_output = 1 / output
        inv_target = 1 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())
        self.loss = loss

        total_unc = aleatoric_unc+epistemic_unc
        self.nll = torch.mean(0.5*torch.log(2*math.pi*(total_unc)) + torch.mul(1.0/(2*total_unc),torch.pow((output-target),2))).item() 
        self.aleatoric_nll = torch.mean(0.5*torch.log(2*math.pi*(aleatoric_unc)) + torch.mul(1.0/(2*aleatoric_unc),torch.pow((output-target),2))).item()
        self.epistemic_nll = torch.mean(0.5*torch.log(2*math.pi*(epistemic_unc)) + torch.mul(1.0/(2*epistemic_unc),torch.pow((output-target),2))).item()

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0
        self.loss_count = 0.0 # specical handle as loss can be nan if the pose if not valid

        self.sum_irmse, self.sum_imae = 0, 0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_absrel, self.sum_lg10 = 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0
        self.sum_delta1_lit, self.sum_delta2_lit, self.sum_delta3_lit = 0, 0, 0
        self.sum_data_time, self.sum_gpu_time = 0, 0
        self.sum_loss = 0
        self.sum_nll, self.sum_aleatoric_nll, self.sum_epistemic_nll = 0, 0, 0
        self.sum_avg_depth_masked, self.sum_avg_depth_unmasked, self.sum_avg_depth_gt_masked, self.sum_avg_aleatoric_unc, self.sum_avg_epistemic_unc = 0, 0, 0, 0, 0
        self.sum_depth_unmasked_median, self.sum_depth_masked_median, self.sum_depth_gt_masked_median, self.sum_aleatoric_unc_unmasked_median, self.sum_epistemic_unc_unmasked_median = 0, 0, 0, 0, 0
        self.sum_depth_unmasked_max, self.sum_depth_masked_max, self.sum_depth_gt_masked_max, self.sum_aleatoric_unc_unmasked_max, self.sum_epistemic_unc_unmasked_max = 0, 0, 0, 0, 0
        self.sum_depth_unmasked_min, self.sum_depth_masked_min, self.sum_depth_gt_masked_min, self.sum_aleatoric_unc_unmasked_min, self.sum_epistemic_unc_unmasked_min = 0, 0, 0, 0, 0
        self.sum_depth_unmasked_var, self.sum_depth_masked_var, self.sum_depth_gt_masked_var, self.sum_aleatoric_unc_unmasked_var, self.sum_epistemic_unc_unmasked_var = 0, 0, 0, 0, 0
    def update(self, result, gpu_time, data_time, n=1):
        self.count += n
        self.loss_count = self.loss_count + n if result.loss < np.inf else self.loss_count # specical handle as loss can be nan if the pose if not valid
        self.sum_irmse += n*result.irmse
        self.sum_imae += n*result.imae
        self.sum_mse += n*result.mse
        self.sum_rmse += n*result.rmse
        self.sum_mae += n*result.mae
        self.sum_absrel += n*result.absrel
        self.sum_lg10 += n*result.lg10
        self.sum_delta1 += n*result.delta1
        self.sum_delta2 += n*result.delta2
        self.sum_delta3 += n*result.delta3
        self.sum_delta1_lit += n*result.delta1_lit
        self.sum_delta2_lit += n*result.delta2_lit
        self.sum_delta3_lit += n*result.delta3_lit
        self.sum_data_time += n*data_time
        self.sum_gpu_time += n*gpu_time
        # self.sum_loss += n*result.loss
        self.sum_loss = n*result.loss + self.sum_loss if result.loss < np.inf else self.sum_loss
        self.sum_nll += n*result.nll
        self.sum_aleatoric_nll += n*result.aleatoric_nll
        self.sum_epistemic_nll += n*result.epistemic_nll
        self.sum_avg_aleatoric_unc += n*result.avg_aleatoric_unc
        self.sum_avg_epistemic_unc += n*result.avg_epistemic_unc
        self.sum_avg_depth_unmasked += n*result.avg_depth_unmasked
        self.sum_avg_depth_masked += n*result.avg_depth_masked
        self.sum_avg_depth_gt_masked += n*result.avg_depth_gt_masked

        self.sum_depth_unmasked_median += n*result.depth_unmasked_median
        self.sum_depth_masked_median += n*result.depth_masked_median
        self.sum_depth_gt_masked_median += n*result.depth_gt_masked_median
        self.sum_aleatoric_unc_unmasked_median += n*result.aleatoric_unc_unmasked_median
        self.sum_epistemic_unc_unmasked_median += n*result.epistemic_unc_unmasked_median

        self.sum_depth_unmasked_max += n*result.depth_unmasked_max
        self.sum_depth_masked_max += n*result.depth_masked_max
        self.sum_depth_gt_masked_max += n*result.depth_gt_masked_max
        self.sum_aleatoric_unc_unmasked_max += n*result.aleatoric_unc_unmasked_max
        self.sum_epistemic_unc_unmasked_max += n*result.epistemic_unc_unmasked_max

        self.sum_depth_unmasked_min += n*result.depth_unmasked_min
        self.sum_depth_masked_min += n*result.depth_masked_min
        self.sum_depth_gt_masked_min += n*result.depth_gt_masked_min
        self.sum_aleatoric_unc_unmasked_min += n*result.aleatoric_unc_unmasked_min
        self.sum_epistemic_unc_unmasked_min += n*result.epistemic_unc_unmasked_min

        self.sum_depth_unmasked_var += n*result.depth_unmasked_var
        self.sum_depth_masked_var += n*result.depth_masked_var
        self.sum_depth_gt_masked_var += n*result.depth_gt_masked_var
        self.sum_aleatoric_unc_unmasked_var += n*result.aleatoric_unc_unmasked_var
        self.sum_epistemic_unc_unmasked_var += n*result.epistemic_unc_unmasked_var
    def average(self):
        avg = Result()
        avg.update(
            self.sum_irmse / self.count, self.sum_imae / self.count, self.sum_mse / self.count, self.sum_rmse / self.count, self.sum_mae / self.count, self.sum_absrel / self.count, self.sum_lg10 / self.count, self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count, self.sum_delta1_lit / self.count, self.sum_delta2_lit / self.count, self.sum_delta3_lit / self.count, self.sum_gpu_time / self.count, self.sum_data_time / self.count, self.sum_loss / self.loss_count, self.sum_nll / self.count, self.sum_aleatoric_nll / self.count, self.sum_epistemic_nll / self.count, self.sum_avg_aleatoric_unc / self.count, self.sum_avg_epistemic_unc / self.count, self.sum_avg_depth_unmasked / self.count, self.sum_avg_depth_masked / self.count, self.sum_avg_depth_gt_masked / self.count, self.sum_depth_unmasked_median / self.count, self.sum_depth_masked_median /self.count, self.sum_depth_gt_masked_median / self.count, self.sum_aleatoric_unc_unmasked_median / self.count, self.sum_epistemic_unc_unmasked_median / self.count, self.sum_depth_unmasked_max / self.count, self.sum_depth_masked_max /self.count, self.sum_depth_gt_masked_max / self.count, self.sum_aleatoric_unc_unmasked_max / self.count, self.sum_epistemic_unc_unmasked_max / self.count, self.sum_depth_unmasked_min / self.count, self.sum_depth_masked_min /self.count, self.sum_depth_gt_masked_min / self.count, self.sum_aleatoric_unc_unmasked_min / self.count, self.sum_epistemic_unc_unmasked_min / self.count, self.sum_depth_unmasked_var / self.count, self.sum_depth_masked_var /self.count, self.sum_depth_gt_masked_var / self.count, self.sum_aleatoric_unc_unmasked_var / self.count, self.sum_epistemic_unc_unmasked_var / self.count )
        return avg
