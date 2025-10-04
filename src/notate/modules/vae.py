import torch
import torch.nn as nn
from ..core import module_type2class
import numpy as np

class VAE(nn.Module):
    name = 'vae'
    def __init__(self, var_coef=1.0, eval_vae=False):
        """
        var_coef: float
        eval_vae: bool        
        """
        super().__init__()
        self.var_coef = var_coef
        self.eval_vae = eval_vae

        # Add _device_param
        self._device_param = nn.Parameter(torch.zeros((0,)))
        def hook(model, state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs):
            if prefix+'_device_param' not in state_dict:
                state_dict[prefix+'_device_param'] = model._device_param
        self._register_load_state_dict_pre_hook(hook, with_module=True)

    @property
    def device(self):
        return self._device_param.device
        
    def forward(self, mode='train', mu=None, var=None, latent_size=None, batch_size=None):
        """
        Parameters
        ----------
        mode: Either 'train', 'eval' or 'generate'
        """
        if mode == 'generate':
            return torch.randn(size=(batch_size, latent_size), device=self.device)
        else:
            if mode == 'train' or self.eval_vae:
                latent = mu + torch.randn(*mu.shape, device=mu.device)*torch.sqrt(var)*self.var_coef
            else:
                latent = mu
            return latent
class MinusD_KLLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, mu, var):
        return 0.5*(torch.sum(mu**2)+torch.sum(var)-torch.sum(torch.log(var))-var.numel())
    
class Random(nn.Module):
    def __init__(self):
        super().__init__()
    
    def generate_random_with_variance(self, size, variance):
            # 標準正規分布からの乱数を生成
            mean = 0
            std_dev = 1  # 標準正規分布なので標準偏差は1
            random_numbers = np.random.normal(mean, std_dev, size)
            
            # スケーリングして指定の分散に変換
            scaling_factor = np.sqrt(variance)
            scaled_random_numbers = random_numbers * scaling_factor
            
            return scaled_random_numbers
    
    def generate_correlated_vectors(self, reference_row, random_numbers, correlation):
        """指定された相関係数を持つベクトルを生成"""
        x = reference_row
        y = random_numbers
        
        z = correlation * x + (1 - correlation ** 2) ** 0.5 * y
        
        return z

    def forward(self, mu):
        # 相関係数
        target_correlation = 1.0

        # 新しい行列を保存するリスト
        new_data = []

        # 各行について処理
        for tensor_row in mu:
            row = tensor_row.cpu().numpy()
            # 各行のサイズと分散を取得
            size = len(row)
            variance = np.var(row)
            # 乱数の生成
            random_numbers = self.generate_random_with_variance(size, variance)
            
            # 相関のあるベクトルを生成
            new_vector = self.generate_correlated_vectors(row, random_numbers, target_correlation)
            
            # 新しいデータに追加
            new_data.append(new_vector)

        # 新しいNumpy配列に変換
        new_data = np.array(new_data)
        tensor = torch.from_numpy(new_data).float().to("cuda:0")

        return tensor
    

      
