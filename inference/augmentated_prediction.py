import torch
import itertools
import random
import numpy as np
import torch.nn.functional as F
from typing import Tuple,List

class AugmentatedPrediction():
    
    def __init__(self, network):
        self.network = network
    
    def rotate_3d(self, x: torch.Tensor, angle: float, axes: tuple, device: torch.device) -> torch.Tensor:
        """
        Rotate a 5D tensor in the 3rd, 4th, and 5th dimensions.
        """
        # Define the rotation matrix
        angle = torch.tensor(angle * np.pi / 180, device=device)  # Convert angle to radians
        c, s = torch.cos(angle), torch.sin(angle)

        if axes == (2, 3):
            rotation_matrix = torch.tensor([[1, 0, 0, 0],
                                            [0, c, -s, 0],
                                            [0, s, c, 0]], device=device)
        elif axes == (2, 4):
            rotation_matrix = torch.tensor([[1, 0, 0, 0],
                                            [0, c, s, 0],
                                            [0, -s, c, 0]], device=device)
        elif axes == (3, 4):
            rotation_matrix = torch.tensor([[c, -s, 0, 0],
                                            [s, c, 0, 0],
                                            [0, 0, 1, 0]], device=device)
        
        rotation_matrix = rotation_matrix.unsqueeze(0).repeat(x.size(0), 1, 1)  # Match batch size

        grid = F.affine_grid(rotation_matrix, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, mode='nearest', padding_mode='border')

        return x
    
    def scale_3d(self, x: torch.Tensor, scale_factors: tuple, device: torch.device) -> torch.Tensor:

        scale_matrix = torch.tensor([[1/scale_factors[0], 0, 0, 0],
                                     [0, 1/scale_factors[1], 0, 0],
                                     [0, 0, 1/scale_factors[2], 0]], device=device).float()
        
        scale_matrix = scale_matrix.unsqueeze(0).repeat(x.size(0), 1, 1)  # Match batch size

        grid = F.affine_grid(scale_matrix, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, mode='nearest', padding_mode='border')

        return x
    
    def random_transformation(self, transfer_counts:int, x:torch.Tensor) -> Tuple[torch.Tensor,List]:
    
        transformation = ["rotate","mirror","scale"]
        mirror_axes = (2, 3, 4)
        axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
        actions = []

        for _ in range(transfer_counts):
            action = random.choice(transformation)

            if action == "rotate":
                angle,axes = np.random.uniform(-10, 10), random.choice([(2,3),(2,4),(3,4)])
                x = self.rotate_3d(x, angle, axes, x.device)

                actions.append(["rotate",angle,axes])
            elif action == "mirror":
                axes = random.choice(axes_combinations)
                x = torch.flip(x,axes)

                actions.append(["mirror",axes])
            else:
                scale_factors = tuple(np.random.uniform(0.9, 1.1, size=3))
                x = self.scale_3d(x,scale_factors,x.device)
                actions.append(["scale",scale_factors])

        return x,actions
    
    def inverse_transformation(self, prediction:torch.Tensor, actions:List) -> torch.Tensor:
        while actions:
            action = actions.pop()

            if action[0] == "rotate":
                prediction = self.rotate_3d(prediction, -action[1], action[2], prediction.device) 
            elif action[0] == "mirror":
                prediction = torch.flip(prediction,action[1])
            else:
                prediction = self.scale_3d(prediction, tuple(1/sf for sf in action[1]), prediction.device)

        return prediction
    
    def random_TTA_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        
        # Random TTA
        rnd_tr_x,actions = self.random_transformation(5,x)

        # Predict shape [1, 6, D, H, W]
        prediction = self.network(rnd_tr_x)

        # Inverse TTA
        prediction = self.inverse_transformation(prediction,actions)

        return prediction
    
            
    def estimate_uncertainty_map(self, data:torch.Tensor, scale_indexs:List) -> List[torch.Tensor]:
        
        self.network.eval()
        
        output_list = []
        
        map_counts = 6
        with torch.no_grad():
            for _ in range(map_counts):

                # data.shape [batch_size,1,D,H,W]
                rnd_tr_x,actions = self.random_transformation(5,data) # [batch_size,1,D,H,W]

                # 6 * [batch_size,6,D,H,W]
                outputs = self.network(rnd_tr_x)

                for j in scale_indexs:
                    outputs[j] = self.inverse_transformation(outputs[j], actions.copy())
                    outputs[j] = torch.softmax(outputs[j],dim=1)
                output_list.append(outputs) # shape map_counts * [6(scale) * [batch_size,6,D,H,W]]
            
        # 將同樣scale的output疊再一起
        uncertainty_maps = [torch.empty(0) for _ in range(6)]

        for i in scale_indexs:  # Iterate over the 6 different outputs
            # Extract the i-th tensor from each list in output_list
            tensors = [output_list[j][i] for j in range(map_counts)]  # List of tensors of shape [batch_size, channels, D, H, W]

            # Stack these tensors along a new dimension
            stacked_tensors = torch.stack(tensors, dim=0)  # Shape: [map_counts, batch_size, channels, D, H, W]
            
            # Exclude background
            stacked_tensors = stacked_tensors[:, :, 1:] 
            
            # Calculate the standard deviation along the new dimension
            std_tensor = torch.std(stacked_tensors, dim=0)  # Shape: [batch_size, channels, D, H, W]

            uncertainty_maps[i] = std_tensor
        
        self.network.train()
        return uncertainty_maps
    