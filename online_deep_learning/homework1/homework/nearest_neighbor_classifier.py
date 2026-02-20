# citation: used copilot and chatgpt for guidance

import torch

class NearestNeighborClassifier:
    def __init__(self, x: list[list[float]], y: list[float]):
            """
            Store the data and labels to be used for nearest neighbor classification.
            You do not have to modify this function, but you will need to implement the functions it calls.

            Args:
                x: list of lists of floats, data
                y: list of floats, labels
            """
            self.data, self.label = self.make_data(x, y)
            self.data_mean, self.data_std = self.compute_data_statistics(self.data)
            self.data_normalized = self.input_normalization(self.data)
    
    @classmethod
    def make_data(cls, x: list[list[float]], y: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        return x, y
    
    @classmethod
    def compute_data_statistics(cls, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        return mean, std
    
    def input_normalization(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.data_mean) / self.data_std # normalization puts data on same scale
    
    # [N, D] N - number of samples, D - feature dimension
    def get_nearest_neighbor(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_normalization(x)
        idx = ((self.data_normalized - x)**2).sum(dim=1).argmin()
        return (self.data[idx], self.label[idx])
    
    def get_k_nearest_neighbor(self, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_normalization(x)
        dists = ((self.data_normalized - x)**2).sum(dim=1)
        idx = torch.topk(dists, k, largest=False).indices
        return (self.data[idx], self.label[idx])
    
    def knn_regression(self, x: torch.Tensor, k: int) -> torch.Tensor:
        _, label_k = self.get_k_nearest_neighbor(x, k)
        return label_k.mean()