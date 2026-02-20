# citation: used copilot and chatgpt for guidance

import torch


class PyTorchBasics:
    # return every 3rd element of the input tensor.
    @staticmethod
    def make_it_pytorch_1(x: torch.Tensor) -> torch.Tensor:
        return x[::3]

    # return the maximum value of each row of the final dimension
    @staticmethod
    def make_it_pytorch_2(x: torch.Tensor) -> torch.Tensor:
        return torch.max(x, dim=2).values

    # return the unique elements of the input tensor in sorted order
    @staticmethod
    def make_it_pytorch_3(x: torch.Tensor) -> torch.Tensor:
        return torch.unique(x)

    # return the number of elements in y that are greater than the mean of x
    @staticmethod
    def make_it_pytorch_4(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (y>x.float().mean()).sum()

    # return the transpose of the input tensor
    @staticmethod
    def make_it_pytorch_5(x: torch.Tensor) -> torch.Tensor:
        return x.T

    # return the diagonal elements (top left to bottom right) of the input tensor
    @staticmethod
    def make_it_pytorch_6(x: torch.Tensor) -> torch.Tensor:
        return x.diagonal()

    # return the diagonal elements (top right to bottom left) of the input tensor
    @staticmethod
    def make_it_pytorch_7(x: torch.Tensor) -> torch.Tensor:
        return x.fliplr().diagonal()

    # return the cummulative sum of the input tensor
    @staticmethod
    def make_it_pytorch_8(x: torch.Tensor) -> torch.Tensor:
        return x.cumsum(0)

    # compute the sum of all elements in the rectangle upto (i, j)th element
    @staticmethod
    def make_it_pytorch_9(x: torch.Tensor) -> torch.Tensor:
        return x.cumsum(0).cumsum(1)

    # return the input tensor with all elements less than c set to 0
    @staticmethod
    def make_it_pytorch_10(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return x * (x >= c).float()

    # return the row and column indices of the elements less than c
    @staticmethod
    def make_it_pytorch_11(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return (x<c).nonzero().t()

    # return the elements of x where m is True
    @staticmethod
    def make_it_pytorch_12(x: torch.Tensor, m: torch.BoolTensor) -> torch.Tensor:
        return x[m]

    # return the difference between consecutive elements of the sequence [x, y]
    @staticmethod
    def make_it_pytorch_extra_1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xy = torch.cat((x, y), dim=0)
        z = xy[1:] - xy[:-1]
        return z

    # find the number of elements in x that are equal (abs(x_i-y_j) < 1e-3) to at least one element in y
    @staticmethod
    def make_it_pytorch_extra_2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        y = y.view(1, -1)
        diffs = torch.abs(x - y)
        within_tolerance = diffs < 1e-3
        count = torch.sum(torch.any(within_tolerance, dim=1))
        return count