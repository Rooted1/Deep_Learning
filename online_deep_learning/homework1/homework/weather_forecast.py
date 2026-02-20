# citation: used copilot and chatgpt for guidance

from typing import Tuple

import torch


class WeatherForecast:
    def __init__(self, data_raw: list[list[float]]):
        """
        You are given a list of 10 weather measurements per day.
        Save the data as a PyTorch (num_days, 10) tensor,
        where the first dimension represents the day,
        and the second dimension represents the measurements.
        """
        self.data = torch.as_tensor(data_raw).view(-1, 10)

    def find_min_and_max_per_day(self) -> Tuple[torch.Tensor, torch.Tensor]:
        min_temps = self.data.min(dim=1).values  # Minimum temperature per day
        max_temps = self.data.max(dim=1).values  # Maximum temperature per day
        return min_temps, max_temps

    def find_the_largest_drop(self) -> torch.Tensor:
        daily_avg_temps = self.data.mean(dim=1) # Average temperature per day
        temp_drops = daily_avg_temps[1:] - daily_avg_temps[:-1]  # Day-to-day temperature changes
        largest_drop = temp_drops.min()  # Largest drop in temperature
        return largest_drop

    def find_the_most_extreme_day(self) -> torch.Tensor:
        daily_avg_temps = self.data.mean(dim=1, keepdim=True)  # Average temperature per day (num_days,). use keepdim for broadcasting (num_days, 1)
        deviations = self.data - daily_avg_temps  # Deviations from the average (num_days, 10)
        _, i = deviations.abs().max(dim=1) # Maximum deviation per day (num_days,)
        return self.data[torch.arange(self.data.size(0)), i]

    def max_last_k_days(self, k: int) -> torch.Tensor:
        last_k_days = self.data[-k:]  # Last k days (k, 10)
        max_temp_last_k_days = last_k_days.max(dim=1).values  # Maximum temperature over the last k days (k,)
        return max_temp_last_k_days

    def predict_temperature(self, k: int) -> torch.Tensor:
        last_k_days = self.data[-k:]  # Last k days (k, 10)
        daily_avg_temps = last_k_days.mean(dim=1)  # Average temperature per day for the last k days (k,)
        predicted_temp = daily_avg_temps.mean(dim=0)  # Overall average temperature as prediction for the next day. scalar tensor
        return predicted_temp

    def what_day_is_this_from(self, t: torch.FloatTensor) -> torch.LongTensor:
        distances = (self.data - t).abs().sum(dim=1)  # (num_days,)
        day_index = distances.argmin()  # Index of the day with the smallest distance
        return day_index
