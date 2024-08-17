import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class RLencoder(torch.nn.Module):
    def __init__(self, time_window, fire_window_ratio):
        super(RLencoder, self).__init__()
        self.time_window = time_window
        self.fire_window = int(time_window * fire_window_ratio)
        print("time_window: ", self.time_window)
        print("fire_window: ", self.fire_window)

    def forward(self, x):
        batch_size, _, height, width = x.shape
        spikes = torch.zeros(batch_size, height, width, self.time_window).to(x.device)

        # gray = to_grayscale(x)
        # max_latency = 100
        var = variance_map(x, 5)
        latency = calculate_latency(var, self.time_window, self.fire_window, mode='log')

        for t in range(self.time_window):
            firing_mask = ((t >= latency) & (t < latency + self.fire_window)).float()
            spike_prob = torch.rand(batch_size, height, width).to(x.device)
            spikes[:, :, :, t] = (spike_prob < x.squeeze(1)) * firing_mask

        return spikes

# def to_grayscale(image):
#     return torch.mean(image, dim=1, keepdim=True)

def variance_map(image, kernel_size):
    # kernel_size : odd
    
    pad_size = kernel_size // 2
    padded_image = F.pad(image, pad=(pad_size, pad_size, pad_size, pad_size), mode='reflect')
    
    local_mean = F.avg_pool2d(padded_image, kernel_size, stride=1, padding=0)
    
    squared_image = padded_image ** 2
    local_mean_squared = F.avg_pool2d(squared_image, kernel_size, stride=1, padding=0)
    
    variance = local_mean_squared - local_mean ** 2
    # variance = torch.mean(variance, dim=1, keepdim=True)
    
    return variance

def calculate_latency(div_image, time_window, fire_window, mode='linear'):
    min_div, max_div = div_image.min(), div_image.max()
    normalized_div = (div_image - min_div) / (max_div - min_div)
    if mode == "linear":
        latency = (time_window - fire_window) * (1 - normalized_div)
        return latency
    elif mode == "log":
        latency = 1/(normalized_div + 1/(time_window - fire_window))
        return latency

def plot_raster(spike_trains, num_neurons):
    spike_trains = spike_trains.cpu().numpy()
    plt.figure(figsize=(12, 8))
    
    for neuron_idx in range(num_neurons):
        neuron_spikes = spike_trains[0, neuron_idx // 28, neuron_idx % 28]
        spike_times = neuron_spikes.nonzero()[0]
        plt.vlines(spike_times, neuron_idx + 0.5, neuron_idx + 1.5)
    
    plt.axis([0, 100, 0, 784])
    plt.xlabel('Time')
    plt.ylabel('Neuron Index')
    plt.title('Raster Plot of Spikes')
    plt.show()