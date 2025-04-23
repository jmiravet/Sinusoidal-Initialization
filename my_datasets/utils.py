import torch


def get_n_inputs_from_dataloader(dataloader, n):
    """
    Function to fetch n samples from a DataLoader.

    Args:
        dataloader (DataLoader): PyTorch DataLoader object.
        n (int): Number of samples to fetch.

    Returns:
        Tensor: A batch of `n` samples (images) and their corresponding labels.
    """
    if n == 0:
        return None, None
    all_samples = []
    all_labels = []

    # Accumulate samples until we have `n`
    for inputs, labels in dataloader:
        all_samples.append(inputs)
        all_labels.append(labels)
        
        # Check if we reached `n` samples
        if len(torch.cat(all_samples)) >= n:
            break
    
    # Concatenate all batches collected so far and limit to `n` samples
    samples_batch = torch.cat(all_samples)[:n]
    labels_batch = torch.cat(all_labels)[:n]
    
    return samples_batch, labels_batch

def get_reduced_dataset(dataset, n_samples):
    return torch.utils.data.random_split(dataset, [n_samples, len(dataset)-n_samples])[0]

def get_reduced_dataloader(dataloader, n_samples):
    shuffle = type(dataloader.sampler) == torch.utils.data.sampler.RandomSampler
    dataset = get_reduced_dataset(dataloader.dataset, n_samples=n_samples)
    return torch.utils.data.DataLoader(dataset, batch_size=dataloader.batch_size,
                                        shuffle=shuffle, num_workers=dataloader.num_workers, 
                                        prefetch_factor=dataloader.prefetch_factor, pin_memory=dataloader.pin_memory)

def concatenate_dataloader(dataloaders):
    shuffle = type(dataloaders[0].sampler) == torch.utils.data.sampler.RandomSampler
    datasets = []
    for data in dataloaders:
        datasets.append(data.dataset)
    dataset = torch.utils.data.ConcatDataset(datasets)
    return torch.utils.data.DataLoader(dataset, batch_size=dataloaders[0].batch_size,
                                        shuffle=shuffle, num_workers=dataloaders[0].num_workers, 
                                        prefetch_factor=dataloaders[0].prefetch_factor, pin_memory=dataloaders[0].pin_memory)


def get_balanced_batch_images(dataloader, batch_size, num_classes=10):
    if batch_size == 0:
        return None
    samples_per_class = batch_size / num_classes
    
    batch_samples = []
    batch_labels = [0] * num_classes
    total_samples = 0

    for inputs, labels in dataloader:
        for input, label in zip(inputs, labels):
            if batch_labels[label] < samples_per_class:
                batch_samples.append(input)
                batch_labels[label] += 1
                total_samples += 1
        
        if total_samples >= batch_size:
            break
    
    # Concatenate all batches collected so far and limit to `n` samples
    batch_samples = torch.stack(batch_samples)[:batch_size]
    
    return batch_samples



