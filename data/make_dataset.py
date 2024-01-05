
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset

def load_and_process_data(data_path):
    # Load the corrupted MNIST training data
    for i in range(0, 6):
        # Assuming the corrupted MNIST files are in .pt format similar to standard MNIST
        train_images = torch.load(os.path.join(data_path, 'train_images_{}.pt'.format(i)))
        train_targets = torch.load(os.path.join(data_path, 'train_target_{}.pt'.format(i)))

        if i == 0:
            images = train_images
            targets = train_targets
        else:
            images = torch.cat((images, train_images), dim=0)
            targets = torch.cat((targets, train_targets), dim=0)

    # Convert to single tensor and normalize
    transform = transforms.Normalize((0.0,), (1.0,))
    train_images_normalized = transform(train_images)

    # Making sure the images have mean 0 and standard deviation 1
    mean = train_images_normalized.mean()
    std = train_images_normalized.std()
    train_images_normalized = (train_images_normalized - mean) / std

    # Load the corrupted MNIST test data
    test_images = torch.load(os.path.join(data_path, 'test_images.pt'))
    test_targets = torch.load(os.path.join(data_path, 'test_target.pt'))
    
    # Convert to single tensor and normalize
    test_images_normalized = transform(test_images)

    # Making sure the images have mean 0 and standard deviation 1
    mean = test_images_normalized.mean()
    std = test_images_normalized.std()
    test_images_normalized = (test_images_normalized - mean) / std

    return train_images_normalized, train_targets, test_images_normalized, test_targets

def save_processed_data(tr_images, tr_targets, ts_images, ts_targets, processed_data_path):
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)

    torch.save(tr_images, os.path.join(processed_data_path, 'processed_train_images.pt'))
    torch.save(tr_targets, os.path.join(processed_data_path, 'processed_train_targets.pt'))
    torch.save(ts_images, os.path.join(processed_data_path, 'processed_test_images.pt'))
    torch.save(ts_targets, os.path.join(processed_data_path, 'processed_test_targets.pt'))

def main():
    raw_data_path = './raw'
    processed_data_path = './processed'

    tr_images, tr_targets, ts_images, ts_targets = load_and_process_data(raw_data_path)
    save_processed_data(tr_images, tr_targets, ts_images, ts_targets, processed_data_path)

if __name__ == '__main__':
    main()
