from pathlib import Path
from torchvision import datasets, transforms
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def get_cifar10_datasets():
    transform_vae = transforms.ToTensor()

    # Normalisation + augmentation pour le classifieur (ResNet-18)
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)

    transform_train_cls = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    train_vae = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_vae)
    train_cls = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_train_cls)
    test_ds = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_test)

    return train_vae, train_cls, test_ds

if __name__ == "__main__":
    train_vae, train_cls, test_ds = get_cifar10_datasets()
    print(f"Train : {len(train_vae)} | Test : {len(test_ds)}")
