from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as transforms
from torchvision.transforms import v2

transform =v2.Compose([
    v2.RandomResizedCrop((227, 227), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_2 = transforms.Compose([
    transforms.Resize((227, 227))
])

# idx_str_labels_map = {
#     0: "n01440764",
#     1: "n02102040",
#     2: "n02979186",
#     3: "n03000684",
#     4: "n03028079",
#     5: "n03394916",
#     6: "n03417042",
#     7: "n03425413",
#     8: "n03445777",
#     9: "n03888257",
# }

idx_str_labels_map = {
    0: "n02086240",
    1: "n02087394",
    2: "n02088364",
    3: "n02089973",
    4: "n02093754",
    5: "n02096294",
    6: "n02099601",
    7: "n02105641",
    8: "n02111889",
    9: "n02115641",
}

str_idx_labels_map = {v: k for k, v in idx_str_labels_map.items()}

def transform_image(image):
    return transform(image)

class CustomDataset(Dataset):
    def __init__(self, size, start_idx, df, root_path):
        self.transform = transform_image
        self.df = df
        self.size = size
        self.start_idx = start_idx
        self.root_path = root_path

    def __len__(self):
        """

        """
        return self.size

    def __getitem__(self, idx):
        """
            Args:
              idx (int): Index
          Returns:
              tuple: (sample, target) where target is class_index of the target class.
        """
        index = self.start_idx + idx
        image_path = self.df.at[index, 'path']
        image = read_image(self.root_path / image_path, ImageReadMode.RGB).float()
        label = self.df.at[index, 'noisy_labels_0']
        return transform_2(image), self.transform(image), str_idx_labels_map[label]