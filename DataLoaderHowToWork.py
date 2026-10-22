from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):

    def __init__(self):
        self.data = [10, 20, 30, 40, 50, 60]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        print("正在读取索引：", index)
        return self.data[index]


dataset = MyDataset()

loader = DataLoader(
    dataset=dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

for batch in loader:
    print("当前 Batch：", batch)