import torchvision

prepr = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()
])
voc = torchvision.datasets.VOCDetection(root='data/train',
                                        image_set='train',
                                        transform=prepr)