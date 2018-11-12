import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)


class ChestXrayDataSet(Dataset):

    def __init__(self,
                 data_dir,
                 reports_file,
                 transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """

        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name= items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform
        self.img_size = img_size #if isinstance(img_size,int) else 256
        self.img_type = img_type
        self.img_format = img_format

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        if self.img_format == 'npy':
            image_name = '.'.join([image_name,str(self.img_size),self.img_format])
            #image = Image.fromarray(np.load(image_name))
            image = np.expand_dims(np.load(image_name),0)
        else:
            image = Image.open(image_name).convert(self.img_type)

        label = self.labels[index]
        if self.transform is not None:
            if self.img_format == 'npy':
                image = self.transform(torch.FloatTensor(image))
                # image = self.transform(image)
            else:
                image = self.transform(image)
        return image,  torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)

def get_xforms():
    global args
    # crop_size = 256 if args.augment else int(args.img_size * args.crop)
    img_size = int(args.img_size)
    crop_size = int(img_size * args.crop)
    # normalize = transforms.Normalize([0.485, 0.456, 0.406],
    #                                  [0.229, 0.224, 0.225])
    # normalize = transforms.Normalize([np.mean([0.485, 0.456, 0.406])],
    #                                  [np.mean([0.229, 0.224, 0.225])])
    normalize = transforms.Normalize([0.485],
                                     [0.229])
    # print('size {}\tcrop: {}'.format(type(img_size),type(crop_size)))
    if args.img_format == 'npy':
        if args.augment:
            return {
                'train': transforms.Compose([
                    transforms.ToPILImage(),
                    # transforms.RandomResizedCrop(crop_size),
                    # transforms.RandomHorizontalFlip(),
                    transforms.Resize(img_size),
                    transforms.FiveCrop(crop_size),
                    transforms.Lambda
                    (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Lambda
                    (lambda crops: torch.stack([normalize(crop) for crop in crops]))

                ]),
                'test': transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(crop_size),
                    transforms.ToTensor(),
                    normalize
                ]),
                'val': transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(img_size),
                    transforms.FiveCrop(crop_size),
                    # transforms.RandomHorizontalFlip(),
                    transforms.Lambda
                    (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Lambda
                    (lambda crops: torch.stack([normalize(crop) for crop in crops]))

                ])
            }
        else:
            return {
                'train': transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]),
                'test': transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(crop_size),
                    transforms.ToTensor(),
                    normalize
                ]),
                'val': transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomResizedCrop(crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])
            }
    else:
        if args.augment:

            return {

                'train': transforms.Compose([
                    # transforms.RandomResizedCrop(crop_size),
                    # transforms.RandomHorizontalFlip(),
                    transforms.Resize(img_size),
                    transforms.FiveCrop(crop_size),
                    transforms.Lambda
                    (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Lambda
                    (lambda crops: torch.stack([normalize(crop) for crop in crops]))

                ]),
                'test': transforms.Compose([
                    transforms.Resize(crop_size),
                    transforms.ToTensor(),
                    normalize
                ]),
                'val': transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.FiveCrop(crop_size),
                    transforms.Lambda
                    (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    transforms.Lambda
                    (lambda crops: torch.stack([normalize(crop) for crop in crops]))

                ])
            }
        else:
            return {
                'train': transforms.Compose([
                    transforms.RandomResizedCrop(crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ]),
                'test': transforms.Compose([
                    transforms.Resize(crop_size),
                    transforms.ToTensor(),
                    normalize
                ]),
                'val': transforms.Compose([
                    transforms.RandomResizedCrop(crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])
            }

def get_loaders():
    global args
    image_files_list = {'train':args.train_list,
                        'val':args.val_list,
                        'test':args.test_list}

    # crop_size = int(img_size[0] * args.crop)


    kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}

    data_transforms = get_xforms()

    image_datasets = {key:ChestXrayDataSet(data_dir=args.data,
                                           image_list_file=image_files_list[key],
                                           transform=data_transforms[key],
                                           img_type=args.img_type,
                                           img_size=int(args.img_size),
                                           img_format=args.img_format)
                      for key in data_transforms.keys()}
    if args.distributed:
        samplers = {key:DistributedSampler(dataset=image_datasets[key])
                    for key in image_datasets.keys()}
    else:
        samplers = {key:None for key in image_datasets.keys()}
    dataloaders = {key:DataLoader(dataset=image_datasets[key],
                                  sampler=samplers[key],
                                  batch_size=args.batch[key],
                                  shuffle=samplers[key] is None and key is 'train',
                                  **kwargs)
                   for key in image_datasets.keys()}

    return dataloaders

