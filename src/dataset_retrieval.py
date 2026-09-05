import os
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps

# Unseen classes for different datasets (ZS-SBIR evaluation)
UNSEEN_CLASSES = {
    "sketchy": [
        "bat", "cabin", "cow", "dolphin", "door", "giraffe", "helicopter",
        "mouse", "pear", "raccoon", "rhinoceros", "saw", "scissors",
        "seagull", "skyscraper", "songbird", "sword", "tree", "wheelchair",
        "windmill", "window"
    ],
    "sketchy_ext": [
        "bat", "cabin", "cow", "dolphin", "door", "giraffe", "helicopter",
        "mouse", "pear", "raccoon", "rhinoceros", "saw", "scissors",
        "seagull", "skyscraper", "songbird", "sword", "tree", "wheelchair",
        "windmill", "window"
    ],
    "sketchy_1": [
        "cup", "swan", "harp", "squirrel", "snail", "ray", "pineapple",
        "volcano", "rifle", "scissors", "parrot", "windmill", "teddy_bear",
        "tree", "wine_bottle", "deer", "chicken", "hotdog", "wheelchair",
        "tank", "umbrella", "butterfly", "camel", "horse", "bell"
    ],
    "sketchy_2": [
        "bat", "cabin", "cow", "dolphin", "door", "giraffe", "helicopter",
        "mouse", "pear", "raccoon", "rhinoceros", "saw", "scissors",
        "seagull", "skyscraper", "songbird", "sword", "tree", "wheelchair",
        "windmill", "window"
    ],
    "tuberlin": [
        "helicopter", "wrist-watch", "mermaid", "mosquito", "pear", "couch",
        "hammer", "purse", "house", "tennis-racket", "toilet", "panda",
        "butterfly", "mug", "wineglass", "motorbike", "eyeglasses",
        "hot air balloon", "screwdriver", "skull", "truck", "palm tree",
        "cell phone", "horse", "sailboat", "suv", "church", "floor lamp",
        "pipe (for smoking)", "tv"
    ],
    "quickdraw": [
        "airplane", "alarm_clock", "ant", "apple", "axe", "banana", "bat",
        "bear", "bee", "bench", "bicycle", "bread", "bus", "butterfly",
        "cactus", "cake", "camel", "candle", "car", "castle", "cat", "chair",
        "church", "couch", "cow", "crab", "crocodilian", "dolphin",
        "eyeglasses", "guitar"
    ]
}

# Retrieval metric protocol per dataset. map_k / p_k = 0 means "@all".
DATASET_METRICS = {
    "sketchy":     {"map_k": 200, "p_k": 200},
    "sketchy_ext": {"map_k": 200, "p_k": 200},
    "sketchy_1":   {"map_k": 200, "p_k": 200},
    "sketchy_2":   {"map_k": 200, "p_k": 200},
    "tuberlin":    {"map_k": 0,   "p_k": 100},
    "quickdraw":   {"map_k": 0,   "p_k": 200},
}

# kept for backward compatibility with code importing the flat list
unseen_classes = UNSEEN_CLASSES["sketchy_ext"]


def get_unseen_classes(opts):
    name = getattr(opts, 'dataset', 'sketchy_ext')
    if name not in UNSEEN_CLASSES:
        raise ValueError('unknown --dataset %r, expected one of %s'
                         % (name, sorted(UNSEEN_CLASSES)))
    return UNSEEN_CLASSES[name]


def get_metric_config(opts):
    """(map_k, p_k) for the dataset, unless overridden on the command line.

    0 means @all. --map_k / --p_k default to -1 = "use the dataset protocol".
    """
    default = DATASET_METRICS[getattr(opts, 'dataset', 'sketchy_ext')]
    map_k = int(getattr(opts, 'map_k', -1))
    p_k = int(getattr(opts, 'p_k', -1))
    return (default['map_k'] if map_k < 0 else map_k,
            default['p_k'] if p_k < 0 else p_k)

visualize_classes = [
    "cow",
    "raccoon",
    "scissors",
    "seagull",
    "sword",
    "tree",
]

class Sketchy(torch.utils.data.Dataset):

    def __init__(self, opts, transform, mode='train', used_cat=None, return_orig=False):

        self.opts = opts
        self.transform = transform
        self.return_orig = return_orig

        self.all_categories = os.listdir(os.path.join(self.opts.data_dir, 'sketch'))
        if '.ipynb_checkpoints' in self.all_categories:
            self.all_categories.remove('.ipynb_checkpoints')
            
        if self.opts.data_split > 0:
            np.random.shuffle(self.all_categories)
            if used_cat is None:
                self.all_categories = self.all_categories[:int(len(self.all_categories)*self.opts.data_split)]
            else:
                self.all_categories = list(set(self.all_categories) - set(used_cat))
        else:
            unseen = get_unseen_classes(self.opts)
            if mode == 'train':
                self.all_categories = list(set(self.all_categories) - set(unseen))
            else:
                self.all_categories = [c for c in unseen if c in self.all_categories]

        self.all_sketches_path = []
        self.all_photos_path = {}

        for category in self.all_categories:
            self.all_sketches_path.extend(glob.glob(os.path.join(self.opts.data_dir, 'sketch', category, '*')))
            self.all_photos_path[category] = glob.glob(os.path.join(self.opts.data_dir, 'photo', category, '*'))

        
    def __len__(self):
        return len(self.all_sketches_path)
        
    def __getitem__(self, index):
        filepath = self.all_sketches_path[index]                
        category = filepath.split(os.path.sep)[-2]
        filename = os.path.basename(filepath)
        
        neg_classes = self.all_categories.copy()
        neg_classes.remove(category)

        sk_path  = filepath
        img_path = np.random.choice(self.all_photos_path[category])
        neg_path = np.random.choice(self.all_photos_path[np.random.choice(neg_classes)])

        sk_data  = ImageOps.pad(Image.open(sk_path).convert('RGB'),  size=(self.opts.max_size, self.opts.max_size))
        img_data = ImageOps.pad(Image.open(img_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))
        neg_data = ImageOps.pad(Image.open(neg_path).convert('RGB'), size=(self.opts.max_size, self.opts.max_size))

        sk_tensor  = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)
        
        if self.return_orig:
            return (sk_tensor, img_tensor, neg_tensor, category, filename,
                sk_data, img_data, neg_data)
        else:
            return (sk_tensor, img_tensor, neg_tensor, category, filename)
    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return dataset_transforms

def normal_transform():
    dataset_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return dataset_transforms

class ValidDataset(torch.utils.data.Dataset):
    """ZS-SBIR evaluation set: unseen-class sketches (query) or photos (gallery).

    Generalized ZS-SBIR (--gzs=1): a --gzs_perc fraction of the SEEN-class
    photos is added to the gallery as distractors, labelled -1 so they are never
    relevant to any query. Queries stay unseen-class sketches. Sampling is
    seeded, so the gallery is identical across epochs and runs.
    """

    def __init__(self, args, mode='photo', categories=None):
        """categories: restrict to this class list instead of the unseen split.
        Used by the t-SNE path in experiments/LN_prompt.py, which labels points
        by index into its own 6-class visualize_classes list."""
        super(ValidDataset, self).__init__()
        self.args = args
        self.mode = mode
        self.transform = normal_transform()
        self.gzs_perc = float(getattr(args, 'gzs_perc', 0.2))
        self.seed = 42

        self.global_categories = os.listdir(os.path.join(self.args.data_dir, 'sketch'))
        if '.ipynb_checkpoints' in self.global_categories:
            self.global_categories.remove('.ipynb_checkpoints')

        # the unseen split of the configured dataset, restricted to what is on disk
        wanted = list(categories) if categories is not None else get_unseen_classes(self.args)
        self.categories = [c for c in wanted if c in self.global_categories]
        missing = [c for c in wanted if c not in self.global_categories]
        if missing:
            print('[ValidDataset] %d unseen classes of --dataset=%s are not on disk: %s'
                  % (len(missing), getattr(self.args, 'dataset', '?'), missing))

        subdir = 'photo' if self.mode == 'photo' else 'sketch'
        self.paths, self.labels = [], []
        for category in self.categories:
            paths = glob.glob(os.path.join(self.args.data_dir, subdir, category, '*'))
            self.paths.extend(paths)
            self.labels.extend([self.categories.index(category)] * len(paths))

        # generalized ZS-SBIR: seen-class photos as gallery distractors
        if self.mode == 'photo' and int(getattr(args, 'gzs', 0)) and self.gzs_perc > 0:
            seen = sorted(set(self.global_categories) - set(self.categories))
            rng = np.random.RandomState(self.seed)
            for category in seen:
                paths = sorted(glob.glob(os.path.join(self.args.data_dir, 'photo', category, '*')))
                if not paths:
                    continue
                n = int(round(len(paths) * self.gzs_perc))
                if n <= 0:
                    continue
                chosen = rng.choice(len(paths), size=n, replace=False)
                self.paths.extend([paths[i] for i in chosen])
                self.labels.extend([-1] * n)  # never relevant to an unseen query
            print('[ValidDataset] GZS gallery: %d unseen + %d seen distractors'
                  % (sum(1 for l in self.labels if l >= 0),
                     sum(1 for l in self.labels if l < 0)))

    def __getitem__(self, index):
        filepath = self.paths[index]
        image = ImageOps.pad(Image.open(filepath).convert('RGB'),
                             size=(self.args.max_size, self.args.max_size))
        return self.transform(image), self.labels[index]

    def __len__(self):
        return len(self.paths)
    
if __name__ == '__main__':
    from experiments.options import opts
    import tqdm

    dataset_transforms = Sketchy.data_transform(opts)
    dataset_train = Sketchy(opts, dataset_transforms, mode='train', return_orig=True)
    dataset_val = Sketchy(opts, dataset_transforms, mode='val', used_cat=dataset_train.all_categories, return_orig=True)

    idx = 0
    for data in tqdm.tqdm(dataset_val):
        continue
        (sk_tensor, img_tensor, neg_tensor, filename,
            sk_data, img_data, neg_data) = data

        canvas = Image.new('RGB', (224*3, 224))
        offset = 0
        for im in [sk_data, img_data, neg_data]:
            canvas.paste(im, (offset, 0))
            offset += im.size[0]
        canvas.save('output/%d.jpg'%idx)
        idx += 1
