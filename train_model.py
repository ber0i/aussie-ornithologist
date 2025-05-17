from duckduckgo_search import DDGS
from fastcore.all import L, Path
from fastdownload import download_url
from fastai.vision.all import (
    download_images,
    resize_images,
    verify_images,
    get_image_files,
    DataBlock,
    ImageBlock,
    CategoryBlock,
    RandomSplitter,
    parent_label,
    Resize,
    vision_learner,
    resnet18,
    error_rate,
    PILImage
)


def search_images(term, max_images=100):
    'Search images with search term "term"'

    print(f"Searching for '{term}'")
    with DDGS() as ddgs:
        return L(ddgs.images(term, max_results=max_images)).itemgot('image')


searches = 'crimson rosella', 'cockatoo', 'australian magpie'
path = Path('australian_birds')


for o in searches:
    dest = path/o
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    resize_images(path/o, max_size=400, dest=path/o)

failed = verify_images(get_image_files(path))
failed.map(Path.unlink)

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
learn.export()

urls = search_images('cockatoo', max_images=1)
DEST = 'cockatoo.jpg'
download_url(urls[0], DEST, show_progress=False)
bird_prediction, _, probs = learn.predict(PILImage.create('cockatoo.jpg'))
