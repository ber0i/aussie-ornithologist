import pathlib
import gradio as gr
from fastai.vision.all import (
    load_learner,
    PILImage
)

pathlib.WindowsPath = pathlib.PosixPath

learn = load_learner('export.pkl')
labels = learn.dls.vocab


def predict(img):
    '''
    Predict the bird in the given image.
    Outputs the bird name and how secure the model is.
    '''

    img = PILImage.create(img)
    _, _, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


TITLE = r"""
<h1>Australian Bird Classifier</h1>
"""

DESCRIPTION = r"""
This app can tell apart my three favorite Australian birds, the crimson rosella, the cockatoo, and the Australian magpie🥳.<br>
But what will happen if we show it a pink Galah🦜\?<br>
Image credits for cockatoo, crimson rosella, and magpie: JJ Harrison (https://www.jjharrison.com.au/), licensed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). No changes were made.<br>
Image credits for galah: Charles J. Sharp (https://www.sharpphotography.co.uk/), licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). No changes were made.
"""

ARTICLE= ("<p style='text-align: center'>"
          "<a href='https://tmabraham.github.io/blog/gradio_hf_spaces_tutorial'"
          "target='_blank'>Blog post</a></p>"
)

examples = ['images/cockatoo.jpg', 'images/crimson_rosella.jpg', 'images/magpie.jpg', 'images/galah.jpg']

gr.Interface(fn=predict, inputs=gr.Image(height=512, width=512),
             outputs=gr.Label(num_top_classes=3), title=TITLE,
             description=DESCRIPTION, article=ARTICLE,
             examples=examples).launch()
