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
But what will happen if we show it a pink Galah🦜\?
"""

ARTICLE= ("<p style='text-align: center'>"
          "<a href='https://tmabraham.github.io/blog/gradio_hf_spaces_tutorial'"
          "target='_blank'>Blog post</a></p>"
)

examples = ['cockatoo.jpg', 'rosella.jpg', 'magpie.jpg', 'galah.jpg']

gr.Interface(fn=predict, inputs=gr.Image(height=512, width=512),
             outputs=gr.Label(num_top_classes=3), title=TITLE,
             description=DESCRIPTION, article=ARTICLE,
             examples=examples).launch()
