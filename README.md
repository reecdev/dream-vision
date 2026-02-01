# dream-vision
Real time video stylization via. StableDiffusionImg2ImgPipeline.

## Prerequisities
You must have the HuggingFace diffusers library, PIL, and Flask installed. To install PIL and Flask, run the following command:
```
pip install flask pillow
```

To install the HuggingFace diffusers library, follow your device/OS specific guide.

## Running the model
To run the model, simply run `main.py` and then open the link provided by Flask on your device. It should be accessible on other devices on the same Wi-Fi as well if you configure your firewall correctly.
When opening the link, it will show a "Your Connection Is Not Private" warning. You can simply ignore this.
