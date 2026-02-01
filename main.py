import secrets, string, torch, base64, io
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderTiny, DDIMScheduler
from flask import Flask, send_file, request, abort
from io import BytesIO

model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id, 
    variant="fp16", 
    torch_dtype=torch.float16
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = None
pipe.requires_safety_checker = False
pipe.vae = AutoencoderTiny.from_pretrained(
    "madebyollin/taesd", 
    torch_dtype=torch.float16
).to(device)
pipe.enable_model_cpu_offload()

def toImg(im):
    imio = io.BytesIO()
    im.save(imio, format="WebP", quality=80)
    imio.seek(0)
    return imio

def stylize(im, prompt):
    image = im.convert("RGB").resize((512, 512))
    generator = torch.Generator(device=device).manual_seed(42)

    return toImg(pipe(
        prompt=prompt,
        image=image,
        strength=0.6,
        guidance_scale=8.5,
        num_inference_steps=15,
        generator=generator
    ).images[0])

def createKey():
    chs = string.ascii_letters + string.digits
    return ''.join(secrets.choice(chs) for _ in range(10))

keys = {}

app = Flask(__name__)

@app.route("/")
def main():
    return send_file("web/index.html")

@app.route("/register")
def register():
    key = createKey()
    keys[key] = request.args.get("q")
    return key

@app.route("/completions", methods=['POST'])
def infer():
    data = request.get_json()
    if data.get("key") in keys:
        return send_file(stylize(Image.open(BytesIO(base64.b64decode(data.get("image").split(",")[-1]))), keys[data.get("key")]), mimetype="image/webp")
    else:
        return abort(401)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=443, ssl_context='adhoc')