FROM python:3.11-slim

# FFmpeg for video rendering
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ========= backend structure =========
RUN mkdir -p /app/backend/models /app/backend/services /app/backend/storage /app/frontend
RUN touch /app/backend/__init__.py /app/backend/models/__init__.py /app/backend/services/__init__.py

# -------- requirements --------
RUN printf "%s\n" \
  "fastapi==0.112.2" \
  "uvicorn[standard]==0.30.6" \
  "python-dotenv==1.0.1" \
  "pydantic==2.8.2" \
  "requests==2.32.3" \
  "Pillow==10.4.0" \
  "python-multipart" \
  > /app/backend/requirements.txt
  

RUN python -m pip install --upgrade pip && pip install -r /app/backend/requirements.txt

# -------- backend/models/schemas.py --------
RUN mkdir -p /app/backend/models && cat <<'PY' > /app/backend/models/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class ScriptSection(BaseModel):
    title: str
    text: str
    duration_sec: int = 6
    broll_keywords: Optional[str] = None

class ScriptResponse(BaseModel):
    title: str
    hook: str
    sections: List[ScriptSection]

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speed: Optional[float] = None

class TTSResponse(BaseModel):
    audio_path: str

class ImageGenItem(BaseModel):
    prompt: str
    duration_sec: int = 6

class ImageGenRequest(BaseModel):
    items: List[ImageGenItem]

class ImageGenResponse(BaseModel):
    image_paths: List[str]

class RenderRequest(BaseModel):
    audio_path: str
    image_paths: List[str]
    output_name: str = "output.mp4"
PY

# -------- backend/services/storage.py --------
RUN mkdir -p /app/backend/services && cat <<'PY' > /app/backend/services/storage.py
import os, uuid
from pathlib import Path

ROOT = Path(os.getenv("STORAGE_DIR", Path(__file__).resolve().parent.parent / "storage"))
AUDIO = ROOT / "audio"
IMAGES = ROOT / "images"
VIDEO  = ROOT / "video"
TMP    = ROOT / "tmp"

for p in [ROOT, AUDIO, IMAGES, VIDEO, TMP]:
    p.mkdir(parents=True, exist_ok=True)

def unique_name(ext: str) -> str:
    return f"{uuid.uuid4().hex}.{ext.lstrip('.')}"

def save_bytes(data: bytes, folder: Path, ext: str) -> str:
    name = unique_name(ext)
    path = folder / name
    with open(path, "wb") as f:
        f.write(data)
    return str(path)

def abs_path(rel: str) -> str:
    p = Path(rel)
    if p.is_absolute():
        return str(p)
    return str(ROOT / rel)
PY

# -------- backend/services/config.py --------
RUN cat <<'PY' > /app/backend/services/config.py
import os

# Rescue Stories presets (voice & thumbnail baked in)
DEFAULT_TTS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
DEFAULT_TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))

DEFAULT_OUTPUT_NAME = "rescue_story.mp4"
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
PY

# -------- backend/services/llm.py --------
RUN cat <<'PY' > /app/backend/services/llm.py
import random
from typing import Dict, Any

def _stub_keywords(topic: str):
    base = [
        "rain-soaked alley", "cardboard box", "shivering kitten", "gentle hands",
        "towel and shelter", "first warm meal", "tiny paws", "big hopeful eyes"
    ]
    random.shuffle(base)
    return base[:3]

def generate_script(topic: str) -> Dict[str, Any]:
    sections = []
    skeleton = [
        ("The problem", "Describe the risky situation and environment."),
        ("The discovery", "Who found the animal and a small detail to care about."),
        ("The rescue", "Action, sounds, and sensory details of rescuing."),
        ("Recovery", "Warmth, first food, safe place and hopeful tone.")
    ]
    for title, hint in skeleton:
        keywords = ", ".join(_stub_keywords(topic))
        sections.append({
            "title": title,
            "text": f"{hint} Topic: {topic}. Keep it human and compassionate.",
            "duration_sec": 8,
            "broll_keywords": keywords
        })
    return {
        "title": f"{topic}: A Rescue Story",
        "hook": f"A tiny life was moments from being lost—until someone stopped for {topic}.",
        "sections": sections
    }
PY

# -------- backend/services/tts.py --------
RUN cat <<'PY' > /app/backend/services/tts.py
import os, math, wave, struct, tempfile, requests
from .storage import AUDIO, save_bytes
from . import config

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

def _sine_stub(seconds: float = 8.0, freq: float = 220.0, volume: float = 0.1) -> bytes:
    framerate = 22050
    frames = int(seconds * framerate)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wf = wave.open(tmp.name, "w")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        for i in range(frames):
            val = int(volume * 32767.0 * math.sin(2*math.pi*freq*(i/framerate)))
            wf.writeframesraw(struct.pack('<h', val))
        wf.close()
        with open(tmp.name, "rb") as f:
            return f.read()

def synthesize_tts(text: str, voice: str = None, speed: float = None):
    voice_id = config.DEFAULT_TTS_VOICE_ID
    _ = speed or config.DEFAULT_TTS_SPEED
    if ELEVENLABS_API_KEY:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {"xi-api-key": ELEVENLABS_API_KEY, "accept": "audio/mpeg", "content-type": "application/json"}
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "style": 0.0, "use_speaker_boost": True}
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        path = save_bytes(r.content, AUDIO, ".mp3")
        return {"audio_path": path}
    else:
        wav = _sine_stub(seconds=min(5 + len(text)//20, 150))
        path = save_bytes(wav, AUDIO, ".wav")
        return {"audio_path": path}
PY

# -------- backend/services/images.py --------
RUN cat <<'PY' > /app/backend/services/images.py
import io, textwrap
from PIL import Image, ImageDraw, ImageFont
from .storage import IMAGES, save_bytes

def _stub_image(prompt: str, size=(1280, 720)) -> bytes:
    img = Image.new("RGB", size, (220, 220, 235))
    draw = ImageDraw.Draw(img)
    wrapped = textwrap.fill(prompt, width=32)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.multiline_text((40, 40), wrapped, fill=(20, 20, 20), font=font, spacing=4)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def generate_images(prompts):
    paths = []
    for p in prompts:
        png = _stub_image(f"Cinematic rescue: {p}")
        paths.append(save_bytes(png, IMAGES, ".png"))
    return paths
PY

# -------- backend/services/thumbnail.py --------
RUN cat <<'PY' > /app/backend/services/thumbnail.py
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import io
from .storage import IMAGES, save_bytes

def make_thumbnail(title: str, subject: str = "Rescued kitten", size=(1280,720)) -> str:
    w,h = size
    img = Image.new("RGB", size, (245, 250, 255))
    draw = ImageDraw.Draw(img)

    v = Image.new("L", (w,h), 0)
    dv = ImageDraw.Draw(v)
    dv.ellipse((-int(w*0.1), -int(h*0.2), int(w*1.1), int(h*1.2)), fill=255)
    v = v.filter(ImageFilter.GaussianBlur(120))
    img.paste(Image.new("RGB",(w,h),(230,235,255)), (0,0), v)

    try:
        font_big = ImageFont.truetype("DejaVuSans-Bold.ttf", 72)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 36)
    except:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()

    words = title.split()
    lines, line = [], ""
    for wd in words:
        test = (line + " " + wd).strip()
        if len(test) > 18:
            lines.append(line)
            line = wd
        else:
            line = test
    if line: lines.append(line)

    y = 180
    for ln in lines[:3]:
        draw.text((80,y), ln, fill=(20,22,30), font=font_big, stroke_width=2, stroke_fill=(255,255,255))
        y += 86

    tag = f"RESCUE • {subject.upper()}"
    text_w = draw.textlength(tag, font=font_small)
    draw.rounded_rectangle((80, y+10, 80 + text_w + 36, y+60), 18, fill=(20,22,30))
    draw.text((98,y+18), tag, fill=(245,250,255), font=font_small)

    b = io.BytesIO()
    img.save(b, format="PNG")
    return save_bytes(b.getvalue(), IMAGES, ".png")
PY

# -------- backend/services/render.py --------
RUN cat <<'PY' > /app/backend/services/render.py
import subprocess
from pathlib import Path
from .storage import VIDEO

def _check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False

def _probe_audio_seconds(audio_path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, check=True
        )
        return float(r.stdout.strip())
    except Exception:
        return 10.0

def render_slideshow(image_paths, audio_path, output_name="output.mp4"):
    if not _check_ffmpeg():
        raise RuntimeError("FFmpeg is not installed in container.")
    out_path = str(Path(VIDEO) / output_name)
    total = _probe_audio_seconds(audio_path)
    per = max(total / max(1, len(image_paths)), 2.0)

    inputs, filters = [], []
    for idx, img in enumerate(image_paths):
        inputs += ["-loop","1","-t", f"{per:.3f}","-i", img]
        filters.append(f"[{idx}:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1[v{idx}]")
    filtergraph = ";".join(filters) + ";" + "".join([f"[v{i}]" for i in range(len(image_paths))]) + f"concat=n={len(image_paths)}:v=1:a=0[v]"

    cmd = ["ffmpeg", "-y"] + inputs + ["-i", audio_path,
           "-filter_complex", filtergraph, "-map","[v]","-map",f"{len(image_paths)}:a",
           "-c:v","libx264","-c:a","aac","-shortest","-pix_fmt","yuv420p", out_path]

    subprocess.run(cmd, check=True)
    return out_path
PY

# -------- backend/app.py --------
RUN cat <<'PY' > /app/backend/app.py
import os
from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.models.schemas import ScriptResponse, TTSRequest, TTSResponse, ImageGenRequest, ImageGenResponse, RenderRequest
from backend.services.llm import generate_script
from backend.services.tts import synthesize_tts
from backend.services.images import generate_images
from backend.services.thumbnail import make_thumbnail
from backend.services.render import render_slideshow
from backend.services.storage import abs_path

APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))

app = FastAPI(title="TubeGen Cloud")
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

@app.post("/generate-script", response_model=ScriptResponse)
def gen_script(topic: str = Form(...)):
    return generate_script(topic)

@app.post("/tts", response_model=TTSResponse)
def tts(req: TTSRequest):
    return synthesize_tts(req.text, voice=req.voice, speed=req.speed)

@app.post("/images", response_model=ImageGenResponse)
def images(req: ImageGenRequest):
    paths = generate_images([i.prompt for i in req.items])
    return {"image_paths": paths}

@app.post("/thumbnail")
def thumbnail(title: str = Form(...), subject: str = Form("Rescued kitten")):
    path = make_thumbnail(title, subject)
    return {"thumbnail_path": path}

@app.post("/render")
def render(req: RenderRequest):
    path = render_slideshow(req.image_paths, req.audio_path, req.output_name)
    return {"output_path": path}

@app.get("/download")
def download(path: str):
    real = abs_path(path)
    if not os.path.exists(real):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(real, filename=os.path.basename(real))

@app.post("/oneclick")
def oneclick(topic: str = Form(...)):
    script = generate_script(topic)
    tts_text = script["hook"] + "\\n\\n" + "\\n\\n".join(s["text"] for s in script["sections"])
    audio_path = synthesize_tts(tts_text)["audio_path"]
    prompts = [s.get("broll_keywords") or s["title"] for s in script["sections"]]
    image_paths = generate_images(prompts)
    thumb_path = make_thumbnail(script["title"])
    output_path = render_slideshow(image_paths, audio_path, "rescue_story.mp4")
    return {
        "script": script,
        "audio_path": audio_path,
        "image_paths": image_paths,
        "thumbnail": thumb_path,
        "output_path": output_path
    }
PY

# -------- frontend/index.html --------
RUN cat <<'HTML' > /app/frontend/index.html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>TubeGen Cloud — Rescue Stories</title>
  <link rel="stylesheet" href="/styles.css"/>
</head>
<body>
  <div class="container">
    <h1>TubeGen Cloud — Rescue Stories</h1>

    <section class="card">
      <h2>1) Generate Script</h2>
      <button onclick="oneClick()">✨ One-Click Render (Rescue Story)</button>
      <input id="topic" placeholder="Enter topic (e.g., Stray kitten found in rain)"/>
      <button onclick="genScript()">Generate</button>
      <pre id="scriptOut"></pre>
    </section>

    <section class="card">
      <h2>2) Voiceover (TTS)</h2>
      <textarea id="ttsText" rows="5" placeholder="Paste script text..."></textarea>
      <button onclick="genTTS()">Synthesize TTS</button>
      <div id="ttsOut"></div>
    </section>

    <section class="card">
      <h2>3) Images</h2>
      <textarea id="imgPrompts" rows="5" placeholder="Enter prompts (one per line)"></textarea>
      <button onclick="genImages()">Generate Images</button>
      <div id="imgOut" class="grid"></div>
    </section>

    <section class="card">
      <h2>4) Render Video</h2>
      <input id="audioPath" placeholder="Audio path from TTS"/>
      <input id="outputName" placeholder="output.mp4" value="rescue_story.mp4"/>
      <button onclick="renderVideo()">Render</button>
      <div id="download"></div>
    </section>

    <section class="card">
      <h2>5) Thumbnail Generator (preset style)</h2>
      <input id="thumbTitle" placeholder="Title for thumbnail (auto if One-Click)"/>
      <input id="thumbSubject" placeholder="Subject (e.g., Rescued kitten)"/>
      <button onclick="makeThumb()">Generate Thumbnail</button>
      <div id="thumbOut" class="thumb"></div>
    </section>
  </div>

  <script src="/app.js"></script>
</body>
</html>
HTML

# -------- frontend/styles.css --------
RUN cat <<'CSS' > /app/frontend/styles.css
*{box-sizing:border-box}
body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:#0f1115;color:#eaeef2}
.container{max-width:960px;margin:40px auto;padding:0 16px}
h1{font-size:28px;margin-bottom:16px}
.card{background:#151821;border:1px solid #22263a;border-radius:16px;padding:16px;margin:16px 0;box-shadow:0 8px 20px rgba(0,0,0,0.25)}
.card h2{margin-top:0}
input,textarea,button{width:100%;margin:8px 0;padding:10px;border-radius:12px;border:1px solid #2a3148;background:#0d1017;color:#eaeef2}
button{cursor:pointer;font-weight:600}
pre{background:#0b0d13;border-radius:12px;padding:12px;overflow:auto;max-height:260px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px}
.grid img{width:100%;border-radius:12px;border:1px solid #2a3148}
CSS

# -------- frontend/app.js --------
RUN cat <<'JS' > /app/frontend/app.js
async function genScript(){
  const topic = document.getElementById('topic').value.trim();
  if(!topic) { alert("Enter topic"); return; }
  const fd = new FormData();
  fd.append("topic", topic);
  const r = await fetch(`/generate-script`, { method:"POST", body: fd });
  const j = await r.json();
  document.getElementById('scriptOut').textContent = JSON.stringify(j, null, 2);
  const full = [j.hook, ...j.sections.map(s => s.text)].join("\\n\\n");
  document.getElementById('ttsText').value = full;
  document.getElementById('imgPrompts').value = j.sections.map(s => `${s.title}: ${s.broll_keywords || s.text}`).join("\\n");
}

async function genTTS(){
  const text = document.getElementById('ttsText').value.trim();
  if(!text){ alert("Paste text for TTS"); return; }
  const r = await fetch(`/tts`, { method:"POST", headers:{ "Content-Type":"application/json" }, body: JSON.stringify({ text }) });
  const j = await r.json();
  document.getElementById('audioPath').value = j.audio_path;
  const audioUrl = `/download?path=${encodeURIComponent(j.audio_path)}`;
  document.getElementById('ttsOut').innerHTML = `<audio controls src="${audioUrl}"></audio><div><code>${j.audio_path}</code></div>`;
}

async function genImages(){
  const prompts = document.getElementById('imgPrompts').value.split("\\n").map(s => s.trim()).filter(Boolean);
  if(prompts.length === 0){ alert("Add at least one prompt"); return; }
  const r = await fetch(`/images`, { method:"POST", headers:{ "Content-Type":"application/json" }, body: JSON.stringify({ items: prompts.map(p => ({ prompt: p })) }) });
  const j = await r.json();
  window._imagePaths = j.image_paths;
  const out = document.getElementById('imgOut');
  out.innerHTML = "";
  for(const p of j.image_paths){
    const url = `/download?path=${encodeURIComponent(p)}`;
    const img = document.createElement('img');
    img.src = url;
    out.appendChild(img);
  }
}

async function renderVideo(){
  const audio_path = document.getElementById('audioPath').value.trim();
  const output_name = document.getElementById('outputName').value.trim() || "rescue_story.mp4";
  const image_paths = (window._imagePaths || []);
  if(!audio_path || image_paths.length === 0){
    alert("Generate TTS and Images first");
    return;
  }
  const r = await fetch(`/render`, { method:"POST", headers:{ "Content-Type":"application/json" }, body: JSON.stringify({ audio_path, image_paths, output_name }) });
  const j = await r.json();
  const url = `/download?path=${encodeURIComponent(j.output_path)}`;
  document.getElementById('download').innerHTML = `<a href="${url}" download>⬇️ Download Video</a>`;
}

async function oneClick(){
  const topic = document.getElementById('topic').value.trim() || "Stray kitten found in a storm";
  const fd = new FormData();
  fd.append("topic", topic);
  const r = await fetch(`/oneclick`, { method:"POST", body: fd });
  const j = await r.json();
  document.getElementById('scriptOut').textContent = JSON.stringify(j.script, null, 2);
  document.getElementById('audioPath').value = j.audio_path;
  const audioUrl = `/download?path=${encodeURIComponent(j.audio_path)}`;
  document.getElementById('ttsOut').innerHTML = `<audio controls src="${audioUrl}"></audio>`;
  window._imagePaths = j.image_paths;
  const out = document.getElementById('imgOut'); out.innerHTML = "";
  for(const p of j.image_paths){
    const url = `/download?path=${encodeURIComponent(p)}`;
    const img = document.createElement('img'); img.src = url; out.appendChild(img);
  }
  const thumbUrl = `/download?path=${encodeURIComponent(j.thumbnail)}`;
  document.getElementById('thumbOut').innerHTML = `<img src="${thumbUrl}" style="max-width:320px;border-radius:12px;border:1px solid #2a3148" />`;
  const videoUrl = `/download?path=${encodeURIComponent(j.output_path)}`;
  document.getElementById('download').innerHTML = `<a href="${videoUrl}" download>⬇️ Download Video</a>`;
}
JS

# Set defaults so container binds to Render's $PORT
ENV PORT=8000
ENV APP_HOST=0.0.0.0
ENV APP_PORT=8000
ENV STORAGE_DIR=/app/backend/storage

# Entrypoint (use Render's $PORT if provided)
CMD ["sh","-c","python -m uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
