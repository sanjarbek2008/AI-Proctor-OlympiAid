from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from starlette.middleware.cors import CORSMiddleware

from app.ai_engine import analyze_image, analyze_audio
from app.database import log_violation

app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    from app.ai_engine import device
    print("\n" + "="*50)
    print(f"AI OLYMPIAID STARTING UP")
    print(f"DEVICE BEING USED: {device.upper()}")
    print("="*50 + "\n")


@app.get("/")
def health_check():
    return {"status": "AI Proctor is Active"}

@app.post("/analyze")
async def analyze_session(
    background_tasks: BackgroundTasks,
    session_id: str = Form(...),
    image: UploadFile = File(None),
    audio: UploadFile = File(None)
):
    """
    Receives files, runs AI instantly, and logs to DB in background
    so we don't slow down the response.
    """
    flags = []

    # 1. Image Analysis
    if image:
        img_bytes = await image.read()
        visual_flag, processed_img = analyze_image(img_bytes, session_id=session_id)
        if visual_flag:
            flags.append(visual_flag)
            # Send database work to background task (using annotated image)
            background_tasks.add_task(log_violation, session_id, visual_flag, processed_img, "jpg")

    # 2. Audio Analysis
    if audio:
        audio_bytes = await audio.read()
        audio_flag = analyze_audio(audio_bytes)
        if audio_flag:
            flags.append(audio_flag)
            background_tasks.add_task(log_violation, session_id, audio_flag, audio_bytes, "wav")

    return {
        "status": "processed", 
        "flags_detected": flags,
        "action": "warning" if flags else "ok"
    }