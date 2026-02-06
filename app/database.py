import os
import logging
from datetime import datetime, timedelta
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Supabase
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
bucket_name: str = os.environ.get("PROCTORING_BUCKET_NAME", "proctoring_media")

if not url or not key:
    logger.error("Supabase URL or Key not found in environment variables.")
    raise ValueError("Missing Supabase configuration.")

supabase: Client = create_client(url, key)


def log_violation(session_id: str, reason: str, file_bytes: bytes, file_ext: str):
    """
    Uploads the proof and logs the violation in one go.
    """
    try:
        # 1. Upload File to Storage Bucket
        filename = f"{session_id}/{reason}_{os.urandom(4).hex()}.{file_ext}"
        
        # Ensure bucket exists or handle error (Supabase storage usually requires pre-created buckets)
        supabase.storage.from_(bucket_name).upload(filename, file_bytes, {
            "content-type": f"image/{file_ext}" if file_ext in ['jpg', 'png'] else "audio/wav"
        })

        # 2. Get Public URL
        public_url = supabase.storage.from_(bucket_name).get_public_url(filename)

        # 3. Insert into 'proctoring_media' table
        # Set dynamic expiration (e.g., 7 days from now)
        expires_at = (datetime.now() + timedelta(days=7)).isoformat()
        
        media_data = {
            "session_id": session_id,
            "media_type": "photo" if file_ext in ['jpg', 'png'] else "audio",
            "media_url": public_url,
            "storage_path": filename,
            "capture_reason": reason,
            "expires_at": expires_at
        }
        media_res = supabase.table("proctoring_media").insert(media_data).execute()

        if not media_res.data:
            logger.error("Error saving media record: No data returned")
            return

        media_id = media_res.data[0]['id']

        # 4. Insert into 'proctoring_logs'
        log_data = {
            "session_id": session_id,
            "event_type": reason,
            "severity": "violation",
            "media_id": media_id,
            "event_data": {"description": f"AI detected {reason}"}
        }
        supabase.table("proctoring_logs").insert(log_data).execute()

        logger.info(f"Logged violation: {reason} for session {session_id}")

    except Exception as e:
        logger.error(f"Database Error: {e}")