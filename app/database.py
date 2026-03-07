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


import uuid

def log_violation(session_id: str, reason: str, file_bytes: bytes, file_ext: str):
    """
    Uploads the proof and logs the violation in one go.
    """
    try:
        # 1. Ensure session_id is a valid UUID string
        # If it's a plain string like 'student_001', convert it to a deterministic UUID
        try:
            uuid.UUID(session_id)
            db_session_id = session_id
        except ValueError:
            # Create a deterministic UUID from the string name
            db_session_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, session_id))
            logger.warning(f"Session ID '{session_id}' is not a UUID. Using deterministic UUID: {db_session_id}")

        # 2. Upload File to Storage Bucket
        filename = f"{session_id}/{reason}_{os.urandom(4).hex()}.{file_ext}"
        
        # Ensure bucket exists or handle error (Supabase storage usually requires pre-created buckets)
        supabase.storage.from_(bucket_name).upload(filename, file_bytes, {
            "content-type": f"image/{file_ext}" if file_ext in ['jpg', 'png'] else "audio/wav"
        })

        # 3. Get Public URL
        public_url = supabase.storage.from_(bucket_name).get_public_url(filename)

        # 4. Insert into 'proctoring_media' table
        expires_at = (datetime.now() + timedelta(days=2)).isoformat()
        
        media_data = {
            "session_id": db_session_id,
            "media_type": "photo" if file_ext in ['jpg', 'png'] else "audio",
            "media_url": public_url,
            "storage_path": filename,
            "capture_reason": reason,
            "expires_at": expires_at
        }
        media_res = supabase.table("proctoring_media").insert(media_data).execute()

        if not media_res.data:
            logger.error(f"Error saving media record: {media_res}")
            return

        media_id = media_res.data[0]['id']

        # 5. Insert into 'proctoring_logs'
        log_data = {
            "session_id": db_session_id,
            "event_type": reason,
            "severity": "violation",
            "media_id": media_id,
            "event_data": {"description": f"AI detected {reason}", "original_session_id": session_id}
        }
        supabase.table("proctoring_logs").insert(log_data).execute()

        logger.info(f"Successfully logged violation: {reason} for session {session_id}")

    except Exception as e:
        logger.error(f"Database Error: {e}")