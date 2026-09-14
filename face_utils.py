"""
face_utils.py
--------------------------------------------------------------
Backend for the face-scan attendance app: registration/capture,
FaceNet embedding "training", live recognition, and the
attendance CSV (unique person ID, separate Check-In / Check-Out
entries, plus delete helpers for records and people).

DEEP LEARNING VERSION
This replaces the old OpenCV LBPH recognizer (a classical,
hand-crafted-feature algorithm) with a real deep neural network:

  - MTCNN            -> a small CNN that detects + aligns faces
  - InceptionResnetV1 -> a deep CNN (pretrained on VGGFace2) that
                          turns an aligned face into a 512-number
                          "embedding" vector

Two faces of the same person produce embeddings that are close
together (small Euclidean distance); different people produce
embeddings that are far apart. So "training" here just means:
run every registered photo through the network and store its
embedding. "Recognizing" means: embed the new face and see whose
stored embedding it's closest to.

Requires: torch, torchvision, facenet-pytorch, opencv-python, numpy, pandas
    pip install torch torchvision facenet-pytorch opencv-python numpy pandas

The first time this file is imported, facenet-pytorch will
download the pretrained InceptionResnetV1 weights (~110 MB) from
the internet, then cache them locally. After that first run, no
internet connection is needed.
--------------------------------------------------------------
"""

import os
import shutil
import pickle
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
DATASET_DIR = "dataset"
ATTENDANCE_DIR = "attendance_records"
EMBEDDINGS_PATH = "embeddings.pickle"   # replaces trained_model.yml + labels.pickle
PEOPLE_CSV = "people.csv"

# Max Euclidean distance between two FaceNet embeddings to count as a match.
# Lower = stricter match (fewer false positives, more "Unknown" results).
# 0.9 is a commonly used starting point for VGGFace2-trained embeddings.
DISTANCE_THRESHOLD = 0.9
SCAN_SECONDS = 20

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load the deep learning models ONCE, at import time, so we don't reload the
# network weights on every capture/recognize call.
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MTCNN_DETECTOR = MTCNN(image_size=160, margin=20, keep_all=False, post_process=True, device=DEVICE)
RESNET = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)


# ---------------------------------------------------------------------------
# Small helpers for converting between OpenCV images and model tensors
# ---------------------------------------------------------------------------
def _detect_and_align(rgb_frame):
    """
    Run MTCNN on a full RGB frame. Returns (aligned_tensor, box) or
    (None, None) if no face was found.
    aligned_tensor is already resized/normalized the way InceptionResnetV1
    expects (160x160, values roughly in [-1, 1]).
    """
    boxes, _ = MTCNN_DETECTOR.detect(rgb_frame)
    face_tensor = MTCNN_DETECTOR(rgb_frame)
    if face_tensor is None or boxes is None:
        return None, None
    return face_tensor, boxes[0]


def _stored_image_to_tensor(img_bgr):
    """
    Convert an already-cropped face image (as saved to disk, BGR) into the
    normalized tensor InceptionResnetV1 expects. Used at training time,
    since the stored images are already aligned crops (no need to re-run
    face detection on them).
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (160, 160))
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1)   # HWC -> CHW
    tensor = (tensor - 127.5) / 128.0                          # match facenet-pytorch normalization
    return tensor


def _embed(tensor):
    """Run a single 160x160 face tensor through the deep network -> 512-d numpy vector."""
    with torch.no_grad():
        vec = RESNET(tensor.unsqueeze(0).to(DEVICE))
    return vec.cpu().numpy()[0]


def _tensor_to_bgr_image(face_tensor):
    """Undo FaceNet's normalization so we can save the aligned crop as a viewable .jpg."""
    face_np = face_tensor.permute(1, 2, 0).cpu().numpy()
    face_np = np.clip(face_np * 128.0 + 127.5, 0, 255).astype("uint8")   # back to RGB 0-255
    return cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# People registry (ID <-> Name <-> dataset folder)
# ---------------------------------------------------------------------------
def _load_people_df():
    if os.path.exists(PEOPLE_CSV):
        return pd.read_csv(PEOPLE_CSV, dtype=str)
    return pd.DataFrame(columns=["ID", "Name", "Folder"])


def _save_people_df(df):
    df.to_csv(PEOPLE_CSV, index=False)


def id_exists(person_id):
    df = _load_people_df()
    if df.empty:
        return False
    return str(person_id).strip() in df["ID"].astype(str).values


def get_people():
    """Returns the full people registry as a DataFrame with columns ID, Name, Folder, Photos."""
    df = _load_people_df()
    if df.empty:
        return df
    photo_counts = []
    for folder in df["Folder"]:
        path = os.path.join(DATASET_DIR, folder)
        photo_counts.append(len(os.listdir(path)) if os.path.isdir(path) else 0)
    df = df.copy()
    df["Photos"] = photo_counts
    return df


# ---------------------------------------------------------------------------
# Registration / capture
# ---------------------------------------------------------------------------
def capture_faces(person_id, name, num_samples=40):
    person_id = str(person_id).strip()
    name = name.strip().replace(" ", "_")

    if not person_id or not name:
        return False, "Both a unique ID and a name are required."
    if id_exists(person_id):
        return False, f"ID '{person_id}' is already registered. Please choose a different ID."

    folder_name = f"{person_id}_{name}"
    folder_path = os.path.join(DATASET_DIR, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        shutil.rmtree(folder_path, ignore_errors=True)
        return False, "Could not access the webcam."

    count = 0
    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_tensor, box = _detect_and_align(rgb)
        if face_tensor is not None:
            count += 1
            face_bgr = _tensor_to_bgr_image(face_tensor)
            cv2.imwrite(os.path.join(folder_path, f"{count}.jpg"), face_bgr)

            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{count}/{num_samples}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Registering - press Q to stop early", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if count == 0:
        shutil.rmtree(folder_path, ignore_errors=True)
        return False, "No face detected. Try again with better lighting, facing the camera."

    df = _load_people_df()
    df = pd.concat(
        [df, pd.DataFrame([{"ID": person_id, "Name": name, "Folder": folder_name}])],
        ignore_index=True,
    )
    _save_people_df(df)

    return True, f"Captured {count} images for '{name}' (ID: {person_id})."


# ---------------------------------------------------------------------------
# Delete a registered person entirely
# ---------------------------------------------------------------------------
def delete_person(person_id):
    df = _load_people_df()
    match = df[df["ID"].astype(str) == str(person_id).strip()]
    if match.empty:
        return False, f"No registered person with ID '{person_id}'."

    folder = match.iloc[0]["Folder"]
    shutil.rmtree(os.path.join(DATASET_DIR, folder), ignore_errors=True)

    df = df[df["ID"].astype(str) != str(person_id).strip()]
    _save_people_df(df)

    return True, f"Deleted '{match.iloc[0]['Name']}' (ID: {person_id}). Retrain the model to apply this."


# ---------------------------------------------------------------------------
# Training (i.e. embedding every stored photo with the deep network)
# ---------------------------------------------------------------------------
def train_model():
    people_df = _load_people_df()
    if people_df.empty:
        return False, "No one is registered yet. Register at least one person first."

    embeddings_db = {}   # person_id -> {"name": str, "embeddings": np.ndarray of shape (n, 512)}
    total_images = 0

    for _, row in people_df.iterrows():
        folder_path = os.path.join(DATASET_DIR, row["Folder"])
        if not os.path.isdir(folder_path):
            continue

        vecs = []
        for img_name in os.listdir(folder_path):
            img_bgr = cv2.imread(os.path.join(folder_path, img_name))
            if img_bgr is None:
                continue
            tensor = _stored_image_to_tensor(img_bgr)
            vecs.append(_embed(tensor))
            total_images += 1

        if vecs:
            embeddings_db[str(row["ID"])] = {"name": row["Name"], "embeddings": np.array(vecs)}

    if not embeddings_db:
        return False, "No usable face images found. Try registering someone again."

    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(embeddings_db, f)

    return True, f"Trained FaceNet embeddings on {total_images} images across {len(embeddings_db)} people."


# ---------------------------------------------------------------------------
# Attendance file helpers
# ---------------------------------------------------------------------------
ATTENDANCE_COLUMNS = ["ID", "Name", "Type", "Time"]


def _attendance_path(date=None):
    date = date or datetime.now()
    return os.path.join(ATTENDANCE_DIR, f"attendance_{date.strftime('%Y-%m-%d')}.csv")


def _load_today_df():
    path = _attendance_path()
    if os.path.exists(path):
        df = pd.read_csv(path, dtype={"ID": str})
        for col in ATTENDANCE_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df
    return pd.DataFrame(columns=ATTENDANCE_COLUMNS)


def load_attendance_for_date(date):
    path = _attendance_path(date)
    if os.path.exists(path):
        return pd.read_csv(path, dtype={"ID": str})
    return pd.DataFrame(columns=ATTENDANCE_COLUMNS)


# ---------------------------------------------------------------------------
# Recognition + marking (Check-In / Check-Out aware)
# ---------------------------------------------------------------------------
def _best_match(embedding, embeddings_db):
    """Return (person_id, name, distance) for the closest stored embedding, or (None, None, inf)."""
    best_id, best_name, best_dist = None, None, float("inf")
    for pid, data in embeddings_db.items():
        dist = np.linalg.norm(data["embeddings"] - embedding, axis=1).min()
        if dist < best_dist:
            best_dist, best_id, best_name = dist, pid, data["name"]
    return best_id, best_name, best_dist


def recognize_and_mark(action="in"):
    """
    action: "in" or "out"
    Returns (marked, message) where marked is a list of (id, name) tuples
    for everyone newly logged in this scan.
    """
    action = "in" if action not in ("in", "out") else action

    if not os.path.exists(EMBEDDINGS_PATH):
        return [], "Model not trained yet. Go to 'Train Model' first."

    with open(EMBEDDINGS_PATH, "rb") as f:
        embeddings_db = pickle.load(f)

    today_df = _load_today_df()
    marked = []
    seen_this_session = set()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return [], "Could not access the webcam."

    start = datetime.now()
    while (datetime.now() - start).seconds < SCAN_SECONDS:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_tensor, box = _detect_and_align(rgb)
        if face_tensor is not None:
            embedding = _embed(face_tensor)
            person_id, name, dist = _best_match(embedding, embeddings_db)

            box_color = (0, 0, 255)
            display_text = "Unknown"

            if person_id is not None and dist < DISTANCE_THRESHOLD:
                display_text = name.replace("_", " ")
                box_color = (0, 255, 0)

                already_this_type = not today_df[
                    (today_df["ID"].astype(str) == str(person_id)) & (today_df["Type"] == action)
                ].empty

                if person_id in seen_this_session or already_this_type:
                    display_text += f" (already {action})"
                    box_color = (255, 200, 0)
                elif action == "out" and today_df[
                    (today_df["ID"].astype(str) == str(person_id)) & (today_df["Type"] == "in")
                ].empty:
                    display_text += " (check in first)"
                    box_color = (0, 165, 255)
                else:
                    new_row = {
                        "ID": person_id,
                        "Name": name,
                        "Type": action,
                        "Time": datetime.now().strftime("%H:%M:%S"),
                    }
                    today_df = pd.concat([today_df, pd.DataFrame([new_row])], ignore_index=True)
                    marked.append((person_id, name))
                    seen_this_session.add(person_id)

            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, display_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2)

        label_txt = "CHECK-IN" if action == "in" else "CHECK-OUT"
        cv2.putText(frame, f"{label_txt} - press Q to stop", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Scanning", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if marked:
        today_df.to_csv(_attendance_path(), index=False)
        names_str = ", ".join(n.replace("_", " ") for _, n in marked)
        verb = "in" if action == "in" else "out"
        return marked, f"Checked {verb}: {names_str}"

    return [], "No new matches this scan."


# ---------------------------------------------------------------------------
# Delete attendance records
# ---------------------------------------------------------------------------
def delete_attendance_rows(date, row_indices):
    """Delete specific rows (by DataFrame index) from a given date's CSV."""
    path = _attendance_path(date)
    if not os.path.exists(path):
        return False, "No record file exists for that date."

    df = pd.read_csv(path, dtype={"ID": str})
    before = len(df)
    df = df.drop(index=[i for i in row_indices if i in df.index]).reset_index(drop=True)
    df.to_csv(path, index=False)
    deleted = before - len(df)
    return True, f"Deleted {deleted} record(s)."


def delete_all_attendance(date):
    """Delete the entire attendance file for a given date."""
    path = _attendance_path(date)
    if os.path.exists(path):
        os.remove(path)
        return True, "All records for that date were deleted."
    return False, "No record file exists for that date."
