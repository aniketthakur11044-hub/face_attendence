# Face Recognition Attendance System (Deep Learning Edition)

A Streamlit + OpenCV app that marks attendance by scanning a person's face
with your laptop webcam — recognition is powered by a pretrained deep
neural network (FaceNet / `facenet-pytorch`), not a classical algorithm.

## What changed from the classic version

The original app used OpenCV's **LBPH** recognizer — a classical,
hand-crafted-feature algorithm (not deep learning). This version replaces
it with:

- **MTCNN** — a small CNN that detects and aligns the face in each frame
- **InceptionResnetV1** (pretrained on VGGFace2) — a deep CNN that turns
  an aligned face into a 512-number "embedding" vector

Two photos of the same person produce embeddings that land close together
(small Euclidean distance); different people land far apart. "Training"
now just means embedding every registered photo and saving the vectors to
`embeddings.pickle`. "Recognizing" means embedding the new face and
checking whose stored embedding it's closest to.

Everything else — the Streamlit UI, the ID/name registry, the daily
Check-In/Check-Out CSVs — works exactly the same as before.

## How it works

- **Register New Person** – captures ~40 aligned face photos via webcam
- **Train Model** – runs every registered photo through the deep network
  and saves the resulting embeddings
- **Take Attendance** – opens the webcam, embeds any detected face, and
  logs a Check-In/Check-Out for whoever it's closest to (if close enough)
- **View Attendance Records** – browse and download any day's CSV

## Setup (VS Code, on your own laptop)

1. Open this folder in VS Code.
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   `torch`/`torchvision` are the biggest downloads here. Plain
   `pip install torch torchvision` works fine (CPU-only is plenty for this
   app — no GPU required). If the download is slow or fails, try the
   dedicated CPU wheel index:
   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```
4. Run the app:
   ```
   streamlit run app.py
   ```
   This opens the app in your browser at `http://localhost:8501`.

**Note:** the first time you run the app, `facenet-pytorch` will download
the pretrained InceptionResnetV1 weights (~110 MB) from the internet and
cache them locally (usually under `~/.cache/torch`). After that first run,
no internet connection is needed.

## Usage order

1. Go to **Register New Person**, type a unique ID and a name (no spaces),
   click **Start Capturing**. A webcam window pops up — look around a bit
   so it captures different angles. Press `Q` to stop early if needed.
2. Repeat step 1 for everyone you want to recognize.
3. Go to **Train Model** and click **Train Now**. Do this again anytime you
   add or delete a person.
4. Go to **Take Attendance** and click **Check In** or **Check Out**. A
   webcam window opens for ~20 seconds; anyone recognized gets marked
   automatically.
5. Check **View Attendance Records** to see or download the day's log.

## If you're upgrading from the old LBPH version

The old `dataset/` photos were tight grayscale Haar-cascade crops, and
`trained_model.yml` / `labels.pickle` are LBPH-specific — they won't work
with the new deep-learning pipeline. Before using this version:

1. Delete the old `dataset/`, `trained_model.yml`, `labels.pickle`, and
   `people.csv` (or just start in a fresh project folder).
2. Re-register everyone with **Register New Person**, then **Train Model**.

## Notes & tips

- Runs fine on CPU; a GPU isn't required but will speed up scanning if you
  have one (the code auto-detects CUDA).
- Good, even lighting during registration noticeably improves accuracy.
- If recognition misfires, try re-registering that person with more/
  better-lit samples, or adjust `DISTANCE_THRESHOLD` in `face_utils.py`
  (lower = stricter match, fewer false positives but more "Unknown"
  results).
- All data is stored locally: `dataset/<id_name>/` (aligned face photos),
  `embeddings.pickle` (the trained embeddings), `attendance_records/`
  (daily CSVs).
- To remove someone, use **Manage People** → **Delete a person**, then
  retrain.
