# 🎥 AutoAdaptAI — Video Intelligence Pipeline Using Deep Learning

**AutoAdaptAI** is a modular, end-to-end AI pipeline that transforms raw video content into actionable insights using deep learning. It combines computer vision, automatic speech recognition, and language modeling to analyze video files — delivering annotated images, full transcripts, and intelligent summaries.

This project was built entirely using open-source tools and frameworks like PyTorch, Whisper, torchvision, HuggingFace Transformers, and torchtune. It is structured to allow for future real-time streaming or scalable deployment.

---

## 🧠 Key Features

### 🖼️ Frame Extraction & Object Detection
- Extracts frames from video using OpenCV
- Performs object detection on frames using PyTorch's Faster R-CNN model (`torchvision`)
- Annotates each frame with labels and bounding boxes

### 🗣️ Audio Transcription
- Uses OpenAI's Whisper model to convert speech to text from the video’s audio
- Outputs a human-readable `.txt` transcript

### 📝 Transcript Summarization
- Uses HuggingFace Transformers (DistilBART) to generate a clean summary
- Stores the result in `data/summary.txt`

### ⚙️ Modular & Reproducible
- Runs completely offline in Python
- Cleanly separated into modules for vision, audio, and NLP
- Prepared for streaming and cloud deployment (Kafka + Kubernetes support planned)

---

## 🛠️ Tech Stack

| Feature               | Technology                               |
|----------------------|-------------------------------------------|
| Frame Extraction      | `opencv-python`                          |
| Object Detection      | `torch`, `torchvision`                   |
| Audio Transcription   | `openai-whisper`, `ffmpeg-python`        |
| Summarization         | `transformers`, `torchtune` (optional)   |
| Development Tools     | `VSCode`, Python 3.8+                    |

---

## 📁 Project Structure
```
AutoAdaptAI/
├── src/
│   ├── video_extraction/       # Extract video frames
│   ├── audio_transcription/    # Transcribe audio using Whisper
│   ├── vision_module/          # Detect objects in each frame
│   └── nlp_module/             # Summarize transcript using LLM
├── data/ # Stores output (frames, transcript, summary)
├── requirements.txt # Project dependencies
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### 1. Install Python packages
```bash
pip install -r requirements.txt
```
### 2. Add a sample video

Place your `.mp4` video in the root of the project directory and name it `sample_video.mp4` (or change the filename in the scripts if using a different name).

**Example:**
```
AutoAdaptAI/
├── sample_video.mp4 ✅
├── src/
├── requirements.txt
└── README.md
```

### 3. Extract video frames
Run the following to extract frames from the video:
```bash
python src/video_extraction/extract_frames.py
```

### 4. Transcribe the audio
Run Whisper to convert the video’s audio into text:
```bash
python src/audio_transcription/transcribe_audio.py
```
- Output will be printed to the terminal
- A transcript will be saved to: data/transcript.txt

### 5. Run object detection
Use a pretrained PyTorch model to detect and label objects in each frame:
```bash
python src/vision_module/detect_objects.py
```
- Annotated frames will be saved to: data/annotated/

### 6. Summarize the transcript
Generate a concise summary of the transcript using a language model:
```bash
python src/nlp_module/summarize_transcript.py
```
- Summary is saved to: data/summary.txt
