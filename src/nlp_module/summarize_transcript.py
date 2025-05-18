from transformers import pipeline
import os

def load_transcript(transcript_path):
    with open(transcript_path, "r") as f:
        return f.read()

def summarize_text(text, model_name="sshleifer/distilbart-cnn-12-6"):
    summarizer = pipeline("summarization", model=model_name)
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    transcript_path = "data/transcript.txt"
    
    if not os.path.exists(transcript_path):
        print("❌ No transcript found. Please run transcribe_audio.py first.")
        exit()

    text = load_transcript(transcript_path)
    print("🧠 Running summarization...")
    summary = summarize_text(text)
    
    print("\n📄 Summary:\n", summary)

    with open("data/summary.txt", "w") as f:
        f.write(summary)
