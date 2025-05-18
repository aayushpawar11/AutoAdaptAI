import whisper
import os
import ffmpeg

def extract_audio(video_path, audio_path="temp_audio.wav"):
    (
        ffmpeg
        .input(video_path)
        .output(audio_path, ac=1, ar='16k')
        .overwrite_output()
        .run()
    )
    return audio_path

def transcribe_audio(video_path):
    audio_path = extract_audio(video_path)
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    print(result["text"])
    return result["text"]

if __name__ == "__main__":
    text = transcribe_audio("sample_video.mp4")
    print("\n\n📝 Final Transcript:\n", text)
    # Save transcript to file for summarization
    with open("data/transcript.txt", "w") as f:
        f.write(text)    
