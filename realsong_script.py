import os
import pandas as pd
from pytube import YouTube
from pydub import AudioSegment

# Load the CSV file
csv_file = "real_songs.csv"  # Update with the correct path if necessary
output_folder = "dataset/real_songs/"

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_file)

# Extract YouTube IDs
youtube_ids = df["youtube_id"].unique()

# Function to download and convert YouTube video to MP3
def download_mp3(youtube_id):
    try:
        url = f"https://www.youtube.com/watch?v={youtube_id}"
        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        
        if audio_stream:
            print(f"Downloading: {yt.title}")
            temp_filename = audio_stream.download(output_path=output_folder)
            mp3_filename = os.path.join(output_folder, f"{youtube_id}.mp3")
            
            # Convert to MP3
            audio = AudioSegment.from_file(temp_filename)
            audio.export(mp3_filename, format="mp3")
            
            # Remove the temporary file
            os.remove(temp_filename)
            print(f"Saved as: {mp3_filename}")
        else:
            print(f"No audio stream found for {youtube_id}")
    except Exception as e:
        print(f"Error downloading {youtube_id}: {e}")

# Iterate and download
for youtube_id in youtube_ids:
    download_mp3(youtube_id)

print("Download process completed.")
