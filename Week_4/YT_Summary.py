from youtube_transcript_api import YouTubeTranscriptApi

video_url = "https://www.youtube.com/watch?v=O2gerCxEXvc"

# Extract video ID
if "v=" in video_url:
    video_id = video_url.split("v=")[-1].split("&")[0]
else:
    video_id = video_url.split("/")[-1]

try:
    # Correct usage for v1.2.1
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    transcript = transcript_list.find_transcript(['en']).fetch()
    text = " ".join([entry["text"] for entry in transcript])
    print(text[:1000])  # Print first 1000 characters
except Exception as e:
    print("Failed to fetch transcript:", e)
