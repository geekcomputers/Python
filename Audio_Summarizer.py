import whisper
import re
import openai
import os


def transcript_generator():
    # Load Whisper model
    model = whisper.load_model("base")

    # Transcribe audio file
    result = model.transcribe("audio.mp4")

    # Send the transcript to the summarizer
    provide_summarizer(result)


def provide_summarizer(Text):
    # Set up Groq OpenAI-compatible API credentials
    openai.api_key = os.getenv(
        "OPENAI_API_KEY", "your-api-key-here"
    )  # Replace or set in environment
    openai.api_base = "https://api.groq.com/openai/v1"

    # Extract text from the Whisper result
    text_to_summarize = Text["text"]

    # Send the transcription to Groq for summarization
    response = openai.ChatCompletion.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant who summarizes long text into bullet points.",
            },
            {
                "role": "user",
                "content": f"Summarize the following:\n\n{text_to_summarize}",
            },
        ],
    )

    # Split the response into sentences
    summary = re.split(r"(?<=[.!?]) +", response["choices"][0]["message"]["content"])

    # Save summary to file
    with open("summary.txt", "w+", encoding="utf-8") as file:
        for sentence in summary:
            cleaned = sentence.strip()
            if cleaned:
                file.write("- " + cleaned + "\n")


if __name__ == "__main__":
    transcript_generator()
