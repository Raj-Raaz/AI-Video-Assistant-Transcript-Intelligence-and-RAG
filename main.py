from dotenv import load_dotenv

load_dotenv()

from core.llm_utils import RateLimitError

from utils.audio_processor import process_input_audio
from core.transcriber import transcribe_audio_chunks
from core.summarize import summarize_text, generate_title
from core.extractor import extract_information
from core.rag_engine import build_rag_chain, ask_question


def run_pipeline(source: str, language: str = "en", output_dir=None, on_progress=None) -> dict:

    print("Starting AI Video Assistant")

    report = on_progress or print
    report("Preparing audio...")
    chunks = (process_input_audio(source, output_dir=output_dir)
              if output_dir else process_input_audio(source))
    print(f"Processed audio into {len(chunks)} chunks.")

    report("Transcribing audio...")
    transcription = transcribe_audio_chunks(chunks, language=language, with_segments=True)
    transcript = transcription["text"]
    if not transcript.strip():
        raise ValueError("Transcription returned no text")

    print(f"Transcript: {transcript}")

    # Generate title and summary
    report("Generating title and summary...")
    title = generate_title(transcript)
    summary = summarize_text(transcript)

    # Extract information
    report("Extracting action items, decisions, and questions...")
    information = extract_information(transcript)

    # Prepare video chat
    report("Preparing video chat...")
    rag_chain = build_rag_chain(transcript, segments=transcription["segments"], source=source)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": information["action_items"],
        "key_decisions": information["key_decisions"],
        "questions": information["questions"],
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":

    source = input("Enter YouTube URL or local file path: ").strip()

    language = input("Language (en/hi/hinglish): ").strip() or "en"

    try:
        result = run_pipeline(source, language)
    except RateLimitError as exc:
        print(f"\n{exc}")
        raise SystemExit(1) from None

    print("\n" + "=" * 60)
    print(f"Title: {result['title']}")
    print(f"\nSummary:\n{result['summary']}")
    print(f"\nAction Items:\n{result['action_items']}")
    print(f"\nKey Decisions:\n{result['key_decisions']}")
    print(f"\nOpen Questions:\n{result['questions']}")
    print("=" * 60)

    # Chat with video
    print("\nChat with your video (type 'exit' to quit)\n")

    rag_chain = result["rag_chain"]

    while True:

        question = input("You: ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            answer = ask_question(rag_chain, question)
        except RateLimitError as exc:
            print(f"\n{exc}\n")
            continue

        print(f"\nAssistant: {answer}\n")
