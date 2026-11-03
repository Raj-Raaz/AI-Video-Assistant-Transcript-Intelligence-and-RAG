
from dotenv import load_dotenv

from utils.audio_processor import process_input_audio
from core.transcriber import transcribe_audio_chunks
from core.summarize import summarize_text, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question


load_dotenv()


def run_pipeline(source: str, language: str = "en") -> dict:

    print("Starting AI Video Assistant")

    # Step 1: Process audio
    chunks = process_input_audio(source)
    print(f"Processed audio into {len(chunks)} chunks.")

    # Step 2: Transcribe audio ONCE
    transcript = transcribe_audio_chunks(
        chunks,
        language=language
    )

    print(f"Transcript: {transcript[:300]}")

    # Step 3: Generate title and summary
    title = generate_title(transcript)
    summary = summarize_text(transcript)

    # Step 4: Extract information
    action_items = extract_action_items(transcript)
    key_decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)

    # Step 5: Build RAG chain
    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": key_decisions,
        "questions": questions,
        "rag_chain": rag_chain,
    }


if __name__ == "__main__":

    source = input("Enter YouTube URL or local file path: ").strip()

    language = input("Language (en/hi/hinglish): ").strip() or "en"

    result = run_pipeline(source, language)

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

        answer = ask_question(rag_chain, question)

        print(f"\nAssistant: {answer}\n")
