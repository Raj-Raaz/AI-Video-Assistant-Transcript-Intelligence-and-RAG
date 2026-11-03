from utils.audio_processor import process_input_audio
from core.transcriber import transcribe_audio_chunks

input_file = "https://www.youtube.com/watch?v=UabBYexBD4k" 


chunks = process_input_audio(input_file)
transcription = transcribe_audio_chunks(chunks, translate=False)
print ("Transcription:\n", transcription)