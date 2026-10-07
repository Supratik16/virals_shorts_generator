import subprocess
import json
import os
from datetime import datetime
from llama_cpp import Llama
from faster_whisper import WhisperModel

def extract_audio(video_file_path: str, output_audio_file_path: str = "./test.wav") -> str:
    print(f"Extracting audio from {video_file_path}...")
    command = [
        "ffmpeg", "-y", "-i", video_file_path,
        "-ar", "16000", "-ac", "1", output_audio_file_path
    ]
    # Run the command and suppress the messy FFmpeg terminal output
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Audio extraction complete!")
    return output_audio_file_path

def transcribe_with_timestamps(audio_file_path: str) -> list:
    model = WhisperModel("/home/supratik/Documents/models/faster-whisper-small/",
                            device="cpu",
                            compute_type="int8",
                            local_files_only=True)
    print("model loaded")
    start_time = datetime.now()
    segments, info = model.transcribe(audio_file_path, word_timestamps=True)
    transcript_data = []
    # Faster-whisper returns a generator. We must iterate through it.
    for segment in segments:
        for word in segment.words:
            # We structure the data as a list of dictionaries
            word_data = {
                "word": word.word.strip(),
                "start": round(word.start, 2),
                "end": round(word.end, 2)
            }
            transcript_data.append(word_data)
            # print(f"[{word_data['start']}s -> {word_data['end']}s] {word_data['word']}")
    end_time = datetime.now()
    print(f"Time taken to generate timestamped transcript: {end_time-start_time}")
    return transcript_data

def get_hotspots(model_path: str, json_filepath: str, chunk_minutes: int = 3, keep_top_n: int = 5) -> list:
    """
    PHASE 1: The Radar Triage.
    Chunks the video into large blocks and uses the SLM to find the 'Hot Zones',
    dropping the boring parts of the podcast before we even start clipping.
    """
    with open(json_filepath, 'r', encoding='utf-8') as f:
        words_data = json.load(f)

    llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)
    
    chunks = []
    current_chunk_words = []
    chunk_start = words_data[0]["start"]
    chunk_duration_sec = chunk_minutes * 60

    # 1. Group words into 3-minute macro-blocks
    for item in words_data:
        current_chunk_words.append(item["word"].strip())
        if item["end"] - chunk_start >= chunk_duration_sec and item["word"].endswith(('.', '?', '!')):
            chunks.append({
                "start": chunk_start,
                "end": item["end"],
                "text": " ".join(current_chunk_words)
            })
            current_chunk_words = []
            chunk_start = item["end"]
            
    if current_chunk_words: # Catch the remainder
        chunks.append({"start": chunk_start, "end": words_data[-1]["end"], "text": " ".join(current_chunk_words)})
    print(f"\n📡 PHASE 1: Scanning {len(chunks)} Macro-Blocks ({chunk_minutes} mins each) for Hot Zones...")
    start_time = datetime.now()
    scored_chunks = []
    triage_prompt = """
    You are an expert viral video editor. Evaluate this podcast segment for its viral potential.
    Grade the segment on three specific criteria, using a strict 1 to 5 scale (1=Poor, 3=Average, 5=Excellent).

    Criteria:
    1. Hook: Does it start with a controversial, surprising, or highly engaging statement?
    2. Stakes: Is there high emotion, conflict, or valuable information?
    3. Cohesion: Is there a clear beginning, middle, and end to the thought?

    CRITICAL INSTRUCTION: You must provide a brief 1-sentence reasoning BEFORE giving the integer score for each category.

    Respond ONLY with a JSON object in this exact format:
    {
        "hook_reasoning": "string",
        "hook_score": int,
        "stakes_reasoning": "string",
        "stakes_score": int,
        "cohesion_reasoning": "string",
        "cohesion_score": int
    }
    """

    for i, chunk in enumerate(chunks):
        print(f"  Scanning Block {i+1}/{len(chunks)}...", end="\r")
        try:
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": triage_prompt},
                    {"role": "user", "content": chunk["text"][:3000]} 
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            data = json.loads(response["choices"][0]["message"]["content"])
            # Python does the math to combine the parameters
            total_score = data.get("hook_score", 0) + data.get("stakes_score", 0) + data.get("cohesion_score", 0)
            chunk["score"] = total_score
            chunk["breakdown"] = data # Save the reasoning for debugging!
            scored_chunks.append(chunk)
        except Exception:
            chunk["score"] = 0
            scored_chunks.append(chunk)

    # Sort based on the combined total score (Max possible is 15)
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    print(f"No of hotspots: {keep_top_n}")
    hotspots = scored_chunks[:keep_top_n]
    end_time = datetime.now()
    print(f"Time taken to fetch hotstops from the video: {end_time-start_time}")
    print(f"\n🔥 Identified {len(hotspots)} Hot Zones! Discarding the rest of the video.")
    return hotspots


def generate_valid_clips(json_filepath: str, hotspots: list, min_duration: float = 30.0, max_duration: float = 60.0) -> list:
    """
    PHASE 2: Precision Windowing.
    Runs the sliding window ONLY inside the identified Hot Zones.
    Automatically scales the stride based on video length.
    """
    with open(json_filepath, 'r', encoding='utf-8') as f:
        words_data = json.load(f)

    # Dynamic Stride: If video is long, take bigger steps to save time
    total_video_duration = words_data[-1]["end"]
    stride_seconds = 20.0 if total_video_duration > 1800 else 15.0 # >30 min video = 20s stride

    sentences = []
    current_sentence = []
    sentence_start = None

    for item in words_data:
        if sentence_start is None: sentence_start = item["start"]
        current_sentence.append(item["word"].strip())
        if item["word"].endswith(('.', '?', '!')):
            sentences.append({"text": " ".join(current_sentence), "start": sentence_start, "end": item["end"]})
            current_sentence = []
            sentence_start = None

    valid_clips = []
    clip_id = 1
    last_start_time = -stride_seconds 

    for i in range(len(sentences)): 
        start_time = sentences[i]["start"]
        
        # KEY OPTIMIZATION: Only generate a clip if its start_time falls inside a Hot Zone
        in_hotzone = any(h["start"] <= start_time <= h["end"] for h in hotspots)
        
        if in_hotzone and (start_time - last_start_time >= stride_seconds):
            clip_text = ""
            end_time = start_time
            
            for j in range(i, len(sentences)):
                clip_text += sentences[j]["text"] + " "
                end_time = sentences[j]["end"]
                duration = end_time - start_time
                
                if duration >= min_duration:
                    if duration <= max_duration:
                        valid_clips.append({
                            "clip_id": clip_id,
                            "start": round(start_time, 2),
                            "end": round(end_time, 2),
                            "duration": round(duration, 2),
                            "text": clip_text.strip()
                        })
                        clip_id += 1
                        last_start_time = start_time 
                    break 

    return valid_clips

def get_intellegence_data(model_path: str, valid_clips: list, target_outputs: int = 4, batch_size: int = 4) -> list:
    """
    PHASE 3: Single-Round Tournament & Final Scoring.
    """
    llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)
    print(f"\n⚔️ PHASE 3: Running Knockout Tournament on {len(valid_clips)} precise clips...")
    
    # -------------------------------- SINGLE ROUND TOURNAMENT (Fixes the "Group of Death") -----------------------------
    start_time = datetime.now()
    tournament_winners = []
    batches = [valid_clips[i:i + batch_size] for i in range(0, len(valid_clips), batch_size)]
    
    for batch_idx, batch in enumerate(batches):
        if len(batch) == 1:
            tournament_winners.append(batch[0])
            continue
            
        batch_text = "".join([f"\n--- Clip ID: {c['clip_id']} ---\n{c['text']}\n" for c in batch])
        bracket_prompt = f"""
        You are an expert viral video editor. Evaluate these {len(batch)} clips and select the ONE that has the highest potential to go viral.
        Look for a strong hook, emotional stakes, and a clear narrative payoff.
        
        CRITICAL INSTRUCTION: You must write a brief 1-sentence reasoning for your choice BEFORE outputting the winning ID.
        
        Respond ONLY with a JSON object in this exact format:
        {{
            "reasoning": "string",
            "best_clip_id": int
        }}
        
        Clips for Evaluation:
        {batch_text}
        """
        
        print(f"  Battling Match {batch_idx + 1}/{len(batches)}...", end="\r")
        try:
            res = llm.create_chat_completion(
                messages=[{"role": "user", "content": bracket_prompt}],
                response_format={
                    "type": "json_object",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "reasoning": {"type": "string"},
                            "best_clip_id": {"type": "integer"}
                        },
                        "required": ["reasoning", "best_clip_id"]
                    }
                }, 
                temperature=0.0 # Keep this at 0.0 for strict, analytical choices
            )
            
            data = json.loads(res["choices"][0]["message"]["content"])
            winning_id = data.get("best_clip_id")
            
            # Your brilliant fail-safe remains exactly the same!
            winner = next((c for c in batch if c["clip_id"] == winning_id), batch[0])
            tournament_winners.append(winner)
            
        except Exception as e:
            # Silently default to the first clip if JSON parsing fails
            tournament_winners.append(batch[0])

    end_time = datetime.now()
    print(f"Time taken to run the First round tournament: {end_time-start_time}")

    # ---------------------------------------- FINAL SCORING --------------------------------------------------
    start_time = datetime.now()
    print(f"\n\n🎯 Final Phase: Scoring the {len(tournament_winners)} Tournament Winners...")
    final_scored_clips = []
    
    # We use a strict 1-10 scale per category to give a wider point spread for the final ranking
    scoring_prompt = """
    You are an expert viral video editor evaluating a tournament-winning clip.
    Grade the clip on three specific criteria, using a strict 1 to 10 scale for each.

    Criteria:
    1. Hook: Does the first sentence immediately grab attention?
    2. Stakes: Is there high emotion, conflict, or valuable information?
    3. Payoff: Does the clip end on a satisfying or thought-provoking note?

    CRITICAL INSTRUCTION: You must provide a 1-sentence reasoning BEFORE giving the integer score for each category.

    Respond ONLY with a JSON object in this format:
    {
        "hook_reasoning": "string",
        "hook_score": int,
        "stakes_reasoning": "string",
        "stakes_score": int,
        "payoff_reasoning": "string",
        "payoff_score": int
    }
    """

    for idx, clip in enumerate(tournament_winners):
        print(f"  Scoring Winner {idx + 1}/{len(tournament_winners)}...", end="\r")
        try:
            res = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": scoring_prompt},
                    {"role": "user", "content": f"Clip Transcript:\n\"{clip['text']}\""}
                ],
                response_format={
                    "type": "json_object",
                    # Forcing the schema guarantees it doesn't get "lazy" on the final output
                    "schema": {
                        "type": "object",
                        "properties": {
                            "hook_reasoning": {"type": "string"}, "hook_score": {"type": "integer"},
                            "stakes_reasoning": {"type": "string"}, "stakes_score": {"type": "integer"},
                            "payoff_reasoning": {"type": "string"}, "payoff_score": {"type": "integer"}
                        },
                        "required": ["hook_reasoning", "hook_score", "stakes_reasoning", "stakes_score", "payoff_reasoning", "payoff_score"]
                    }
                }, 
                temperature=0.1 
            )
            data = json.loads(res["choices"][0]["message"]["content"])
            
            clip_copy = clip.copy()
            # Calculate total score out of 30
            clip_copy["score"] = data.get("hook_score", 0) + data.get("stakes_score", 0) + data.get("payoff_score", 0)
            clip_copy["breakdown"] = data
            final_scored_clips.append(clip_copy)
            
        except Exception as e:
            # Fallback if something goes wrong
            clip_copy = clip.copy()
            clip_copy["score"] = 0
            clip_copy["breakdown"] = {"error": str(e)}
            final_scored_clips.append(clip_copy)

    # Sort highest to lowest
    final_scored_clips.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    print("\n\n🔥 TOP VIRAL CLIPS SELECTED 🔥")
    for i, clip in enumerate(final_scored_clips[:target_outputs]):
        print(f"\n--- Rank #{i+1} (Score: {clip.get('score')}/100) ---")
        print(f"Timestamps: {clip['start']}s -> {clip['end']}s")
        print(f"Reasoning: {clip.get('reasoning')}")
    end_time = datetime.now()
    print(f"Time taken to evaluate final scoring: {end_time - start_time}")
    return final_scored_clips[:target_outputs]

def generate_final_video_clips(video_file_path: str, final_clips: list, output_dir: str = "./", start_buffer: float = 0.2, end_buffer: float = 0.2) -> list:
    """
    Takes the precise timestamps from the SLM scoring engine and uses FFmpeg 
    to physically cut the original video into individual .mp4 shorts.
    Adds a small start and end buffer to prevent abrupt audio clipping.
    """
    print(f"\n🎬 Physically cutting {len(final_clips)} viral shorts with a {start_buffer}s audio buffer...")
    saved_files = []
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the base name of the original video (e.g., "test" from "test.mp4")
    base_name = os.path.splitext(os.path.basename(video_file_path))[0]

    for idx, clip in enumerate(final_clips):
        rank = idx + 1
        
        # 1. Grab original exact timestamps
        original_start = clip['start']
        original_end = clip['end']
        
        # 2. Apply the start buffer (using max() to ensure it never drops below 0.0 seconds)
        buffered_start = max(0.0, original_start - start_buffer)
        
        # 3. Apply the end buffer and calculate the new total duration
        buffered_end = original_end + end_buffer
        buffered_duration = buffered_end - buffered_start
        
        score = clip.get('score', 0)
        
        # Create a clean filename: e.g., "test_viral_rank1_score26.mp4"
        output_filename = os.path.join(output_dir, f"{base_name}_viral_rank{rank}_score{score}.mp4")
        
        print(f"  ✂️  Cutting Rank #{rank} (Buffered: {buffered_start:.2f}s to {buffered_end:.2f}s)...", end="\r")
        
        # FFmpeg command for precise, frame-accurate cutting
        command = [
            "ffmpeg", "-y",             
            "-ss", str(round(buffered_start, 2)),     # Seek to the buffered start time
            "-i", video_file_path,      
            "-t", str(round(buffered_duration, 2)),   # Use the new buffered duration
            "-c:v", "libx264",          
            "-preset", "fast",          
            "-c:a", "aac",              
            output_filename
        ]
        
        try:
            # Suppress terminal spam from FFmpeg
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            saved_files.append(output_filename)
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error cutting clip #{rank}: {e}")
            
    print(f"\n✅ Successfully generated {len(saved_files)} viral shorts in '{output_dir}'!")
    return saved_files

if _name_ == "_main_":
    video_file_path = "./podcast.mp4"
    audio_file_path = extract_audio(video_file_path)
    # transcript = transcribe_with_timestamps(audio_file_path)
    transcript_filepath = "transcript.json"
    with open(f"{transcript_filepath}", "r") as file:              # TODO remove this line later
        transcript = json.loads(file.read())                       # TODO remove this line later
    total_duration_secs = transcript[-1]["end"]
    total_minutes = total_duration_secs / 60.0
    print(f"🎬 Video detected: {total_minutes:.2f} minutes long.")
    # with open(f"{transcript_filepath}", "w+") as file:
    #     file.write(json.dumps(transcript, indent=4))

    # Options on UI
    # (15,30) - Short & Punchy
    # (30,60) - Standard Reel
    # (60,90) - Long-Form Story
    # (45,60) - YT Shorts Optimized
    options = [(15,30), (30,60), (60,90), (45,60)]
    min_duration, max_duration = options[1]

    # compressed_transcript = generate_valid_clips(transcript_file, min_duration, max_duration)
    model_path = "/home/supratik/Documents/models/Phi-3.5-mini-instruct-Q5_K_M.gguf"
    # threshold video length got skipping hotspot evaluation logic
    threshold_length = 15
    if total_minutes < threshold_length:
        print("Short video detected, bypassing hotstop evaluation to save compute")
        hotspots = [{"start": 0.0, "end": total_duration_secs}]
    else:
        print("Long video detected")
        # calculate the number of 3 min chunks
        total_chunks = total_minutes/3
        hotspots = get_hotspots(model_path, transcript_filepath, chunk_minutes = 3,
                                 keep_top_n = max(2, int(total_chunks * 0.25)))

    valid_clips = generate_valid_clips(transcript_filepath, hotspots)
    top_clips = get_intellegence_data(model_path, valid_clips)    
    # with open("./top_clips_.json", "r") as file:
    #     top_clips = json.loads(file.read())
    if top_clips:
        generated_files = generate_final_video_clips(
            video_file_path=video_file_path, 
            final_clips=top_clips, 
            output_dir="./"
        )