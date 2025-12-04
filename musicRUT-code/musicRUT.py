# ---------------------------------------------
# MusicRUT — Interactive Streamlit app that:
# 1) lets a user pick a seed song,
# 2) captures what "aspects" they like (chips),
# 3) gathers quick like/dislike feedback on previews,
# 4) optionally blends in creative inspiration (film/tv/art/books/poetry),
# 5) recommends 15 tracks, explains the curation,
# 6) generates abstract cover art with Stable Diffusion (Automatic1111),
# 7) exports to a private Spotify playlist (if configured).
# ---------------------------------------------

import streamlit as st
import pandas as pd
import requests
import os
from dotenv import load_dotenv
import lyricsgenius
from hybrid_music_rec import HybridMusicRecommender
from spotify_enricher import SpotifyEnricher
import numpy as np
from numpy.linalg import norm
import base64
from io import BytesIO
from PIL import Image

# Load env vars (e.g., GENIUS_ACCESS_TOKEN, TMDB_API_KEY, Spotify creds)
load_dotenv()

# Configure Streamlit page chrome
st.set_page_config(page_title="MusicRUT", page_icon="🎵", layout="centered")

# ---------------------------------------------------
# Logo support (centered image shown at top & bottom)
# ---------------------------------------------------
def _load_logo_image():
    # Try common locations; keep this cheap and safe
    candidates = [
        "musicRUT.png"            
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return Image.open(p)
            except Exception:
                pass
    return None

LOGO_IMG = _load_logo_image()

def show_logo_centered(img, width=220):
    # Helper to center the logo visually using a 3-col layout
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.image(img, width=width)

# spacing so the logo isn't under the fixed header
st.markdown(
    """
    <style>
      /* Add enough space below Streamlit's fixed header */
      .block-container { padding-top: 4.25rem; padding-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------
# Language helpers (strict English-only mode)
# ---------------------------------------------
def _english_only_active():
    # Treat slider at 0 as strict "English only"; also allow future hard flag via weight
    return st.session_state.get('language_pref_value', 50) == 0 or \
           st.session_state.get('english_weight', 0.5) >= 0.95

def detect_english_with_llm(track_name, artist_name):
    """Use LLM to detect if a song is in English. Returns True if English, False otherwise."""
    prompt = f"""Is the song "{track_name}" by {artist_name} sung in English?
Consider the song title and artist name. Respond with only "YES" if it's in English, or "NO" if it's in another language.
Response:"""
    try:
        response = call_ollama(prompt, timeout=5)
        if response:
            return "YES" in response.upper()
        return True  # Default to True if LLM fails
    except Exception:
        return True  # Default to True if detection fails

def _filter_english_only(df):
    """Filter dataframe to keep only English songs using LLM detection."""
    if df is None or len(df) == 0:
        return df
    
    # Use LLM to detect English songs
    english_mask = []
    for _, row in df.iterrows():
        track_name = row.get('track_name', '')
        artist_name = row.get('artists', '')
        is_english = detect_english_with_llm(track_name, artist_name)
        english_mask.append(is_english)
    
    return df[english_mask].reset_index(drop=True) if any(english_mask) else df

# before starting run cd stable-diffusion-webui and ./webui.sh --api in terminal 
# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_resource
def load_recommender():
    # Build the (cached) hybrid recommender so it doesn't reload each rerun
    return HybridMusicRecommender(use_sample=False)

@st.cache_resource
def load_spotify_enricher():
    # Try to initialize the Spotify helper that fetches album art/URLs/etc.
    # If credentials are missing, return a safe no-op stub so app still runs.
    try:
        enricher = SpotifyEnricher()
        if hasattr(enricher, 'enabled') and enricher.enabled:
            print("✓ Spotify enricher loaded successfully")
        else:
            print("⚠️ Spotify enricher loaded but not enabled")
        return enricher
    except Exception as e:
        print(f"⚠️ Spotify enricher failed to load: {e}")
        class DummyEnricher:
            enabled = False
            def enrich_dataframe(self, df, **kwargs):
                return df
        return DummyEnricher()

@st.cache_resource
def get_genius_client():
    # Build a Genius client for lyric snippets (optional)
    token = os.getenv("GENIUS_ACCESS_TOKEN")
    if not token:
        st.warning("⚠️ Genius API token not found. Lyrics analysis disabled.")
        return None
    return lyricsgenius.Genius(token, remove_section_headers=True, skip_non_songs=True)

# -----------------------------
# Helpers
# -----------------------------
def get_user_location():
    # Lightweight IP-to-country (used to bias language/region a bit)
    try:
        r = requests.get('https://ipapi.co/json/', timeout=3)
        if r.status_code == 200:
            d = r.json()
            return d.get('country_code', 'US'), d.get('country_name', 'United States')
    except Exception as e:
        print(f"Location detection error: {e}")
    return 'US', 'United States'

def get_lyrics(track_name, artist_name):
    # Fetch up to 500 chars of lyrics to enrich LLM prompts (optional)
    genius = get_genius_client()
    if not genius:
        return None
    try:
        song = genius.search_song(track_name, artist_name)
        if song and song.lyrics:
            return song.lyrics[:500]
        return None
    except Exception as e:
        print(f"Error fetching lyrics: {e}")
        return None

def call_ollama(prompt, timeout=45):
    # Call local Ollama text model (e.g., llama3) and return raw response text
    try:
        r = requests.post(
            'http://localhost:11434/api/generate',
            json={"model": "llama3", "prompt": prompt, "stream": False},
            timeout=timeout
        )
        return r.json().get('response')
    except Exception as e:
        print(f"Ollama error: {e}")
        return None

def analyze_image_with_llava(image_path):
    # Use local Ollama 'llava' vision model to extract 5–7 short aesthetic traits
    # from an uploaded image (for Art/Photography inspiration)
    try:
        with open(image_path, 'rb') as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        r = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": "llava",
                "prompt": "Analyze this image and extract 5-7 musical qualities (2-5 words each), output one per line, no preamble.",
                "images": [image_data],
                "stream": False
            },
            timeout=60
        )
        if r.status_code == 200:
            result = r.json().get('response', '')
            lines = [ln.strip().lstrip('0123456789.-*•:) ') for ln in result.split('\n')]
            out = [ln for ln in lines if ln and len(ln) < 45 and ln.count(',') <= 1]
            return out[:7] if len(out) >= 3 else None
    except Exception as e:
        print(f"llava error: {e}")
        return None
    return None

def analyze_creative_work(medium, content_data):
    # Turn a film/tv/theatre/art/book/poetry reference into 5–7 concise traits
    # that will steer the final playlist vibe
    if medium in ("Film","TV Show"):
        prompt = f"""Analyze this {medium.lower()} and output 5–7 musical characteristics (2–5 words), one per line.

Title: {content_data.get('title','Unknown')}
Overview: {content_data.get('overview','No description')}

Output:"""
    elif medium == "Theatre":
        prompt = f"""Analyze this theatrical production; output 5–7 musical characteristics (2–5 words), one per line.

Title: {content_data.get('title','Unknown')}
Overview: {content_data.get('overview','No description')}

Output:"""
    elif medium in ("Art","Photography"):
        prompt = "Analyze this visual artwork; output 5–7 musical characteristics (2–5 words), one per line.\n\nOutput:"
    elif medium == "Book":
        prompt = f"""Analyze this book; output 5–7 musical characteristics (2–5 words), one per line.

Title: {content_data.get('title','Unknown')}
Authors: {content_data.get('authors','Unknown')}
Description: {content_data.get('description','No description')}
Genres/Categories: {content_data.get('categories','')}


Output:"""
    elif medium == "Poetry":
        prompt = f"""Analyze this poetry excerpt; output 5–7 musical characteristics (2–5 words), one per line.

Excerpt:
{content_data.get('text','')[:500]}

Output:"""
    else:
        return []

    resp = call_ollama(prompt)
    if resp:
        # Strip numbering/bullets and filter meta lines
        lines = [ln.strip().lstrip('0123456789.-*•:) ') for ln in resp.split('\n')]
        out = [ln for ln in lines if ln and len(ln) < 45 and ln.count(',') <= 1
               and not any(k in ln.lower() for k in ['here are','analysis','output','characteristics','extract'])]
        return out[:7]
    return []

def search_tmdb(query, media_type='movie'):
    # TMDb search for Film/TV titles (used in Creative Inspiration step)
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        st.warning("⚠️ TMDb API key not found. Film/TV search disabled.")
        return []
    try:
        url = f"https://api.themoviedb.org/3/search/{media_type}"
        r = requests.get(url, params={'api_key': api_key,'query': query,'language':'en-US','page':1}, timeout=5)
        if r.status_code == 200:
            return r.json().get('results', [])[:5]
    except Exception as e:
        print(f"TMDb search error: {e}")
    return []

def search_google_books(query, max_results=5):
    # Google Books search for Book titles (used in Creative Inspiration step)
    try:
        url = "https://www.googleapis.com/books/v1/volumes"
        params = {
            "q": query,
            "maxResults": max_results,
            "printType": "books",
            "langRestrict": "en"
        }
        api_key = os.getenv("GOOGLE_BOOKS_API_KEY")
        if api_key:
            params["key"] = api_key

        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200:
            items = r.json().get("items", [])
            results = []
            for item in items:
                info = item.get("volumeInfo", {})
                results.append({
                    "id": item.get("id"),
                    "title": info.get("title", "Unknown title"),
                    "authors": ", ".join(info.get("authors", [])),
                    "description": info.get("description", ""),
                    "publishedDate": info.get("publishedDate", ""),
                    "categories": ", ".join(info.get("categories", [])),
                    "pageCount": info.get("pageCount")
                })
            return results
    except Exception as e:
        print(f"Google Books search error: {e}")
    return []

# -----------------------------
# Preview-aware reranking helpers
# -----------------------------
def _feat_vec_from_row(row):
    """Track row -> compact vector (0..1-ish). Works with dict or pandas Series."""
    # Safe numeric getter that tolerates both dict and pandas.Series
    def getf(k, default=0.0):
        try:
            v = float(row.get(k, default))
        except Exception:
            try:
                v = float(row[k])
            except Exception:
                v = default
        return v

    # Normalize tempo (assumes useful 60..200 bpm band)
    tempo = getf('tempo', 120.0)
    tempo01 = min(max((tempo - 60.0) / 140.0, 0.0), 1.0)

    # Map loudness (-60..0 dB) into 0..1 using a logistic curve
    loud = getf('loudness', -12.0)
    loud01 = 1.0 / (1.0 + np.exp(-0.15 * (loud + 12.0)))

    # Feature vector used for cosine similarity
    vec = np.array([
        tempo01,
        getf('energy'),
        getf('valence'),
        getf('danceability'),
        getf('acousticness'),
        getf('instrumentalness'),
        getf('speechiness'),
        getf('liveness'),
        loud01
    ], dtype='float32')
    n = norm(vec)
    return vec / n if n > 0 else vec

def _cos(a, b):
    # Cosine similarity with zero-vector guards
    na, nb = norm(a), norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def rerank_with_preview_feedback(
    candidates_df,
    preview_tracks,
    feedback_dict,
    alpha=0.35,  # Boost weight for liked tracks (empirically tuned)
    beta=0.45,   # Penalty weight for disliked tracks (empirically tuned)
    ban_threshold=0.92,  # Similarity threshold for hard filtering (0.92 = very similar)
    genre_beta=0.20  # Genre penalty weight
):
   """
    Re-rank recommendations using preview feedback.
    
    - Liked previews: boost similar tracks (+35%)
    - Disliked previews: penalize similar tracks (-45%)
    - Very similar to dislikes (>92%): remove entirely
    
    Alpha/beta values tuned through user testing.
    """
    # No-op if there's nothing to work with
    if candidates_df is None or len(candidates_df) == 0:
        return candidates_df
    if not preview_tracks or not isinstance(preview_tracks, list):
        return candidates_df

    # Partition preview tracks by user feedback and precompute their feature vectors
    liked, disliked = [], []
    for tr in preview_tracks:
        key = f"{tr['track_name']}_{tr['artists']}"
        fb = feedback_dict.get(key)
        if not fb:
            continue
        vec = _feat_vec_from_row(tr)
        if fb == 'yes':
            liked.append((vec, tr))
        elif fb == 'no':
            disliked.append((vec, tr))

    if not liked and not disliked:
        return candidates_df

    import pandas as pd
    # Track which genres were liked/disliked to inject a small penalty if needed
    disliked_genres = {t.get('track_genre') for _, t in disliked if pd.notna(t.get('track_genre'))}
    liked_genres = {t.get('track_genre') for _, t in liked if pd.notna(t.get('track_genre'))}

    rows = []
    for _, row in candidates_df.iterrows():
        v = _feat_vec_from_row(row)
        like_sim = max((_cos(v, lv) for lv, _ in liked), default=0.0)
        dislike_sim = max((_cos(v, dv) for dv, _ in disliked), default=0.0)

        # If it's extremely similar to a disliked preview, drop it outright
        if dislike_sim >= ban_threshold:
            continue

        # Light genre penalty for genres seen among dislikes (unless also liked)
        g = row.get('track_genre')
        g_pen = genre_beta if (g in disliked_genres and g not in liked_genres) else 0.0

        base = float(row.get('similarity_score', 0.0))
        new_score = base + alpha * like_sim - beta * dislike_sim - g_pen

        r = row.copy()
        r['re_rank_score'] = new_score
        rows.append(r)

    # If everything got filtered, fall back to similarity_score
    if not rows:
        return candidates_df.sort_values('similarity_score', ascending=False)

    out = pd.DataFrame(rows).sort_values(['re_rank_score', 'similarity_score'], ascending=False)
    return out

# -----------------------------
# Narrative helpers
# -----------------------------
def generate_recommendation_summary(seed_track, chips, creative_inspiration, language_pref, recommendations_df):
    """
    One lively paragraph (4–6 sentences), no metrics, explains why these 15 were picked.
    Includes 1–2 sentences tying creative inspiration if provided.
    """
    # Build a natural-language "language preference" phrase for the prompt
    selected_aspects = ", ".join(chips[:4]) if chips else "your taste"
    liked_any = any(v == 'yes' for v in st.session_state.get('preview_feedback', {}).values())

    if language_pref == 0:
        lang_phrase = "keeps everything in English"
    elif language_pref <= 40:
        lang_phrase = "leans English while staying open to a few surprises"
    elif language_pref <= 60:
        lang_phrase = "welcomes voices from anywhere"
    else:
        lang_phrase = "joyfully crosses borders and languages"

    # If the user added a creative reference, prepare detailed inspiration context
    inspiration_sentence = ""
    inspiration_context = ""
    if creative_inspiration and creative_inspiration.get('medium') != 'Skip':
        med = creative_inspiration.get('medium', '').lower()
        title = creative_inspiration.get('title', 'your pick')
        traits = creative_inspiration.get('characteristics', [])[:3]
        trait_text = ", ".join(traits) if traits else "the mood you had in mind"
        
        # Build detailed context for the LLM to reference
        inspiration_context = f"The user provided creative inspiration from {med}: '{title}'. "
        inspiration_context += f"They selected these specific characteristics: {trait_text}. "
        inspiration_context += "You MUST reference the title and these specific characteristics and explain how they shaped the playlist."
        
        inspiration_sentence = (
            f" We also wove in vibes from your {med} choice **{title}**, letting its {trait_text} "
            f"guide the colors and contours of the mix."
        )

    # Ask the LLM for a single paragraph "editor's note" explaining the curation
    prompt = f"""You are a warm, stylish music editor.
Write ONE paragraph of 4–6 sentences that explains why these 15 songs were chosen.

Context for tone (do not repeat as a list):
- Seed track: "{seed_track['track_name']}" by {seed_track['artists']}.
- The user said they care about: {selected_aspects}.
- The user {'liked some of the previews they heard' if liked_any else 'skipped the previews but still wants fresh picks'}.
- Language preference: {lang_phrase}.
- Creative inspiration: {inspiration_context if inspiration_context else "none"}.

Requirements:
- Make it fun and personal, as if you're handing them a bespoke mixtape.
- Focus on what they enjoyed or asked for (their selections and yes-clicks), and how that shaped the curation.
- If creative inspiration is present, you MUST include 1–2 sentences that SPECIFICALLY mention the creative work by its title and reference the exact characteristics the user selected. Be concrete about how those characteristics influenced the song choices.
- NO mentions of numbers, BPM, tempo, energy, valence, percentages, or popularity.
- Single flowing paragraph only.
"""

    response = call_ollama(prompt, timeout=45)
    if response and isinstance(response, str):
        # Clean up metric-y words if the model leaked them
        text = " ".join(response.strip().split())
        banned_terms = ["bpm", "tempo", "energy", "valence", "popularity", "%", "percent", "khz", "db"]
        lower = text.lower()
        if any(bt in lower for bt in banned_terms):
            sentences = [s.strip() for s in text.split(".") if s.strip()]
            sentences = [s for s in sentences if not any(bt in s.lower() for bt in banned_terms)]
            text = ". ".join(sentences)
        # If we have a good paragraph but it forgot to mention inspiration, append gently
        if len(text.split()) > 25:
            # Check if creative inspiration was mentioned specifically (by title or characteristics)
            if inspiration_sentence and creative_inspiration:
                title = creative_inspiration.get('title', '')
                mentioned = any([
                    title.lower() in text.lower() if title else False,
                    "inspiration" in text.lower(),
                    "film" in text.lower(),
                    "artwork" in text.lower(),
                    "painting" in text.lower(),
                    "movie" in text.lower(),
                    "book" in text.lower()
                ])
                if not mentioned:
                    text = text.rstrip(". ") + inspiration_sentence
            return text

    # Fallback if LLM didn't cooperate
    fallback = (
        f"Starting from **\"{seed_track['track_name']}\"** by **{seed_track['artists']}**, we leaned into what you told us you love — "
        f"{selected_aspects}. The picks you gave a thumbs-up to shaped the mood even further, "
        f"so the set flows with the same sparks you reacted to while still opening a few new doors. "
        f"{' ' + inspiration_sentence.strip() if inspiration_sentence else ''} "
        f"In the end, it {lang_phrase}, feels handcrafted for your taste, "
        f"and treats discovery like a little adventure from track to track."
    ).replace("  ", " ").strip()
    return fallback

def generate_listening_contexts(seed_track, chips, creative_inspiration, recommendations_df):
    """
    Returns EXACTLY three short 'Perfect Moments' strings.
    Format per line: '🎯 Text...' (first token is an emoji).
    No metrics or genres.
    """
    # Ask the LLM for three concise "when to listen" scenarios (with emoji)
    top_aspects = ", ".join(chips[:3]) if chips else "your taste"
    insp_snippet = ""
    if creative_inspiration and creative_inspiration.get("medium") != "Skip":
        med = creative_inspiration.get("medium", "").lower()
        title = creative_inspiration.get("title", "your pick")
        insp_snippet = f" Inspired by your {med} choice '{title}'."

    prompt = f"""You are a warm, imaginative music editor.
Write EXACTLY THREE lines suggesting perfect listening moments for this bespoke playlist.
Each line MUST:
- start with a single emoji (e.g., 🎧, 🌆, ☀️),
- be 6–14 words long,
- avoid numbers, audio jargon, or genres,
- feel vivid, everyday, and delightful.

Context (do not quote):
- Seed: "{seed_track.get('track_name','')}" by {seed_track.get('artists','')}.
- Listener cares about: {top_aspects}.
-{insp_snippet if insp_snippet else ""}

Output ONLY the three lines, one per line, no bullets, no extra text.
"""
    try:
        txt = call_ollama(prompt, timeout=30) or ""
    except Exception:
        txt = ""

    # Enforce format/cleaning rules post-hoc
    lines = [ln.strip().lstrip("0123456789.-*•:) ").strip() for ln in (txt or "").split("\n")]
    lines = [ln for ln in lines if ln and all(x not in ln.lower() for x in ["tempo","bpm","energy","valence","%","percent"])]

    cleaned = []
    for ln in lines:
        if not ln:
            continue
        first = ln.split(" ", 1)[0]
        # If line doesn't start with emoji, add a generic one
        if len(first) == 0 or first.isalnum():
            ln = "🎵 " + ln
        cleaned.append(ln)

    cleaned = cleaned[:3]
    # If fewer than 3 valid lines, fill with tasteful defaults
    while len(cleaned) < 3:
        base = [
            "🎧 Quiet commute, city humming while the songs color your thoughts",
            "🌅 Soft morning light as the mix eases you into the day",
            "🌙 Late-night unwind, headphones on, letting scenes play behind your eyelids"
        ]
        if insp_snippet and "🎬" not in "".join(cleaned):
            base[1] = "🎬 Night-in mood, your inspiration casting a glow over each transition"
        for b in base:
            if len(cleaned) >= 3: break
            if b not in cleaned:
                cleaned.append(b)
    return cleaned[:3]

# -----------------------------
# Stable Diffusion (Automatic1111)
# -----------------------------
def generate_playlist_artwork_prompt(seed_track, chips, creative_inspiration, recommendations_df):
    # Aggregate audio stats + genre to build a descriptive SD prompt,
    # optionally weaving in traits from the creative inspiration step
    avg_energy = recommendations_df['energy'].mean()
    avg_valence = recommendations_df['valence'].mean()
    avg_tempo = recommendations_df['tempo'].mean()
    unique_genres = list(recommendations_df['track_genre'].unique()[:3])

    creative_context = ""
    if creative_inspiration and creative_inspiration.get('medium') != 'Skip':
        medium = creative_inspiration['medium']
        title = creative_inspiration.get('title', 'Unknown')
        chars = creative_inspiration.get('characteristics', [])
        creative_context = f"Inspired by {medium}: {title}. Visual qualities: {', '.join(chars[:3])}."

    # Map avg energy/valence into mood/palette/style buckets
    if avg_energy > 0.7 and avg_valence > 0.6:
        mood = "high-energy, euphoric, vibrant, electric"
        colors = "bright neon colors, electric blues and magentas, radiant yellows, vibrant oranges"
        atmosphere = "energetic motion, dynamic light rays, pulsing waves"
        style_hints = "vaporwave, cyberpunk, festival energy"
    elif avg_energy > 0.7 and avg_valence < 0.4:
        mood = "intense, dark, powerful, aggressive"
        colors = "deep reds, dark purples, stark blacks with crimson accents"
        atmosphere = "dramatic storm energy, lightning, chaotic movement"
        style_hints = "industrial, dark surrealism, aggressive abstract"
    elif avg_energy < 0.4 and avg_valence < 0.4:
        mood = "melancholic, introspective, ethereal"
        colors = "muted blues, soft grays, deep indigos, fog white"
        atmosphere = "misty, dreamlike, floating particles, soft focus"
        style_hints = "ethereal dreamscape, melancholic minimalism, ambient"
    elif avg_energy < 0.4 and avg_valence > 0.6:
        mood = "gentle, uplifting, peaceful, warm"
        colors = "soft pastels, golden yellows, gentle pinks, warm oranges"
        atmosphere = "sun-dappled glow, floating lightness, peaceful radiance"
        style_hints = "impressionist warmth, gentle abstract, minimalism"
    else:
        mood = "balanced, atmospheric, flowing"
        colors = "cool teals, warm ambers, earth tones, gradients"
        atmosphere = "fluid motion, harmonious flow, balanced light"
        style_hints = "contemporary abstract, balanced composition, modern minimalism"

    # Gentle genre-specific texture hints (kept abstract)
    genre_hints = ""
    genre_lower = seed_track['track_genre'].lower()
    if 'electronic' in genre_lower or 'edm' in genre_lower: genre_hints = ", digital glitch art, geometric patterns, synth waves"
    elif 'rock' in genre_lower or 'indie' in genre_lower: genre_hints = ", textured layers, grungy aesthetics, organic chaos"
    elif 'jazz' in genre_lower or 'soul' in genre_lower: genre_hints = ", smooth gradients, vintage grain, warm analog feel"
    elif 'classical' in genre_lower: genre_hints = ", elegant flowing forms, orchestral movement, timeless composition"
    elif 'hip hop' in genre_lower or 'rap' in genre_lower: genre_hints = ", bold geometric shapes, urban textures, street art influence"

    # Compose the text prompt that we ask Ollama to refine/expand
    ollama_prompt = f"""Create a Stable Diffusion prompt for abstract playlist artwork.

PLAYLIST:
- Seed: "{seed_track['track_name']}" by {seed_track['artists']}
- Primary Genre: {seed_track['track_genre']}
- Exploring: {', '.join(unique_genres)}
- Vibes: {', '.join(chips[:3])}
- Energy: {avg_energy:.2f} | Mood: {avg_valence:.2f} | Tempo: {avg_tempo:.0f} BPM
- Creative: {creative_context or 'None'}

STRUCTURE: [Core abstract concept], [palette: {colors}], [atmosphere: {atmosphere}], [texture/movement], [style: {style_hints}{genre_hints}], [technical quality]

Rules:
- Abstract only (no people/text/logos/instruments)
- <= 200 words
- Output ONLY the Stable Diffusion prompt
"""
    # Ask the LLM for a richer SD prompt; otherwise build a safe fallback
    resp = call_ollama(ollama_prompt, timeout=30)
    if resp and len(resp) > 50:
        prompt = resp.strip().strip('"\'')
        for p in ['here is',"here's",'prompt:','stable diffusion prompt:','the prompt is']:
            if prompt.lower().startswith(p): prompt = prompt[len(p):].strip()
        if "highly detailed" not in prompt.lower():
            prompt += ", highly detailed, 8k resolution, professional digital artwork, masterpiece quality"
        negative_prompt = "text, words, letters, logos, people, faces, humans, instruments, photograph, photorealistic, album title, artist name, song lyrics"
        return prompt, negative_prompt

    # Fallback short prompt/negative prompt
    fallback_prompt = f"Abstract {mood} artwork, {colors}, {atmosphere}, {style_hints}{genre_hints}, album cover aesthetic, digital art, highly detailed, 8k resolution, professional artwork, masterpiece quality"
    fallback_negative = "text, words, people, faces, photograph, realistic, instruments, musicians"
    return fallback_prompt, fallback_negative

def generate_artwork_with_automatic1111(prompt, negative_prompt, width=768, height=768):
    # Call Automatic1111 txt2img API and return a PIL image (or error message)
    try:
        r = requests.post(
            'http://127.0.0.1:7860/sdapi/v1/txt2img',
            json={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": 30,
                "cfg_scale": 7.5,
                "width": width,
                "height": height,
                "sampler_name": "DPM++ 2M Karras",
                "seed": -1,
                "enable_hr": False
            },
            timeout=120
        )
        if r.status_code == 200:
            b64 = r.json()['images'][0]
            img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
            return img, None
        return None, f"API returned status code {r.status_code}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to Automatic1111. Start it with --api at http://127.0.0.1:7860"
    except requests.exceptions.Timeout:
        return None, "Image generation timed out."
    except Exception as e:
        return None, f"Error: {str(e)}"

# -----------------------------
# Chips & previews
# -----------------------------
def generate_track_chips(track_data):
    # Ask the LLM for 8 concise traits specific to the seed track (uses lyrics if available)
    lyrics = get_lyrics(track_data['track_name'], track_data['artists'])
    lyrics_section = f"\nLyrics excerpt:\n{lyrics}\n" if lyrics else "(Lyrics not available)"
    prompt = f"""Output EXACTLY 8 concise characteristics (2–5 words) about this track, one per line.

Title: {track_data['track_name']}
Artist: {track_data['artists']}
Energy: {track_data['energy']:.2f} | Acousticness: {track_data['acousticness']:.2f}
Valence: {track_data['valence']:.2f} | Tempo: {int(track_data['tempo'])} BPM
Danceability: {track_data['danceability']:.2f} | Instrumentalness: {track_data['instrumentalness']:.2f}

{lyrics_section}

Output:"""
    resp = call_ollama(prompt)
    if resp:
        # Clean and keep <= 8 short lines
        chips = []
        for ln in resp.split('\n'):
            ln = ln.strip().lstrip('0123456789.-*•:) ')
            if ln and len(ln) <= 45 and ln.count(',') <= 1 and ';' not in ln:
                if not any(k in ln.lower() for k in ['output','analysis','characteristics']):
                    chips.append(ln)
        chips = chips[:8]
        if len(chips) >= 6: return chips
    # If LLM fails, build reasonable defaults from audio features
    return generate_fallback_chips(track_data, lyrics)

def get_deezer_preview(track_name, artist_name):
    # Fetch a 30s MP3 preview URL from Deezer (if available)
    try:
        q = f"{track_name} {artist_name}".replace("&","and")
        r = requests.get('https://api.deezer.com/search', params={'q': q, 'limit': 1}, timeout=5)
        if r.status_code == 200 and r.json().get('data'):
            return r.json()['data'][0].get('preview')
    except Exception as e:
        print(f"Deezer preview error: {e}")
    return None

def get_diverse_preview_tracks(recommender, seed_features, track_data, selected_chips, num_previews=3):
    # Pull 3 preview tracks that are strong matches but genre-diverse relative to each other
    genre_lower = track_data['track_genre'].lower()
    is_indie_rock = any(k in genre_lower for k in ['indie','alternative','rock'])
    popularity_range = (15, 50) if is_indie_rock else (15, 55)

    recs = recommender.recommend(
        user_profile=seed_features, n=50, diversity_weight=0.8,
        bpm_range=(track_data['tempo'] - 30, track_data['tempo'] + 30),
        popularity_range=popularity_range, similarity_method='hybrid',
        seed_genre=track_data['track_genre'], seed_artist=track_data['artists'],
        user_location=st.session_state.user_location, selected_chips=selected_chips,
        english_weight=st.session_state.get('english_weight', 0.5)
    )
    if len(recs) == 0: return []
    recs = recs[recs['track_name'] != track_data['track_name']]

    # English-only hard filter for previews (strict when slider == 0)
    if _english_only_active():
        recs = _filter_english_only(recs)
        if len(recs) == 0:
            return []

    # Prefer one per genre if possible, then fill remaining slots
    out, used = [], set()
    recs = recs.sort_values('similarity_score', ascending=False)
    for _, row in recs.iterrows():
        g = row['track_genre']
        if g not in used and len(out) < num_previews:
            out.append(row); used.add(g)
        if len(out) >= num_previews: break
    if len(out) < num_previews:
        for _, row in recs.iterrows():
            if not any(t['track_name'] == row['track_name'] for t in out):
                out.append(row)
                if len(out) >= num_previews: break
    return out

def generate_fallback_chips(track_data, lyrics=None):
    # Deterministic backup traits when LLM isn't available
    chips = []
    is_electronic = track_data['acousticness'] < 0.3
    chips += (["Electronic production","Synth-based sound"] if is_electronic else ["Acoustic instrumentation"])
    if track_data['energy'] > 0.7: chips += ["High energy","Intense vibe"]
    elif track_data['energy'] < 0.3: chips += ["Calm atmosphere","Mellow energy"]
    else: chips += ["Moderate energy"]
    if track_data['valence'] > 0.7: chips += ["Uplifting mood"]
    elif track_data['valence'] < 0.3: chips += ["Melancholic feel"]
    else: chips += ["Balanced emotions"]
    if track_data['danceability'] > 0.7: chips += ["Very danceable"]
    if lyrics: chips += ["Lyrical content"]
    return chips[:8]

def generate_playlist_title(seed_track, chips, creative_inspiration):
    # Request a short creative title from the LLM; fall back to "<Seed> Vibes"
    creative_context = ""
    if creative_inspiration and creative_inspiration.get('medium') != 'Skip':
        creative_context = f"Creative inspiration: {creative_inspiration['medium']} - {creative_inspiration.get('title','Unknown')}"
    prompt = f"""Generate a creative Spotify playlist title (3–6 words).
Seed: "{seed_track['track_name']}" by {seed_track['artists']}
Vibes: {', '.join(chips[:3])}
{creative_context or 'No creative inspiration'}
Output ONLY the title:"""
    resp = call_ollama(prompt, timeout=20)
    if resp:
        title = resp.strip().strip('"\'')
        for p in ['here is','the title is','how about']:
            if title.lower().startswith(p): title = title[len(p):].strip()
        if 0 < len(title) < 60:
            return title
    return f"{seed_track['track_name']} Vibes"

def create_spotify_playlist(spotify_enricher, track_ids, playlist_name, playlist_description="", cover_image=None):
    # Create a private playlist, add tracks (batched 100), optionally upload cover art
    try:
        if not hasattr(spotify_enricher, 'sp') or not spotify_enricher.sp:
            return None, "Spotify API not configured"
        sp = spotify_enricher.sp
        user_id = sp.current_user()['id']
        playlist = sp.user_playlist_create(user=user_id, name=playlist_name, public=False, description=playlist_description)
        playlist_id = playlist['id']
        playlist_url = playlist['external_urls']['spotify']

        # Deduplicate track URIs and add in batches
        uris = [f"spotify:track:{tid}" for tid in track_ids if tid]
        for i in range(0, len(uris), 100):
            sp.playlist_add_items(playlist_id, uris[i:i+100])

        # Optional: upload a 640x640 JPEG cover (must be Base64 < 256KB)
        if cover_image is not None:
            try:
                img = cover_image.resize((640,640), Image.Resampling.LANCZOS)
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=85, optimize=True)
                if len(buf.getvalue()) > 256000:
                    buf = BytesIO()
                    img.save(buf, format="JPEG", quality=70, optimize=True)
                b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                sp.playlist_upload_cover_image(playlist_id, b64)
                print("✓ Cover image uploaded")
            except Exception as e:
                print(f"⚠️ Cover upload failed: {e}")
        return playlist_url, None
    except Exception as e:
        return None, str(e)

# -----------------------------
# Session
# -----------------------------
# Initialize session-state keys on first load
if 'step' not in st.session_state:
    st.session_state.step = 'seed_selection'
    st.session_state.selected_chips = []
    st.session_state.preview_feedback = {}
    st.session_state.user_location = get_user_location()[0]
    st.session_state.creative_inspiration = None
    st.session_state.liked_preview_ids = []
    st.session_state.playlist_title = None
    st.session_state.playlist_artwork = None

# Build core services (cached) for this run
recommender = load_recommender()
spotify_enricher = load_spotify_enricher()

# Sidebar status panel: Spotify enrichment + Stable Diffusion availability
# ----------------------------------------------------------------------
if not spotify_enricher.enabled:
    st.sidebar.info("💡 Spotify enrichment disabled (no album art fetch). Configure Spotify API to enable.")
else:
    st.sidebar.success("✓ Spotify enrichment active")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 Playlist Artwork")
try:
    if requests.get('http://127.0.0.1:7860/sdapi/v1/sd-models', timeout=2).status_code == 200:
        st.sidebar.success("✓ Stable Diffusion active")
    else:
        st.sidebar.warning("⚠️ Stable Diffusion API error")
except:
    st.sidebar.info("💡 Start Automatic1111 with `--api` to enable artwork generation")

# -----------------------------
# STEP 1: Seed selection
# -----------------------------
# User searches seed song; we store selected row (optionally enriched by Spotify)
if st.session_state.step == 'seed_selection':
    # Show the logo instead of a text title at the very top
    if LOGO_IMG:
        show_logo_centered(LOGO_IMG, width=220)
    else:
        st.title("MusicRUT🎵")  # fallback if logo not found

    st.write("Let's find new music to get you out of your rut. Start by choosing a song you like.")

    q = st.text_input("Search for a song:", placeholder="e.g.,Somebody Else")
    if q:
        # Simple local search over recommender's dataframe
        results = recommender.df[recommender.df['track_name'].str.contains(q, case=False, na=False)].head(10)
        if len(results) > 0:
            st.write(f"**Found {len(results)} tracks:**")
            for idx, row in results.iterrows():
                c1, c2 = st.columns([4,1])
                with c1:
                    st.write(f"**{row['track_name']}** by {row['artists']}")
                with c2:
                    # Selecting sets the seed and advances to chip selection
                    if st.button("Select →", key=f"select_{idx}", width='stretch'):
                        d = row.to_dict()
                        if spotify_enricher.enabled:
                            try:
                                edf = spotify_enricher.enrich_dataframe(pd.DataFrame([d]), batch_size=1)
                                if len(edf) > 0:
                                    d = edf.iloc[0].to_dict()
                            except Exception as e:
                                print(f"Spotify enrichment failed: {e}")
                        st.session_state.seed_track = d
                        st.session_state.step = 'chip_selection'
                        st.rerun()
                st.divider()
        else:
            st.warning("No tracks found. Try a different search term.")

# -----------------------------
# STEP 2: Chip selection
# -----------------------------
# Generate or display 6–8 "chips" (short traits) and let the user choose 2–4
elif st.session_state.step == 'chip_selection':
    t = st.session_state.seed_track
    c1, c2 = st.columns([1,3])
    with c1:
        if t.get('album_art'): st.image(t['album_art'], width='content')
    with c2:
        st.title(f"🎵 {t['track_name']}")
        st.write(f"**by {t['artists']}**")

    st.divider()
    # Only compute chips once per seed
    if 'chips' not in st.session_state:
        with st.spinner("Analyzing track characteristics..."):
            st.session_state.chips = generate_track_chips(t)
    chips = st.session_state.chips

    st.markdown("### 🎯 What aspects do you want in similar tracks?")
    st.write("Select 2-4 characteristics that matter most to you:")
    cols = st.columns(4)
    for i, chip in enumerate(chips):
        col = cols[i % 4]
        with col:
            sel = chip in st.session_state.selected_chips
            if st.button(f"{'✓ ' if sel else ''}{chip}", key=f"chip_{i}", type="primary" if sel else "secondary", width='stretch'):
                # Toggle selection with a max of 4 chips
                if sel:
                    st.session_state.selected_chips.remove(chip)
                else:
                    if len(st.session_state.selected_chips) < 4:
                        st.session_state.selected_chips.append(chip)
                    else:
                        st.warning("You can select up to 4 aspects")
                st.rerun()
    st.divider()
    # Navigation buttons
    b1, b2 = st.columns([1,3])
    with b1:
        if st.button("← Back", width='content'):
            st.session_state.step = 'seed_selection'
            st.session_state.selected_chips = []
            if 'chips' in st.session_state: del st.session_state.chips
            st.rerun()
    with b2:
        if len(st.session_state.selected_chips) >= 2:
            if st.button("Continue to Language Preference →", type="primary", width='stretch'):
                st.session_state.step = 'language_preference'
                st.rerun()
        else:
            st.button(f"Select at least 2 aspects ({len(st.session_state.selected_chips)}/2)", disabled=True, width='stretch')

# -----------------------------
# STEP 3: Language preference
# -----------------------------
# Map a 0..100 slider to an english_weight used during recommendation
elif st.session_state.step == 'language_preference':
    t = st.session_state.seed_track
    selected = st.session_state.selected_chips
    st.title("🌍 Language Preference")
    c1, c2 = st.columns([1,3])
    with c1:
        if t.get('album_art'): st.image(t['album_art'], width='content')
    with c2:
        st.write(f"**Seed track:** {t['track_name']} by {t['artists']}")
        st.caption(f"**Selected aspects:** {', '.join(selected)}")
    st.divider()

    # Slider is deliberately label-less; we show captions left/right
    language_pref = st.slider(" ", 0, 100, 50, 10, format=" ", label_visibility="collapsed")
    d1, _, _, d4 = st.columns(4)
    with d1: st.caption("⬅️ **English only**")
    with d4: st.caption("**All languages welcome!** ➡️")

    # Map to an english_weight (bias towards English tracks if desired)
    if language_pref == 0: english_weight = 1.0; st.success("🎯 **English Only**")
    elif language_pref <= 20: english_weight = 0.90; st.info("📍 **Mostly English**")
    elif language_pref <= 40: english_weight = 0.70; st.info("📍 **Prefer English**")
    elif language_pref <= 60: english_weight = 0.50; st.info("📍 **Balanced Mix**")
    elif language_pref <= 80: english_weight = 0.30; st.info("📍 **Open to International**")
    else: english_weight = 0.15; st.info("📍 **Love International**")

    st.session_state.english_weight = english_weight
    st.session_state.language_pref_value = language_pref

    st.divider()
    # Navigation
    b1, b2 = st.columns([1,3])
    with b1:
        if st.button("← Back", width='content'):
            st.session_state.step = 'chip_selection'; st.rerun()
    with b2:
        if st.button("Continue to Preview →", type="primary", width='stretch'):
            st.session_state.step = 'preview'; st.rerun()

# -----------------------------
# STEP 4: Preview & Refine
# -----------------------------
# Play up to 3 30s previews and capture thumbs-up/down to influence re-ranking
elif st.session_state.step == 'preview':
    t = st.session_state.seed_track
    selected = st.session_state.selected_chips
    st.title("🎧 Preview & Refine")
    c1, c2 = st.columns([1,3])
    with c1:
        if t.get('album_art'): st.image(t['album_art'], width='content')
    with c2:
        st.write(f"**Seed track:** {t['track_name']} by {t['artists']}")
        st.caption(f"**Selected aspects:** {', '.join(selected)}")
    st.divider()
    st.markdown("I'll play you **3 tracks** — just say if you like them.")

    # Banner if English-only mode is effectively on
    if _english_only_active():
        st.success("🎯 **English-only mode active**")

    # First entry: compute preview candidates once
    if 'preview_tracks' not in st.session_state:
        with st.spinner("Finding preview tracks..."):
            seed = np.array([t['tempo'],t['energy'],t['valence'],t['danceability'],t['acousticness'],t['instrumentalness'],t['speechiness'],t['liveness'],t['loudness']])
            st.session_state.preview_tracks = get_diverse_preview_tracks(recommender, seed, t, selected, num_previews=3)

    preview_tracks = st.session_state.preview_tracks
    if len(preview_tracks) == 0:
        # If we can't find any, encourage going back to tweak settings
        st.error("No preview tracks found. Try adjusting your preferences.")
        if st.button("← Back", width='content'):
            st.session_state.step = 'language_preference'; st.rerun()
        st.stop()

    st.divider()
    # Render each preview card with Yes/No buttons
    all_feedback = True
    for idx, track in enumerate(preview_tracks):
        tid = f"{track['track_name']}_{track['artists']}"
        st.markdown(f"### Track {idx + 1}")
        col1, col2 = st.columns([3,1])
        with col1:
            st.subheader(f"🎵 {track['track_name']}")
            st.caption(f"**{track['artists']}**")
            preview_url = get_deezer_preview(track['track_name'], track['artists'])
            if preview_url:
                st.audio(preview_url, format='audio/mp3')
            else:
                # If no Deezer preview, offer a YouTube search link
                st.caption("⚠️ Preview not available for this track")
                yt = f"https://www.youtube.com/results?search_query={track['track_name']}+{track['artists']}".replace(" ","+")
                st.markdown(f"[🔍 Search on YouTube]({yt})")
        with col2:
            st.caption("Do you like this?")
            c_yes, c_no = st.columns(2)
            fb = st.session_state.preview_feedback.get(tid)
            with c_yes:
                if st.button("👍 Yes" if fb != 'yes' else "✅ Yes", key=f"yes_{idx}", type="primary" if fb=='yes' else "secondary", width='stretch'):
                    st.session_state.preview_feedback[tid] = 'yes'
                    # Remember liked preview Spotify IDs to prioritize in final playlist
                    sid = track.get('spotify_id')
                    if sid and sid not in st.session_state.liked_preview_ids:
                        st.session_state.liked_preview_ids.append(sid)
                    st.rerun()
            with c_no:
                if st.button("👎 No" if fb != 'no' else "❌ No", key=f"no_{idx}", type="primary" if fb=='no' else "secondary", width='stretch'):
                    st.session_state.preview_feedback[tid] = 'no'; st.rerun()
            if not fb: all_feedback = False
        st.divider()

    # Navigation for this step
    b1, b2 = st.columns([1,3])
    with b1:
        if st.button("← Back", width='content'):
            st.session_state.step = 'language_preference'
            if 'preview_tracks' in st.session_state: del st.session_state.preview_tracks
            st.rerun()
    with b2:
        if all_feedback:
            if st.button("Continue to Creative Inspiration →", type="primary", width='stretch'):
                st.session_state.step = 'creative_inspiration'; st.rerun()
        else:
            n = len(st.session_state.preview_feedback)
            st.button(f"Rate all tracks to continue ({n}/3)", disabled=True, width='stretch')

# -----------------------------
# STEP 3.5: Creative Inspiration
# -----------------------------
# Optional step to capture an external creative reference and derive traits
elif st.session_state.step == 'creative_inspiration':
    t = st.session_state.seed_track
    selected = st.session_state.selected_chips
    st.title("One last step to understand what inspires you")
    c1, c2 = st.columns([1,3])
    with c1:
        if t.get('album_art'): st.image(t['album_art'], width='content')
    with c2:
        st.write(f"**Seed track:** {t['track_name']} by {t['artists']}")
        st.caption(f"**Selected aspects:** {', '.join(selected)}")
    st.divider()

    st.markdown("""
    ### What other art have you been into lately?
    Share a film, show, artwork, book, or poetry. I’ll blend its aesthetic into your playlist.
    *(Optional — skip if you're ready!)*""")

    # If already set, show a success note and allow moving on
    if st.session_state.creative_inspiration and st.session_state.creative_inspiration.get('medium') != 'Skip':
        medium = st.session_state.creative_inspiration['medium']
        title = st.session_state.creative_inspiration.get('title', 'Unknown')
        st.success(f"✓ Added {medium} inspiration: **{title}**")
        st.caption("Click 'Get my playlist' below or add a different medium.")
        st.markdown("---")

    # Medium chooser
    cols = st.columns(3)
    opts = [("🎬 Film","Film"),("📺 TV Show","TV Show"),("🎭 Theatre","Theatre"),("🎨 Visual Art","Art"),("📸 Photography","Photography"),("📚 Book","Book"),("✍️ Poetry","Poetry")]
    for i,(label, medium) in enumerate(opts):
        with cols[i % 3]:
            if st.button(label, key=f"medium_{medium}", type="secondary", width='stretch'):
                st.session_state.selected_medium = medium; st.rerun()

    # Depending on medium, either search TMDb, upload an image, or enter text
    if 'selected_medium' in st.session_state:
        sel = st.session_state.selected_medium
        st.divider()
        st.markdown(f"### {sel} Selection")
        if sel in ["Film","TV Show"]:
            s = st.text_input(f"Search for a {sel.lower()}:", placeholder="e.g., Blade Runner 2049")
            if s:
                media_type = 'movie' if sel == "Film" else 'tv'
                res = search_tmdb(s, media_type)
                if res:
                    st.write(f"**Found {len(res)} results:**")
                    for r in res:
                        title = r.get('title') or r.get('name','Unknown')
                        overview = r.get('overview','No description available')
                        release = r.get('release_date') or r.get('first_air_date','')
                        year = release[:4] if release else ''
                        c1,c2 = st.columns([4,1])
                        with c1:
                            st.write(f"**{title}** {f'({year})' if year else ''}")
                            st.caption(overview[:150] + ("..." if len(overview)>150 else ""))
                        with c2:
                            # Clicking "Next" analyzes the selected result via LLM
                            if st.button("Next →", key=f"tmdb_{r.get('id',title)}", width='stretch'):
                                with st.spinner("Analyzing creative work..."):
                                    data = {'title': title,'overview': overview,'genre': ', '.join([str(g) for g in r.get('genre_ids', [])])}
                                    chars = analyze_creative_work(sel, data)
                                    st.session_state.creative_inspiration = {'medium': sel,'title': title,'content': data,'characteristics': chars}
                                    del st.session_state.selected_medium; st.rerun()
                        st.divider()
                else:
                    st.warning("No results found. Try a different search term.")
        elif sel == "Theatre":
            # TMDb-backed search, similar to Film/TV
            s = st.text_input("Search for a theatre production:", placeholder="e.g., Hamilton, Waiting for Godot")
            if s:
                # Use movie search as a proxy for recorded productions / adaptations
                res = search_tmdb(s, media_type='movie')
                if res:
                    st.write(f"**Found {len(res)} results:**")
                    for r in res:
                        title = r.get('title') or r.get('name','Unknown')
                        overview = r.get('overview','No description available')
                        release = r.get('release_date') or r.get('first_air_date','')
                        year = release[:4] if release else ''
                        c1, c2 = st.columns([4,1])
                        with c1:
                            st.write(f"**{title}** {f'({year})' if year else ''}")
                            st.caption(overview[:150] + ("..." if len(overview) > 150 else ""))
                        with c2:
                            if st.button("Next →", key=f"theatre_tmdb_{r.get('id', title)}", width='stretch'):
                                with st.spinner("Analyzing theatrical production..."):
                                    data = {
                                        'title': title,
                                        'overview': overview,
                                        'genre': ', '.join([str(g) for g in r.get('genre_ids', [])])
                                    }
                                    chars = analyze_creative_work(sel, data)
                                    st.session_state.creative_inspiration = {
                                        'medium': sel,
                                        'title': title,
                                        'content': data,
                                        'characteristics': chars
                                    }
                                    del st.session_state.selected_medium
                                    st.rerun()
                        st.divider()
                else:
                    st.warning("No results found. Try a different search term.")

        elif sel in ["Art","Photography"]:
            # Image upload (optionally analyzed by llava if installed)
            st.write("📎 **Upload an image:**")
            try:
                test = requests.get('http://localhost:11434/api/tags', timeout=2)
                models = [m['name'] for m in test.json().get('models', [])]
                llava_ok = any('llava' in m for m in models)
            except:
                llava_ok = False
            if not llava_ok:
                st.warning("⚠️ llava vision model not found. Install with: `ollama pull llava`")
                st.caption("Without llava, we'll use text-based inference.")
            else:
                st.success("✓ llava vision model ready")
            up = st.file_uploader("Upload image", type=['png','jpg','jpeg'], key="art_upload")
            if up is not None:
                st.image(up, caption="Your uploaded image", width='stretch')
                if st.button("Next →", type="primary", key=f"analyze_{sel}", width='stretch'):
                    with st.spinner("🎨 Analyzing visual aesthetic..."):
                        import tempfile
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                            tmp.write(up.getvalue()); tmp_path = tmp.name
                        chars = analyze_image_with_llava(tmp_path) if llava_ok else None
                        if not chars:
                            st.info("💡 Using text-based inference.")
                            chars = ["atmospheric textures","visual rhythm","color-driven mood","compositional flow","emotional resonance"]
                        try: os.unlink(tmp_path)
                        except: pass
                        st.session_state.creative_inspiration = {'medium': sel,'title': "Visual artwork",'content': {'description': "AI-analyzed image"},'characteristics': chars}
                        del st.session_state.selected_medium; st.rerun()
            else:
                st.caption("👆 Upload an image to continue")
            
        elif sel == "Book":
            # Google Books–backed search and analysis
            title_query = st.text_input("Search for a book:", placeholder="e.g., The Great Gatsby")
            if title_query:
                res = search_google_books(title_query)
                if res:
                    st.write(f"**Found {len(res)} results:**")
                    for b in res:
                        c1, c2 = st.columns([4,1])
                        with c1:
                            line_title = b.get("title", "Unknown title")
                            authors = b.get("authors", "")
                            year = (b.get("publishedDate") or "")[:4]
                            title_line = line_title + (f" ({year})" if year else "")
                            st.write(f"**{title_line}**")
                            if authors:
                                st.caption(authors)
                            desc = b.get("description") or ""
                            if desc:
                                st.caption(desc[:160] + ("..." if len(desc) > 160 else ""))
                        with c2:
                            if st.button("Next →", type="primary", key=f"googlebooks_{b['id']}", width='stretch'):
                                with st.spinner("Analyzing literary work..."):
                                    data = {
                                        'title': b.get("title"),
                                        'authors': b.get("authors"),
                                        'description': b.get("description"),
                                        'categories': b.get("categories"),
                                        'publishedDate': b.get("publishedDate"),
                                        'pageCount': b.get("pageCount")
                                    }
                                    chars = analyze_creative_work(sel, data)
                                    st.session_state.creative_inspiration = {
                                        'medium': sel,
                                        'title': b.get("title", "Unknown"),
                                        'content': data,
                                        'characteristics': chars
                                    }
                                    del st.session_state.selected_medium
                                    st.rerun()
                        st.divider()
                else:
                    st.warning("No results found. Try a different search term.")

        elif sel == "Poetry":
            # Paste excerpt, then LLM analysis
            text = st.text_area("Poetry excerpt:", placeholder="Paste a few lines...", height=150)
            if st.button("Next →", type="primary", key="analyze_poetry", width='stretch'):
                if text and len(text) > 20:
                    with st.spinner("Analyzing poetic work..."):
                        data = {'text': text}
                        chars = analyze_creative_work(sel, data)
                        st.session_state.creative_inspiration = {'medium': sel,'title': "Poetry excerpt",'content': data,'characteristics': chars}
                        del st.session_state.selected_medium; st.rerun()
                else:
                    st.warning("Please paste at least a few lines of poetry.")

    st.divider()
    # Footer controls: continue to recommendations or go back
    if 'selected_medium' not in st.session_state:
        c_left, c_right = st.columns([3,1])
        with c_left:
            btn = "🎵 Get my playlist" if (st.session_state.creative_inspiration and st.session_state.creative_inspiration.get('medium') != 'Skip') else "⏭️ Skip to my playlist"
            if st.button(btn, type="primary", width='stretch'):
                if not st.session_state.creative_inspiration:
                    st.session_state.creative_inspiration = {'medium': 'Skip'}
                st.session_state.step = 'recommendations'; st.rerun()
        with c_right:
            if st.button("← Back", width='stretch'):
                st.session_state.step = 'preview'; st.rerun()
    else:
        if st.button("← Back to Medium Selection", width='content'):
            del st.session_state.selected_medium; st.rerun()

# -----------------------------
# STEP 5: Recommendations (15, centered, title shown & persisted)
# -----------------------------
# Build final 15 recs, generate cover art + editorial summary, allow export to Spotify
elif st.session_state.step == 'recommendations':
    t = st.session_state.seed_track
    selected = st.session_state.selected_chips

    # Compute recommendations, re-rank using preview feedback, enrich album art, cache result
    if 'final_recommendations' not in st.session_state:
        with st.spinner("🎵 Getting you out of your music rut..."):
            seed = np.array([t['tempo'],t['energy'],t['valence'],t['danceability'],t['acousticness'],t['instrumentalness'],t['speechiness'],t['liveness'],t['loudness']])
            chips_all = selected.copy()
            if st.session_state.creative_inspiration and st.session_state.creative_inspiration.get('medium') != 'Skip':
                chips_all.extend(st.session_state.creative_inspiration.get('characteristics', [])[:3])

            # Request more results initially to ensure we have enough after filtering
            initial_n = 50 if _english_only_active() else 30
            
            results = recommender.recommend(
                user_profile=seed, n=initial_n, diversity_weight=0.5,
                bpm_range=(t['tempo'] - 20, t['tempo'] + 20),
                popularity_range=(15, 55), similarity_method='hybrid',
                seed_genre=t['track_genre'], seed_artist=t['artists'],
                user_location=st.session_state.user_location,
                selected_chips=chips_all, english_weight=st.session_state.get('english_weight', 0.5)
            )
            results = results[results['track_name'] != t['track_name']]

            # English-only hard filter for final recs (strict when slider == 0)
            if _english_only_active():
                results = _filter_english_only(results)
                
                # If we don't have enough results after filtering, get more
                if len(results) < 20:
                    st.warning("⚠️ Limited English-only results. Expanding search...")
                    backup_results = recommender.recommend(
                        user_profile=seed, n=100, diversity_weight=0.5,
                        bpm_range=(t['tempo'] - 30, t['tempo'] + 30),
                        popularity_range=(10, 70), similarity_method='hybrid',
                        seed_genre=t['track_genre'], seed_artist=t['artists'],
                        user_location=st.session_state.user_location,
                        selected_chips=chips_all, english_weight=st.session_state.get('english_weight', 0.5)
                    )
                    backup_results = backup_results[backup_results['track_name'] != t['track_name']]
                    backup_results = _filter_english_only(backup_results)
                    
                    # Combine results, keeping original ordering priority
                    results = pd.concat([results, backup_results]).drop_duplicates(subset=['track_name', 'artists']).head(30)

            # Try to fetch album art/Spotify links
            if spotify_enricher.enabled and len(results) > 0:
                try:
                    with st.spinner("Fetching album artwork..."):
                        results = spotify_enricher.enrich_dataframe(results.copy(), batch_size=10)
                    # Optional: re-apply English filter if enrichment added rows/changed ordering
                    if _english_only_active():
                        results = _filter_english_only(results)
                except Exception as e:
                    print(f"Warning: Spotify enrichment failed: {e}")
                    st.info("💡 Album artwork unavailable - continuing with recommendations")

            # Use preview thumbs (yes/no) to push/pull candidates and lightly filter
            preview_tracks = st.session_state.get('preview_tracks', [])
            feedback = st.session_state.get('preview_feedback', {})
            results = rerank_with_preview_feedback(
                results,
                preview_tracks=preview_tracks,
                feedback_dict=feedback,
                alpha=0.35, beta=0.45, ban_threshold=0.92, genre_beta=0.20
            )

            display_results = results.head(15).copy()
            
            # Ensure we have at least 15 recommendations
            if len(display_results) < 15:
                st.warning(f"⚠️ Only found {len(display_results)} matching recommendations. Consider adjusting your language or genre preferences for more results.")
            
            st.session_state.final_recommendations = display_results

            # Generate and persist a playlist title for export
            st.session_state.playlist_title = generate_playlist_title(t, selected, st.session_state.creative_inspiration)

            # Attempt automatic unique cover artwork with Automatic1111
            if len(display_results) > 0:
                with st.spinner("🎨 Generating your unique playlist artwork (15–20 seconds)..."):
                    try:
                        art_prompt, art_neg = generate_playlist_artwork_prompt(t, selected, st.session_state.creative_inspiration, display_results)
                        art_img, err = generate_artwork_with_automatic1111(art_prompt, art_neg, width=768, height=768)
                        st.session_state.playlist_artwork = art_img if art_img else None
                        if err: print(f"Artwork generation: {err}")
                    except Exception as e:
                        print(f"Artwork generation error: {e}")
                        st.session_state.playlist_artwork = None

    display_results = st.session_state.final_recommendations

    if len(display_results) > 0:
        # Header: playlist title + seed context
        title_to_show = st.session_state.playlist_title or f"{t['track_name']} Vibes"
        st.title(title_to_show)
        st.caption(f"Rooted in **{t['track_name']}** — {t['artists']}")
        st.markdown("---")

        # Two-column: artwork (if any) + editorial summary paragraph
        col_art, col_desc = st.columns([1,1.5])
        with col_art:
            if st.session_state.get('playlist_artwork'):
                st.image(st.session_state.playlist_artwork, width='stretch')
            else:
                st.info("No artwork generated. Start Automatic1111 with --api and reload.")
        with col_desc:
            summary = generate_recommendation_summary(t, selected, st.session_state.creative_inspiration, st.session_state.get('language_pref_value', 50), display_results)
            st.markdown(summary)

        # "Perfect Moments" — three friendly one-liners about when to play the mix
        st.markdown("---")
        st.markdown("## 🎧 Perfect Moments")
        contexts = generate_listening_contexts(t, selected, st.session_state.creative_inspiration, display_results)
        sp1, c1, c2, c3, sp2 = st.columns([1, 2, 2, 2, 1])

        def context_card(col, text):
            # Small HTML card with emoji + one-liner
            parts = text.split(" ", 1)
            emoji, body = (parts[0], parts[1]) if len(parts) == 2 else ("🎵", text)
            col.markdown(
                f"""
                <div style="text-align:center; padding: 1rem; border-radius: 12px;
                            background: rgba(255,255,255,0.04);">
                    <div style="font-size: 1.8rem; line-height:1;">{emoji}</div>
                    <div style="margin-top: .5rem;">{body}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        if len(contexts) >= 3:
            context_card(c1, contexts[0]); context_card(c2, contexts[1]); context_card(c3, contexts[2])

        # Grid of 15 recommendations with album art, genre, Spotify link, and preview
        st.markdown("---")
        st.markdown(f"### 🎵 Your {len(display_results)} Recommendations")
        st.caption("Click any track to preview or open in Spotify")

        grid_cols = st.columns(3)
        for i, (_, row) in enumerate(display_results.iterrows()):
            with grid_cols[i % 3]:
                if pd.notna(row.get('album_art')): st.image(row['album_art'], width='stretch')
                st.markdown(f"**{row['track_name']}**")
                st.caption(f"{row['artists']}")
                if pd.notna(row.get('track_genre')): st.write(f"🎸 {row['track_genre'].title()}")
                if pd.notna(row.get('spotify_url')): st.markdown(f"[🎧 Open in Spotify]({row['spotify_url']})")
                prev = get_deezer_preview(row['track_name'], row['artists'])
                if prev: st.audio(prev)
                st.divider()

        # Export button: creates a private Spotify playlist and uploads cover art
        if spotify_enricher.enabled:
            if st.button("🎵 Export to Spotify Playlist", type="primary", width='stretch'):
                with st.spinner("Creating your Spotify playlist..."):
                    title = st.session_state.playlist_title or generate_playlist_title(t, selected, st.session_state.creative_inspiration)
                    creative_text = ""
                    if st.session_state.creative_inspiration and st.session_state.creative_inspiration.get('medium') != 'Skip':
                        medium = st.session_state.creative_inspiration['medium']
                        ctitle = st.session_state.creative_inspiration.get('title', 'Unknown')
                        creative_text = f" Inspired by {medium.lower()}: {ctitle}."
                    desc = f"Generated from '{t['track_name']}' by {t['artists']}.{creative_text} Created with MusicRUT."

                    # Start with positively rated previews, then add the 15 recs (deduped)
                    track_ids = list(dict.fromkeys(st.session_state.get('liked_preview_ids', [])))
                    for _, row in display_results.iterrows():
                        sid = row.get('spotify_id')
                        if pd.notna(sid): track_ids.append(sid)
                    track_ids = list(dict.fromkeys(track_ids))

                    cover = st.session_state.get('playlist_artwork')
                    url, err = create_spotify_playlist(spotify_enricher, track_ids, title, desc, cover_image=cover)
                    if url:
                        st.success(f"✅ Playlist created: **{title}**")
                        st.markdown(f"[🎧 Open in Spotify]({url})")
                    else:
                        st.error(f"❌ Failed to create playlist: {err}")
        else:
            st.info("💡 Configure Spotify API to export playlists")

        # Reset flow to discover more music
        if st.button("← Discover More Music", type="secondary", width='content'):
            st.session_state.step = 'seed_selection'
            st.session_state.selected_chips = []
            st.session_state.preview_feedback = []
            st.session_state.creative_inspiration = None
            st.session_state.liked_preview_ids = []
            for k in ['chips','preview_tracks','final_recommendations','selected_medium','playlist_artwork','artwork_prompt','artwork_negative','playlist_title']:
                if k in st.session_state: del st.session_state[k]
            st.rerun()
    else:
        st.warning("No similar tracks found.")

    # Bottom-of-app logo (nice bookend). Shown regardless of results.
    if LOGO_IMG:
        st.markdown("---")
        show_logo_centered(LOGO_IMG, width=220)


    # References: 
    # Anthropic (2024) Claude 3.5 Sonnet. [Large language model]. Available at: https://claude.ai (Accessed: 18 November 2024).
