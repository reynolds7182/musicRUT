# MusicRUT
MusicRUT is a music recommendation system built to help people break out of algorithmic ruts and discover songs that actually feel exciting again. It combines a seed track you already love with preference choices and creative inspiration from films, books, artwork, photography, and theatre to shape a more personal playlist. This repository includes the full development process, weekly logs, and the final system used for my thesis project.

I designed and built MusicRUT as a speculative proposal for a feature that Spotify could integrate into its platform.

<img width="616" alt="Screenshot 2025-11-24 at 9 59 14 PM" src="https://git.arts.ac.uk/user-attachments/assets/ee258502-c477-4374-a862-7116a1599d08" />



## Quick Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
mkdir data
# Place your spotify.csv in data/ folder
```

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys (see below)
```

### 4. Install Ollama
```bash
# Install from https://ollama.ai

# Required: Text generation for chips and summaries
ollama pull llama3

# Optional but recommended: Image analysis for Art/Photography inspiration
ollama pull llava

# Start the service
ollama serve
```


### 5. Run
```bash
streamlit run musicRUT.py
```

Open http://localhost:8501 in your browser.

## API Keys (Optional but Recommended)

### Spotify (for playlist export)
1. Go to https://developer.spotify.com/dashboard
2. Create an app
3. Copy Client ID and Secret to `.env`
4. Add redirect URI: `http://localhost:8888/callback`

### Genius (optional - for lyrics)
1. Visit https://genius.com/api-clients
2. Create API client and generate token
3. Add to `.env`

### TMDB (optional - for film metadata)
1. Sign up at https://www.themoviedb.org/
2. Get API key from Settings → API
3. Add to `.env`

## Dataset Format

Your `data/spotify.csv` needs these columns:
- **track_name**, **artists**, **album_name**, **track_genre**
- **tempo**, **energy**, **valence**, **danceability**
- **acousticness**, **instrumentalness**, **speechiness**, **liveness**, **loudness**
- **popularity**

Minimum 10,000 tracks recommended.

## Features

- Hybrid recommendation engine (cosine + euclidean similarity)
- Language filtering (English → International)
- Cross-modal inspiration from film, art, and literature
- 30-second preview system with feedback
- AI-generated playlist artwork (requires Automatic1111)
- Spotify playlist export

## Troubleshooting

**"Cannot find data/spotify.csv"**
- Make sure your dataset is in the data/ folder

**"Cannot connect to Ollama"**
- Run `ollama serve` in a separate terminal

**"No Spotify credentials"**
- Edit `.env` with your API credentials

**Preview/artwork not working**
- Previews: Automatic via Deezer (no setup needed)
- Artwork: Requires Automatic1111 with `--api` flag

## Optional: Artwork Generation

For AI-generated playlist covers:
```bash
# Install from https://github.com/AUTOMATIC1111/stable-diffusion-webui
cd stable-diffusion-webui
./webui.sh --api
```

## System Requirements

- Python 3.8+
- 4GB RAM minimum
- ~500MB disk space

---

**Note:** First run takes 5-10 minutes for language detection (cached after).

## AI Assistance Disclaimer

During development, I used the Claude language model as a coding assistant. I relied on it mainly for debugging, refactoring, and helping structure functions inside the two core files. It was used as a support tool to speed up problem solving. All recommendation logic, architectural choices, and system design decisions were created manually and guided by my own research goals.

*Formatted using ChatGPT*
