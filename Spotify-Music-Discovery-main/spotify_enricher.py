"""
Spotify Enricher - Fetches album art and metadata to enhance CSV-based recommendations
"""
import spotipy
from spotipy.oauth2 import SpotifyOAuth  # Changed from SpotifyClientCredentials
import os
from dotenv import load_dotenv
from functools import lru_cache
import time

load_dotenv()

class SpotifyEnricher:
    """
    Lightweight Spotify integration for fetching album art and metadata,
    plus creating playlists via OAuth.
    """
    
    def __init__(self):
        """Initialize Spotify client with OAuth (needed for playlist creation)"""
        try:
            # Use OAuth for user authentication (allows playlist creation)
            auth_manager = SpotifyOAuth(
                client_id=os.getenv("SPOTIPY_CLIENT_ID"),
                client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
                redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI", "http://localhost:8888/callback"),
                scope="playlist-modify-private user-library-read ugc-image-upload",  # Added ugc-image-upload
                cache_path=".spotify_cache",
                open_browser=True
)
            
            self.sp = spotipy.Spotify(auth_manager=auth_manager)
            
            # Test authentication by getting current user
            self.sp.current_user()
            
            self.enabled = True
            print("✅ Spotify OAuth authenticated successfully")
            
        except Exception as e:
            print(f"⚠️ Spotify enrichment disabled: {e}")
            self.enabled = False
            self.sp = None
    
    @lru_cache(maxsize=1000)
    def search_track_metadata(self, track_name, artist_name):
        """
        Search Spotify for a track and return metadata.
        Cached to avoid repeated API calls for the same track.
        
        Returns dict with:
        - album_art: URL to album cover image
        - spotify_url: Link to track on Spotify
        - preview_url: 30s preview (if available)
        - release_date: Track release date
        """
        if not self.enabled:
            return None
        
        try:
            # Search for track
            query = f"track:{track_name} artist:{artist_name}"
            results = self.sp.search(q=query, type='track', limit=1)
            
            if results['tracks']['items']:
                track = results['tracks']['items'][0]
                
                return {
                    'album_art': track['album']['images'][0]['url'] if track['album']['images'] else None,
                    'album_art_small': track['album']['images'][-1]['url'] if track['album']['images'] else None,
                    'spotify_url': track['external_urls']['spotify'],
                    'preview_url': track.get('preview_url'),
                    'release_date': track['album']['release_date'],
                    'spotify_id': track['id'],
                    'album_name': track['album']['name']
                }
            
            return None
            
        except Exception as e:
            print(f"Error fetching metadata for {track_name}: {e}")
            return None
    
    def enrich_track(self, track_data):
        """
        Enrich a single track dict with Spotify metadata.
        
        Args:
            track_data: Dict with 'track_name' and 'artists' keys
        
        Returns:
            Enhanced track_data dict with Spotify metadata added
        """
        if not self.enabled:
            return track_data
        
        metadata = self.search_track_metadata(
            track_data['track_name'], 
            track_data['artists']
        )
        
        if metadata:
            track_data.update(metadata)
        
        return track_data
    
    def enrich_dataframe(self, df, batch_size=50, delay=0.1):
        """
        Enrich a pandas DataFrame with Spotify metadata.
        
        Args:
            df: DataFrame with 'track_name' and 'artists' columns
            batch_size: Process in batches to avoid rate limits
            delay: Delay between batches (seconds)
        
        Returns:
            DataFrame with new columns: album_art, spotify_url, etc.
        """
        if not self.enabled:
            return df
        
        print(f"Enriching {len(df)} tracks with Spotify data...")
        
        # Add new columns
        df['album_art'] = None
        df['album_art_small'] = None
        df['spotify_url'] = None
        df['spotify_preview_url'] = None
        df['release_date'] = None
        df['spotify_id'] = None
        
        # Process in batches
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            
            for idx, row in batch.iterrows():
                metadata = self.search_track_metadata(
                    row['track_name'],
                    row['artists']
                )
                
                if metadata:
                    df.at[idx, 'album_art'] = metadata.get('album_art')
                    df.at[idx, 'album_art_small'] = metadata.get('album_art_small')
                    df.at[idx, 'spotify_url'] = metadata.get('spotify_url')
                    df.at[idx, 'spotify_preview_url'] = metadata.get('preview_url')
                    df.at[idx, 'release_date'] = metadata.get('release_date')
                    df.at[idx, 'spotify_id'] = metadata.get('spotify_id')
            
            # Progress update
            processed = min(i + batch_size, len(df))
            print(f"  Processed {processed}/{len(df)} tracks...")
            
            # Rate limit protection
            if i + batch_size < len(df):
                time.sleep(delay)
        
        enriched_count = df['album_art'].notna().sum()
        print(f"✅ Enriched {enriched_count}/{len(df)} tracks with Spotify data")
        
        return df


# Test
if __name__ == "__main__":
    import pandas as pd
    
    enricher = SpotifyEnricher()
    
    if enricher.enabled:
        # Test single track
        test_track = {
            'track_name': 'Vroom Vroom',
            'artists': 'Charli XCX'
        }
        
        enriched = enricher.enrich_track(test_track)
        print("\nEnriched track:")
        for key, value in enriched.items():
            print(f"  {key}: {value}")
        
        # Test dataframe
        test_df = pd.DataFrame([
            {'track_name': 'Vroom Vroom', 'artists': 'Charli XCX'},
            {'track_name': 'Blinding Lights', 'artists': 'The Weeknd'},
            {'track_name': 'A Change of Heart', 'artists': 'The 1975'}
        ])
        
        enriched_df = enricher.enrich_dataframe(test_df)
        print("\nEnriched DataFrame:")
        print(enriched_df[['track_name', 'artists', 'album_art', 'spotify_url']].to_string())
    else:
        print("❌ Spotify enricher not enabled")