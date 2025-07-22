import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import multer from 'multer';
import OpenAI from 'openai';
import SpotifyWebApi from 'spotify-web-api-node';
import axios from 'axios';
import path from 'path';
import fs from 'fs';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Configure multer for file uploads
const upload = multer({ 
  dest: 'uploads/',
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB limit
});

// Initialize OpenAI
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Initialize Spotify API
const spotifyApi = new SpotifyWebApi({
  clientId: process.env.SPOTIFY_CLIENT_ID,
  clientSecret: process.env.SPOTIFY_CLIENT_SECRET,
});

// Spotify access token management
let spotifyAccessToken: string | null = null;
let tokenExpirationTime: number = 0;

const getSpotifyAccessToken = async () => {
  try {
    if (spotifyAccessToken && Date.now() < tokenExpirationTime) {
      return spotifyAccessToken;
    }

    const response = await axios.post('https://accounts.spotify.com/api/token', 
      'grant_type=client_credentials',
      {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
          'Authorization': `Basic ${Buffer.from(`${process.env.SPOTIFY_CLIENT_ID}:${process.env.SPOTIFY_CLIENT_SECRET}`).toString('base64')}`
        }
      }
    );

    spotifyAccessToken = response.data.access_token;
    tokenExpirationTime = Date.now() + (response.data.expires_in * 1000) - 60000; // Refresh 1 minute early
    spotifyApi.setAccessToken(spotifyAccessToken);
    
    return spotifyAccessToken;
  } catch (error) {
    console.error('Error getting Spotify access token:', error);
    throw new Error('Failed to authenticate with Spotify');
  }
};

// Helper function to analyze image with OpenAI Vision
const analyzeImage = async (imagePath: string) => {
  try {
    const imageBuffer = fs.readFileSync(imagePath);
    const base64Image = imageBuffer.toString('base64');
    
    const response = await openai.chat.completions.create({
      model: "gpt-4-vision-preview",
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text: "Analyze this image and describe the mood, atmosphere, activities, colors, and overall vibe. Focus on what kind of music would complement this scene or feeling. Be descriptive and emotional in your response."
            },
            {
              type: "image_url",
              image_url: {
                url: `data:image/jpeg;base64,${base64Image}`
              }
            }
          ]
        }
      ],
      max_tokens: 300
    });

    return response.choices[0].message.content;
  } catch (error) {
    console.error('Error analyzing image:', error);
    throw new Error('Failed to analyze image');
  }
};

// Helper function to get music recommendations from GPT
const getMusicRecommendations = async (description: string) => {
  try {
    const prompt = `Based on this description: "${description}"
    
    Please suggest 10-15 songs that would match this vibe perfectly. Consider the mood, energy level, genre preferences, and atmosphere described. 
    
    Format your response as a JSON array with objects containing:
    - "track": song title
    - "artist": artist name
    - "reason": brief explanation why this song fits the vibe
    
    Focus on popular and well-known songs that are likely to be available on Spotify.`;

    const response = await openai.chat.completions.create({
      model: "gpt-4",
      messages: [
        {
          role: "system",
          content: "You are a music expert who creates perfect playlists based on moods, activities, and vibes. Always respond with valid JSON."
        },
        {
          role: "user",
          content: prompt
        }
      ],
      max_tokens: 1000,
      temperature: 0.7
    });

    const content = response.choices[0].message.content;
    return JSON.parse(content || '[]');
  } catch (error) {
    console.error('Error getting music recommendations:', error);
    throw new Error('Failed to get music recommendations');
  }
};

// Helper function to search Spotify for tracks
const searchSpotifyTracks = async (recommendations: any[]) => {
  try {
    await getSpotifyAccessToken();
    
    const spotifyTracks = await Promise.all(
      recommendations.map(async (rec) => {
        try {
          const searchQuery = `track:"${rec.track}" artist:"${rec.artist}"`;
          const searchResult = await spotifyApi.searchTracks(searchQuery, { limit: 1 });
          
          if (searchResult.body.tracks?.items.length > 0) {
            const track = searchResult.body.tracks.items[0];
            return {
              id: track.id,
              name: track.name,
              artist: track.artists[0].name,
              preview_url: track.preview_url,
              external_urls: track.external_urls,
              image: track.album.images[0]?.url,
              reason: rec.reason
            };
          }
          return null;
        } catch (error) {
          console.error(`Error searching for ${rec.track} by ${rec.artist}:`, error);
          return null;
        }
      })
    );

    return spotifyTracks.filter(track => track !== null);
  } catch (error) {
    console.error('Error searching Spotify tracks:', error);
    throw new Error('Failed to search Spotify tracks');
  }
};

// Routes

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', message: 'Promptify server is running!' });
});

// Text-based recommendation endpoint
app.post('/api/recommend/text', async (req, res) => {
  try {
    const { text } = req.body;
    
    if (!text) {
      return res.status(400).json({ error: 'Text input is required' });
    }

    console.log('Processing text recommendation for:', text);

    // Get music recommendations from GPT
    const recommendations = await getMusicRecommendations(text);
    
    // Search Spotify for the recommended tracks
    const spotifyTracks = await searchSpotifyTracks(recommendations);

    res.json({
      success: true,
      input: text,
      recommendations: spotifyTracks,
      total: spotifyTracks.length
    });

  } catch (error) {
    console.error('Error in text recommendation:', error);
    res.status(500).json({ 
      error: 'Failed to generate recommendations',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Image-based recommendation endpoint
app.post('/api/recommend/image', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Image file is required' });
    }

    console.log('Processing image recommendation for:', req.file.filename);

    // Analyze image with OpenAI Vision
    const imageDescription = await analyzeImage(req.file.path);
    
    // Get music recommendations based on image analysis
    const recommendations = await getMusicRecommendations(imageDescription || '');
    
    // Search Spotify for the recommended tracks
    const spotifyTracks = await searchSpotifyTracks(recommendations);

    // Clean up uploaded file
    fs.unlinkSync(req.file.path);

    res.json({
      success: true,
      imageDescription,
      recommendations: spotifyTracks,
      total: spotifyTracks.length
    });

  } catch (error) {
    console.error('Error in image recommendation:', error);
    
    // Clean up uploaded file in case of error
    if (req.file) {
      try {
        fs.unlinkSync(req.file.path);
      } catch (cleanupError) {
        console.error('Error cleaning up file:', cleanupError);
      }
    }

    res.status(500).json({ 
      error: 'Failed to process image and generate recommendations',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Create playlist endpoint (future enhancement)
app.post('/api/playlist/create', async (req, res) => {
  try {
    // This would require user authentication with Spotify
    // For now, return the tracks that can be manually added to a playlist
    const { tracks, playlistName } = req.body;
    
    res.json({
      success: true,
      message: 'Playlist data prepared',
      playlistName,
      tracks,
      note: 'Manual playlist creation required - copy track links to Spotify'
    });
  } catch (error) {
    console.error('Error creating playlist:', error);
    res.status(500).json({ error: 'Failed to create playlist' });
  }
});

// Error handling middleware
app.use((error: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
  console.error('Unhandled error:', error);
  res.status(500).json({ 
    error: 'Internal server error',
    message: error.message 
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🎵 Promptify server running on port ${PORT}`);
  console.log(`🔗 API endpoints available at http://localhost:${PORT}/api`);
  
  // Initialize Spotify token on startup
  getSpotifyAccessToken().catch(console.error);
});

export default app;