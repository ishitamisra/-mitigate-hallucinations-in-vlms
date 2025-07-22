# 🚀 Quick Start Guide - Promptify

Get Promptify up and running in just a few minutes!

## ⚡ Quick Setup (Automated)

```bash
# Run the automated setup script
chmod +x setup.sh
./setup.sh
```

## 🔑 Get Your API Keys

### 1. OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Sign up/login and go to API Keys
3. Create a new API key
4. Copy the key

### 2. Spotify API Credentials
1. Visit [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/)
2. Login and create a new app
3. Copy Client ID and Client Secret

## 🔧 Configure Environment

Edit `server/.env`:
```env
OPENAI_API_KEY=your_openai_api_key_here
SPOTIFY_CLIENT_ID=your_spotify_client_id_here
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret_here
```

## 🎵 Start the App

```bash
# Start both frontend and backend
npm run dev

# OR start them separately:
# Terminal 1 - Backend
cd server && npm run dev

# Terminal 2 - Frontend  
cd client && npm start
```

## 🌐 Access the App

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000

## 🎯 Try It Out!

1. **Text Mode**: Try "I'm feeling nostalgic and want to reminisce"
2. **Image Mode**: Upload a sunset photo or cozy indoor scene
3. **Listen**: Click play buttons for 30-second previews
4. **Open in Spotify**: Click Spotify buttons to open full tracks

## 🐛 Common Issues

**"Failed to authenticate with Spotify"**
- Double-check your Spotify Client ID and Secret
- Make sure there are no extra spaces in your .env file

**"Failed to analyze image"**  
- Ensure your OpenAI API key is valid
- Check that you have GPT-4 Vision access

**Port already in use**
- Kill existing processes: `pkill -f "node"`
- Or use different ports in the .env file

## 📱 Features to Try

- **Smart Text Analysis**: Describe complex moods and activities
- **Image Mood Detection**: Upload photos with different vibes
- **Music Preview**: Listen before you decide
- **Spotify Integration**: Direct links to full tracks
- **AI Explanations**: See why each song was chosen

---

**Need help?** Check the full [README.md](README.md) for detailed documentation!