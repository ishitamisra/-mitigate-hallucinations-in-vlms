# 🎵 Promptify - AI-Powered Music Discovery

Promptify is a web application that generates tailored music recommendations based on user input. Simply upload an image or write a sentence about your vibe, what you're doing, or anything you want; the app then uses OpenAI's GPT models to create a customized music experience using Spotify's vast music library.

![Promptify Banner](https://via.placeholder.com/800x400/6366f1/ffffff?text=Promptify+-+AI+Music+Discovery)

## ✨ Features

- **🖼️ Image-Based Recommendations**: Upload any image and get music that matches the mood, colors, and atmosphere
- **📝 Text-Based Recommendations**: Describe your vibe, activity, or mood in natural language
- **🎧 AI-Powered Analysis**: Uses OpenAI GPT-4 Vision and text models for intelligent music curation
- **🎵 Spotify Integration**: Leverages Spotify's vast music library for recommendations
- **▶️ Preview Tracks**: Listen to 30-second previews directly in the app
- **🔗 Direct Spotify Links**: One-click access to full tracks on Spotify
- **📱 Responsive Design**: Beautiful, modern UI that works on all devices
- **⚡ Real-time Processing**: Fast AI analysis and music matching

## 🚀 Tech Stack

### Frontend
- **React 18** with TypeScript
- **Tailwind CSS** for styling
- **Lucide React** for icons
- **React Dropzone** for file uploads
- **Axios** for API communication

### Backend
- **Node.js** with Express
- **TypeScript** for type safety
- **OpenAI API** (GPT-4 Vision & GPT-4)
- **Spotify Web API** for music data
- **Multer** for file upload handling

## 🛠️ Installation & Setup

### Prerequisites
- Node.js (v18 or higher)
- npm or yarn
- OpenAI API key
- Spotify Developer account

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd promptify
```

### 2. Backend Setup
```bash
cd server
npm install

# Copy environment template
cp .env.example .env
```

### 3. Configure Environment Variables
Edit `server/.env` with your API credentials:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Spotify Configuration
SPOTIFY_CLIENT_ID=your_spotify_client_id_here
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret_here

# Server Configuration
PORT=5000
NODE_ENV=development
```

### 4. Frontend Setup
```bash
cd ../client
npm install
```

### 5. Getting API Keys

#### OpenAI API Key
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key to your `.env` file

#### Spotify API Credentials
1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/)
2. Log in with your Spotify account
3. Click "Create an App"
4. Fill in app details (name, description)
5. Copy Client ID and Client Secret to your `.env` file

### 6. Run the Application

#### Start the Backend Server
```bash
cd server
npm run dev
```
The server will start on `http://localhost:5000`

#### Start the Frontend (in a new terminal)
```bash
cd client
npm start
```
The client will start on `http://localhost:3000`

## 📖 Usage

### Text-Based Recommendations
1. Select the "Describe Your Vibe" tab
2. Type in your mood, activity, or music preference
3. Click "Get Recommendations"
4. Browse your personalized playlist

**Example prompts:**
- "I'm feeling nostalgic and want to reminisce about the good old days"
- "I need energetic workout music to pump me up"
- "Cozy rainy day vibes for reading a book"
- "Late night coding session with focus music"

### Image-Based Recommendations
1. Select the "Upload an Image" tab
2. Drag and drop an image or click to select
3. Click "Analyze Image"
4. Get music recommendations based on the image's mood and atmosphere

**Best image types:**
- Scenic photos (sunsets, landscapes, cityscapes)
- Lifestyle shots (activities, gatherings, workouts)
- Artistic images with strong mood/atmosphere
- Clear, well-lit photos work best

### Listening to Recommendations
- Click the play button to hear 30-second previews
- Click the Spotify button to open the full track
- Use "Copy All Links" to save the playlist

## 🏗️ Project Structure

```
promptify/
├── client/                 # React frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── services/       # API services
│   │   ├── types.ts        # TypeScript definitions
│   │   └── App.tsx         # Main app component
│   ├── public/
│   └── package.json
├── server/                 # Express backend
│   ├── index.ts           # Main server file
│   ├── uploads/           # Temporary image storage
│   ├── .env.example       # Environment template
│   └── package.json
└── README.md
```

## 🎯 API Endpoints

### `POST /api/recommend/text`
Generate recommendations based on text input
```json
{
  "text": "I'm feeling chill and want to relax"
}
```

### `POST /api/recommend/image`
Generate recommendations based on uploaded image
- Accepts: multipart/form-data with image file
- Supported formats: JPEG, PNG, GIF, WebP (max 10MB)

### `GET /api/health`
Health check endpoint

## 🔧 Development

### Available Scripts

#### Backend
- `npm run dev` - Start development server with hot reload
- `npm run build` - Build TypeScript to JavaScript
- `npm start` - Start production server

#### Frontend
- `npm start` - Start development server
- `npm run build` - Build for production
- `npm test` - Run tests

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes |
| `SPOTIFY_CLIENT_ID` | Spotify app client ID | Yes |
| `SPOTIFY_CLIENT_SECRET` | Spotify app client secret | Yes |
| `PORT` | Server port (default: 5000) | No |
| `NODE_ENV` | Environment (development/production) | No |

## 🚀 Deployment

### Backend Deployment
1. Build the TypeScript code: `npm run build`
2. Set environment variables on your hosting platform
3. Deploy the `dist/` folder and `package.json`
4. Run `npm install --production` and `npm start`

### Frontend Deployment
1. Build the React app: `npm run build`
2. Deploy the `build/` folder to your static hosting service
3. Set `REACT_APP_API_URL` to your backend URL

### Recommended Platforms
- **Backend**: Railway, Render, Heroku, DigitalOcean
- **Frontend**: Vercel, Netlify, AWS S3 + CloudFront

## 🐛 Troubleshooting

### Common Issues

**"Failed to authenticate with Spotify"**
- Check your Spotify Client ID and Secret
- Ensure your Spotify app is not in development mode restrictions

**"Failed to analyze image"**
- Ensure your OpenAI API key is valid and has GPT-4 Vision access
- Check image file size (max 10MB) and format

**"Network error occurred"**
- Verify the backend server is running on the correct port
- Check CORS settings if running on different domains

**Preview audio not playing**
- Some Spotify tracks don't have preview URLs
- Browser may block autoplay - user interaction required

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI](https://openai.com/) for GPT-4 and Vision API
- [Spotify](https://developer.spotify.com/) for the Web API
- [Lucide](https://lucide.dev/) for beautiful icons
- [Tailwind CSS](https://tailwindcss.com/) for styling

## 📧 Support

If you have any questions or run into issues, please:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed information

---

**Built with ❤️ using OpenAI GPT-4, Spotify Web API, React, and Tailwind CSS**