import React, { useState } from 'react';
import { Music, Upload, Type, Sparkles } from 'lucide-react';
import ImageUpload from './components/ImageUpload';
import TextInput from './components/TextInput';
import RecommendationList from './components/RecommendationList';
import LoadingSpinner from './components/LoadingSpinner';
import { Track } from './types';
import { getRecommendationsFromText, getRecommendationsFromImage } from './services/api';

function App() {
  const [activeTab, setActiveTab] = useState<'text' | 'image'>('text');
  const [loading, setLoading] = useState(false);
  const [recommendations, setRecommendations] = useState<Track[]>([]);
  const [inputDescription, setInputDescription] = useState<string>('');
  const [error, setError] = useState<string>('');

  const handleTextRecommendation = async (text: string) => {
    setLoading(true);
    setError('');
    setRecommendations([]);
    
    try {
      const response = await getRecommendationsFromText(text);
      setRecommendations(response.recommendations);
      setInputDescription(text);
    } catch (err) {
      setError('Failed to get recommendations. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleImageRecommendation = async (file: File) => {
    setLoading(true);
    setError('');
    setRecommendations([]);
    
    try {
      const response = await getRecommendationsFromImage(file);
      setRecommendations(response.recommendations);
      setInputDescription(response.imageDescription || 'Image uploaded');
    } catch (err) {
      setError('Failed to process image and get recommendations. Please try again.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md border-b border-white/20 sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="bg-gradient-to-r from-purple-500 to-pink-500 p-2 rounded-xl">
                <Music className="w-8 h-8 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                  Promptify
                </h1>
                <p className="text-sm text-gray-600">AI-Powered Music Discovery</p>
              </div>
            </div>
            <div className="flex items-center space-x-2 text-sm text-gray-600">
              <Sparkles className="w-4 h-4" />
              <span>Powered by OpenAI & Spotify</span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4">
            Discover Music That Matches
            <span className="bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
              {' '}Your Vibe
            </span>
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Upload an image or describe your mood, and let AI create the perfect playlist for you using Spotify's vast music library.
          </p>
        </div>

        {/* Tab Selection */}
        <div className="flex justify-center mb-8">
          <div className="bg-white rounded-xl p-2 shadow-lg border border-gray-100">
            <div className="flex space-x-2">
              <button
                onClick={() => setActiveTab('text')}
                className={`flex items-center space-x-2 px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
                  activeTab === 'text'
                    ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-md'
                    : 'text-gray-600 hover:text-purple-600 hover:bg-purple-50'
                }`}
              >
                <Type className="w-5 h-5" />
                <span>Describe Your Vibe</span>
              </button>
              <button
                onClick={() => setActiveTab('image')}
                className={`flex items-center space-x-2 px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
                  activeTab === 'image'
                    ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-md'
                    : 'text-gray-600 hover:text-purple-600 hover:bg-purple-50'
                }`}
              >
                <Upload className="w-5 h-5" />
                <span>Upload an Image</span>
              </button>
            </div>
          </div>
        </div>

        {/* Input Section */}
        <div className="mb-8">
          {activeTab === 'text' ? (
            <TextInput onSubmit={handleTextRecommendation} loading={loading} />
          ) : (
            <ImageUpload onUpload={handleImageRecommendation} loading={loading} />
          )}
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-8 p-4 bg-red-50 border border-red-200 rounded-xl text-red-700">
            {error}
          </div>
        )}

        {/* Loading State */}
        {loading && <LoadingSpinner />}

        {/* Results Section */}
        {recommendations.length > 0 && !loading && (
          <div className="mb-8">
            <div className="text-center mb-6">
              <h3 className="text-2xl font-bold text-gray-900 mb-2">
                Your Personalized Playlist
              </h3>
              <p className="text-gray-600">
                Based on: <span className="italic">"{inputDescription}"</span>
              </p>
            </div>
            <RecommendationList tracks={recommendations} />
          </div>
        )}

        {/* Footer */}
        <footer className="text-center text-gray-500 text-sm mt-16 pb-8">
          <p>
            Built with ❤️ using OpenAI GPT-4, Spotify Web API, React, and Tailwind CSS
          </p>
        </footer>
      </main>
    </div>
  );
}

export default App;
