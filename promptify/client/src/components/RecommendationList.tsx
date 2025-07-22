import React, { useState } from 'react';
import { ExternalLink, Play, Pause, Volume2, Info } from 'lucide-react';
import { Track } from '../types';

interface RecommendationListProps {
  tracks: Track[];
}

const RecommendationList: React.FC<RecommendationListProps> = ({ tracks }) => {
  const [currentlyPlaying, setCurrentlyPlaying] = useState<string | null>(null);
  const [audioElements, setAudioElements] = useState<{ [key: string]: HTMLAudioElement }>({});

  const handlePlayPreview = (trackId: string, previewUrl: string | null) => {
    if (!previewUrl) return;

    // Stop any currently playing audio
    if (currentlyPlaying && audioElements[currentlyPlaying]) {
      audioElements[currentlyPlaying].pause();
      audioElements[currentlyPlaying].currentTime = 0;
    }

    if (currentlyPlaying === trackId) {
      // If clicking the same track, stop it
      setCurrentlyPlaying(null);
    } else {
      // Play new track
      let audio = audioElements[trackId];
      if (!audio) {
        audio = new Audio(previewUrl);
        audio.addEventListener('ended', () => setCurrentlyPlaying(null));
        setAudioElements(prev => ({ ...prev, [trackId]: audio }));
      }
      
      audio.play().catch(console.error);
      setCurrentlyPlaying(trackId);
    }
  };

  const openSpotify = (spotifyUrl: string) => {
    window.open(spotifyUrl, '_blank', 'noopener,noreferrer');
  };

  if (tracks.length === 0) {
    return null;
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="grid gap-4">
        {tracks.map((track, index) => (
          <div
            key={track.id}
            className="bg-white rounded-xl shadow-lg border border-gray-100 p-6 hover:shadow-xl transition-all duration-200 group"
          >
            <div className="flex items-start space-x-4">
              {/* Track Number */}
              <div className="flex-shrink-0 w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center text-white font-bold text-sm">
                {index + 1}
              </div>

              {/* Album Art */}
              <div className="flex-shrink-0">
                {track.image ? (
                  <img
                    src={track.image}
                    alt={`${track.name} album art`}
                    className="w-16 h-16 rounded-lg object-cover"
                  />
                ) : (
                  <div className="w-16 h-16 bg-gray-200 rounded-lg flex items-center justify-center">
                    <Volume2 className="w-6 h-6 text-gray-400" />
                  </div>
                )}
              </div>

              {/* Track Info */}
              <div className="flex-1 min-w-0">
                <h3 className="font-semibold text-gray-900 text-lg truncate">
                  {track.name}
                </h3>
                <p className="text-gray-600 truncate">{track.artist}</p>
                
                {/* AI Reasoning */}
                <div className="mt-3 p-3 bg-blue-50 rounded-lg border border-blue-100">
                  <div className="flex items-start space-x-2">
                    <Info className="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" />
                    <p className="text-sm text-blue-800">{track.reason}</p>
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex-shrink-0 flex items-center space-x-2">
                {/* Preview Play Button */}
                {track.preview_url && (
                  <button
                    onClick={() => handlePlayPreview(track.id, track.preview_url)}
                    className="p-3 bg-gray-100 hover:bg-gray-200 rounded-full transition-colors duration-200 group-hover:bg-purple-100 group-hover:text-purple-600"
                    title="Play 30s preview"
                  >
                    {currentlyPlaying === track.id ? (
                      <Pause className="w-5 h-5" />
                    ) : (
                      <Play className="w-5 h-5" />
                    )}
                  </button>
                )}

                {/* Open in Spotify */}
                <button
                  onClick={() => openSpotify(track.external_urls.spotify)}
                  className="p-3 bg-spotify-green hover:bg-green-600 text-white rounded-full transition-colors duration-200 flex items-center space-x-1"
                  title="Open in Spotify"
                >
                  <ExternalLink className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Footer Actions */}
      <div className="mt-8 text-center">
        <div className="bg-white rounded-xl shadow-lg border border-gray-100 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">
            Love this playlist?
          </h3>
          <p className="text-gray-600 mb-4">
            Click the Spotify links above to add these songs to your library or create a custom playlist.
          </p>
          <div className="flex justify-center space-x-4">
            <button
              onClick={() => {
                const trackLinks = tracks.map(track => `${track.name} by ${track.artist}: ${track.external_urls.spotify}`).join('\n');
                navigator.clipboard.writeText(trackLinks).then(() => {
                  alert('Track links copied to clipboard!');
                }).catch(() => {
                  alert('Failed to copy links. Please copy them manually.');
                });
              }}
              className="btn-secondary"
            >
              Copy All Links
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RecommendationList;