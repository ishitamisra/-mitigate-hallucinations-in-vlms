import React, { useState } from 'react';
import { Send, Lightbulb } from 'lucide-react';

interface TextInputProps {
  onSubmit: (text: string) => void;
  loading: boolean;
}

const TextInput: React.FC<TextInputProps> = ({ onSubmit, loading }) => {
  const [text, setText] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (text.trim() && !loading) {
      onSubmit(text.trim());
    }
  };

  const examples = [
    "I'm feeling nostalgic and want to reminisce about the good old days",
    "I need energetic workout music to pump me up",
    "Cozy rainy day vibes for reading a book",
    "Late night coding session with focus music",
    "Road trip with friends - fun and upbeat",
    "Romantic dinner for two"
  ];

  const handleExampleClick = (example: string) => {
    setText(example);
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="card">
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label htmlFor="text-input" className="block text-lg font-semibold text-gray-900 mb-3">
              Describe your vibe, mood, or activity
            </label>
            <textarea
              id="text-input"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Tell me about your mood, what you're doing, or the vibe you're looking for..."
              className="input-field resize-none h-32"
              disabled={loading}
              maxLength={500}
            />
            <div className="flex justify-between items-center mt-2">
              <span className="text-sm text-gray-500">
                {text.length}/500 characters
              </span>
              <button
                type="submit"
                disabled={!text.trim() || loading}
                className={`flex items-center space-x-2 px-6 py-3 rounded-lg font-semibold transition-all duration-200 ${
                  !text.trim() || loading
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    : 'bg-gradient-to-r from-purple-500 to-pink-500 text-white hover:shadow-lg transform hover:scale-105'
                }`}
              >
                <Send className="w-5 h-5" />
                <span>{loading ? 'Generating...' : 'Get Recommendations'}</span>
              </button>
            </div>
          </div>
        </form>
      </div>

      {/* Example Prompts */}
      <div className="mt-8">
        <div className="flex items-center space-x-2 mb-4">
          <Lightbulb className="w-5 h-5 text-yellow-500" />
          <h3 className="text-lg font-semibold text-gray-900">Need inspiration? Try these examples:</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {examples.map((example, index) => (
            <button
              key={index}
              onClick={() => handleExampleClick(example)}
              disabled={loading}
              className="text-left p-4 bg-white border border-gray-200 rounded-lg hover:border-purple-300 hover:bg-purple-50 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <p className="text-gray-700 text-sm">{example}</p>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default TextInput;