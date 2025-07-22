import React from 'react';
import { Music, Brain, Sparkles } from 'lucide-react';

const LoadingSpinner: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center py-16">
      <div className="relative">
        {/* Rotating outer circle */}
        <div className="animate-spin rounded-full h-24 w-24 border-4 border-purple-200 border-t-purple-500"></div>
        
        {/* Inner music icon */}
        <div className="absolute inset-0 flex items-center justify-center">
          <Music className="w-8 h-8 text-purple-500 animate-pulse" />
        </div>
      </div>
      
      <div className="mt-8 text-center">
        <h3 className="text-xl font-semibold text-gray-900 mb-2">
          AI is crafting your perfect playlist...
        </h3>
        <div className="flex items-center justify-center space-x-6 text-sm text-gray-600">
          <div className="flex items-center space-x-2">
            <Brain className="w-4 h-4 text-blue-500" />
            <span>Analyzing your input</span>
          </div>
          <div className="flex items-center space-x-2">
            <Sparkles className="w-4 h-4 text-yellow-500" />
            <span>Finding perfect matches</span>
          </div>
          <div className="flex items-center space-x-2">
            <Music className="w-4 h-4 text-green-500" />
            <span>Curating playlist</span>
          </div>
        </div>
        
        <div className="mt-6 max-w-md mx-auto">
          <div className="bg-gray-200 rounded-full h-2 overflow-hidden">
            <div className="bg-gradient-to-r from-purple-500 to-pink-500 h-full rounded-full animate-pulse"></div>
          </div>
        </div>
        
        <p className="mt-4 text-sm text-gray-500">
          This may take 10-30 seconds depending on the complexity...
        </p>
      </div>
    </div>
  );
};

export default LoadingSpinner;