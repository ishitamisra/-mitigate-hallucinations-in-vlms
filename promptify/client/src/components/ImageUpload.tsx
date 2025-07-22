import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Image as ImageIcon, X, Camera } from 'lucide-react';

interface ImageUploadProps {
  onUpload: (file: File) => void;
  loading: boolean;
}

const ImageUpload: React.FC<ImageUploadProps> = ({ onUpload, loading }) => {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setSelectedImage(file);
      
      // Create preview URL
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.bmp', '.webp']
    },
    maxSize: 10 * 1024 * 1024, // 10MB
    multiple: false,
    disabled: loading
  });

  const handleUpload = () => {
    if (selectedImage && !loading) {
      onUpload(selectedImage);
    }
  };

  const clearImage = () => {
    setSelectedImage(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
  };

  const hasFileRejections = fileRejections.length > 0;

  return (
    <div className="max-w-4xl mx-auto">
      <div className="card">
        {!selectedImage ? (
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all duration-200 ${
              isDragActive
                ? 'border-purple-500 bg-purple-50'
                : 'border-gray-300 hover:border-purple-400 hover:bg-gray-50'
            } ${loading ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            <input {...getInputProps()} />
            <div className="flex flex-col items-center space-y-4">
              <div className="bg-gradient-to-r from-purple-500 to-pink-500 p-4 rounded-full">
                <Upload className="w-8 h-8 text-white" />
              </div>
              <div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">
                  {isDragActive ? 'Drop your image here' : 'Upload an image'}
                </h3>
                <p className="text-gray-600 mb-4">
                  Drag and drop an image, or click to select one
                </p>
                <p className="text-sm text-gray-500">
                  Supports: JPEG, PNG, GIF, WebP (max 10MB)
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            <div className="relative">
              <img
                src={previewUrl!}
                alt="Selected"
                className="w-full h-64 object-cover rounded-lg"
              />
              <button
                onClick={clearImage}
                disabled={loading}
                className="absolute top-2 right-2 bg-red-500 text-white p-2 rounded-full hover:bg-red-600 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <ImageIcon className="w-5 h-5 text-gray-500" />
                <div>
                  <p className="font-medium text-gray-900">{selectedImage.name}</p>
                  <p className="text-sm text-gray-500">
                    {(selectedImage.size / (1024 * 1024)).toFixed(2)} MB
                  </p>
                </div>
              </div>
              
              <button
                onClick={handleUpload}
                disabled={loading}
                className={`flex items-center space-x-2 px-6 py-3 rounded-lg font-semibold transition-all duration-200 ${
                  loading
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    : 'bg-gradient-to-r from-purple-500 to-pink-500 text-white hover:shadow-lg transform hover:scale-105'
                }`}
              >
                <Camera className="w-5 h-5" />
                <span>{loading ? 'Analyzing...' : 'Analyze Image'}</span>
              </button>
            </div>
          </div>
        )}

        {hasFileRejections && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
            <h4 className="font-medium text-red-800 mb-2">File rejected:</h4>
            {fileRejections.map(({ file, errors }) => (
              <div key={file.name} className="text-sm text-red-700">
                <p className="font-medium">{file.name}</p>
                <ul className="list-disc list-inside ml-2">
                  {errors.map((error) => (
                    <li key={error.code}>{error.message}</li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Tips Section */}
      <div className="mt-8 bg-blue-50 border border-blue-200 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-blue-900 mb-3">💡 Tips for better results:</h3>
        <ul className="space-y-2 text-sm text-blue-800">
          <li>• Upload photos that capture a specific mood or atmosphere</li>
          <li>• Scenic photos, lifestyle shots, and artistic images work great</li>
          <li>• The AI will analyze colors, lighting, activities, and overall vibe</li>
          <li>• Clear, well-lit images tend to produce better recommendations</li>
        </ul>
      </div>
    </div>
  );
};

export default ImageUpload;