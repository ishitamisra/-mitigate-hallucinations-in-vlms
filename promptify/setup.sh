#!/bin/bash

# Promptify Setup Script
echo "🎵 Setting up Promptify - AI-Powered Music Discovery"
echo "=================================================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js (v18 or higher) first."
    echo "   Visit: https://nodejs.org/"
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js version $NODE_VERSION detected. Please upgrade to Node.js v18 or higher."
    exit 1
fi

echo "✅ Node.js $(node -v) detected"

# Install root dependencies
echo "📦 Installing root dependencies..."
npm install

# Install server dependencies
echo "🔧 Installing server dependencies..."
cd server
npm install
cd ..

# Install client dependencies
echo "⚛️  Installing client dependencies..."
cd client
npm install
cd ..

# Copy environment template
echo "📝 Setting up environment configuration..."
if [ ! -f "server/.env" ]; then
    cp server/.env.example server/.env
    echo "✅ Created server/.env from template"
else
    echo "ℹ️  server/.env already exists, skipping..."
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Edit server/.env with your API keys:"
echo "   - OpenAI API key (https://platform.openai.com/)"
echo "   - Spotify Client ID & Secret (https://developer.spotify.com/)"
echo ""
echo "2. Start the development servers:"
echo "   npm run dev"
echo ""
echo "3. Open http://localhost:3000 in your browser"
echo ""
echo "📖 For detailed instructions, see README.md"
echo "🐛 Having issues? Check the troubleshooting section in README.md"