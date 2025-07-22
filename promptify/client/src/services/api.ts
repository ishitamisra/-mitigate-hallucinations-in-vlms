import axios from 'axios';
import { RecommendationResponse } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 seconds timeout for AI processing
});

export const getRecommendationsFromText = async (text: string): Promise<RecommendationResponse> => {
  try {
    const response = await api.post('/recommend/text', { text });
    return response.data;
  } catch (error) {
    console.error('Error getting text recommendations:', error);
    if (axios.isAxiosError(error)) {
      throw new Error(error.response?.data?.message || 'Failed to get recommendations');
    }
    throw new Error('Network error occurred');
  }
};

export const getRecommendationsFromImage = async (file: File): Promise<RecommendationResponse> => {
  try {
    const formData = new FormData();
    formData.append('image', file);

    const response = await api.post('/recommend/image', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  } catch (error) {
    console.error('Error getting image recommendations:', error);
    if (axios.isAxiosError(error)) {
      throw new Error(error.response?.data?.message || 'Failed to process image');
    }
    throw new Error('Network error occurred');
  }
};

export const checkServerHealth = async (): Promise<boolean> => {
  try {
    const response = await api.get('/health');
    return response.data.status === 'OK';
  } catch (error) {
    console.error('Server health check failed:', error);
    return false;
  }
};