export interface Track {
  id: string;
  name: string;
  artist: string;
  preview_url: string | null;
  external_urls: {
    spotify: string;
  };
  image: string;
  reason: string;
}

export interface RecommendationResponse {
  success: boolean;
  input?: string;
  imageDescription?: string;
  recommendations: Track[];
  total: number;
}

export interface ApiError {
  error: string;
  message?: string;
}