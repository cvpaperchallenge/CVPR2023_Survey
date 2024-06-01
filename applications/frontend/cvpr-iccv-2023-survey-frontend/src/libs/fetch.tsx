import { PaperInfo, PaperDetails, FetchResult } from '@/libs/types';
import { toast } from 'sonner'

interface ErrorResponse {
  message: string;
}

async function fetchFromAPI<T>(url: string, options?: RequestInit): Promise<FetchResult<T>> {
  try {
    const response = await fetch(url, options);
    const data = await response.json() as T;

    if (!response.ok) {
      const errorData = data as unknown as ErrorResponse;
      return {
        error: `Failed to fetch data.\n${errorData.message}`,
      };
    }

    return { data };
  } catch (error) {
    if (error instanceof Error) {
      return {
        error: `An unexpected error occurred.\n${error.message}`,
      };
    } else {
      return {
        error: 'An unexpected error occurred.',
      };
    }
  }
}

export const handleFetchResult = <T,>(
  result: FetchResult<T>,
  errorMessage: string,
  successMessage?: string
  ): T | null => {
  if (result.error) {
    toast.error(errorMessage);
    return null;
  }
  if (successMessage){
    toast.success(successMessage);
  }
  return result.data || null;
};

export const handleFetchArrayResult = <T,>(
  result: FetchResult<T>,
  errorMessage: string,
  successMessage?: string
): T | [] => {
  if (result.error) {
    toast.error(errorMessage);
    return [];
  }
  if (successMessage) {
    toast.success(successMessage);
  }
  return result.data || [];
};

export async function getPaperList(conference: string): Promise<FetchResult<PaperInfo[]>> {
  const url = `${process.env.NEXT_PUBLIC_BACKEND_API_URL}/${conference}`;
  return fetchFromAPI<PaperInfo[]>(url);
}

export async function searchPapers(query: string, conference: string): Promise<FetchResult<PaperInfo[]>> {
  const url = `${process.env.NEXT_PUBLIC_BACKEND_API_URL}/${conference}`;
  const options: RequestInit = {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query }),
  };
  return fetchFromAPI<PaperInfo[]>(url, options);
}

export async function getPaperDetails(conference: string, id: string): Promise<FetchResult<PaperDetails>> {
  const url = `${process.env.NEXT_PUBLIC_BACKEND_API_URL}/${conference}/${id}`;
  return fetchFromAPI<PaperDetails>(url);
}

export async function sendFeedback(name: string, feedback: string): Promise<FetchResult<null>> {
  const url = `${process.env.NEXT_PUBLIC_BACKEND_API_URL}/feedback`;
  const options: RequestInit = {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ name, feedback }),
  };
  return fetchFromAPI<null>(url, options);
}