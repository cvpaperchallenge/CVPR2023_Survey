import { Paper, PaperDetails, FetchResult } from '@/libs/types';

async function fetchFromAPI<T>(url: string, options?: RequestInit): Promise<FetchResult<T>> {
  try {
    const response = await fetch(url, options);
    const data = await response.json() as T;

    if (!response.ok) {
      return {
        error: data?.message || 'Failed to fetch data',
      };
    }

    return { data };
  } catch (error) {
    return {
      error: error.message || 'An unexpected error occurred',
    };
  }
}

export async function getPaperLists(conference: string): Promise<FetchResult<Paper[]>> {
  const url = `${process.env.NEXT_PUBLIC_BACKEND_API_URL}/${conference}`;
  return fetchFromAPI<Paper[]>(url);
}

export async function searchPapers(query: string, conference: string): Promise<FetchResult<Paper[]>> {
  const url = `${process.env.NEXT_PUBLIC_BACKEND_API_URL}/${conference}`;
  const options: RequestInit = {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query }),
  };
  return fetchFromAPI<Paper[]>(url, options);
}

export async function getPaperDetails(conference: string, id: string): Promise<FetchResult<PaperDetails>> {
  const url = `${process.env.NEXT_PUBLIC_BACKEND_API_URL}/${conference}/${id}`;
  return fetchFromAPI<PaperDetails>(url);
}