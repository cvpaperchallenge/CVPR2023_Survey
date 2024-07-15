import { toast } from 'sonner'

import { PaperInfo, PaperDetails, FetchResult } from '@/lib/types'

interface ErrorResponse {
  message: string
}

async function fetchFromAPI<T>(
  url: string,
  options?: RequestInit
): Promise<FetchResult<T>> {
  try {
    const response = await fetch(url, options)
    const data = (await response.json()) as T

    if (!response.ok) {
      const errorData = data as unknown as ErrorResponse
      return {
        error: `Failed to fetch data.\n${errorData.message}`,
      }
    }

    return { data }
  } catch (error) {
    if (error instanceof Error) {
      return {
        error: `An unexpected error occurred.\n${error.message}`,
      }
    } else {
      return {
        error: 'An unexpected error occurred.',
      }
    }
  }
}

export const handleFetchResult = <T,>(
  result: FetchResult<T>,
  errorMessage: string,
  successMessage?: string
): T | null => {
  if (result.error) {
    toast.error(errorMessage)
    return null
  }
  if (successMessage) {
    toast.success(successMessage)
  }
  return result.data || null
}

export const handleFetchArrayResult = <T,>(
  result: FetchResult<T>,
  errorMessage: string,
  successMessage?: string
): T | [] => {
  if (result.error) {
    toast.error(errorMessage)
    return []
  }
  if (successMessage) {
    toast.success(successMessage)
  }
  return result.data || []
}

async function GetCachedItem<T>(
  url: string,
  cacheKey: string,
  cacheTime: number
): Promise<FetchResult<T>> {
  const now = Math.floor(Date.now() / 1000)
  const cachedExpiryTime = Number(
    localStorage.getItem(`${cacheKey}_expiryTime`)
  )

  // If the cache has expired
  if (!cachedExpiryTime || now > cachedExpiryTime) {
    // Set the new expiry time
    const expiryTime = String(now + cacheTime)
    localStorage.setItem(`${cacheKey}_expiryTime`, expiryTime)

    // Fetch the data from the API
    const result = await fetchFromAPI<T>(url)

    // If the data is valid, cache it
    if (result.data) {
      localStorage.setItem(cacheKey, JSON.stringify(result.data))
    }
    return { data: result.data }
  }

  // If the cache has not expired
  const cachedData = localStorage.getItem(cacheKey)
  // Check if the cached data exists
  if (cachedData) {
    const data = JSON.parse(cachedData) as T
    return { data }
  }

  // If the cached data does not exist
  const result = await fetchFromAPI<T>(url)
  return { data: result.data }
}

export async function getPaperList(): Promise<FetchResult<PaperInfo[]>> {
  const url = `${process.env.NEXT_PUBLIC_BACKEND_API_URL}/paper`
  const result = await GetCachedItem<PaperInfo[]>(url, 'paper', 60 * 60 * 1)
  return result
  // return fetchFromAPI<PaperInfo[]>(url)
}

export async function getPaperDetails(
  id: string
): Promise<FetchResult<PaperDetails>> {
  const url = `${process.env.NEXT_PUBLIC_BACKEND_API_URL}/paper/${id}`
  return fetchFromAPI<PaperDetails>(url)
}

export async function sendFeedback(
  name: string,
  feedback: string
): Promise<FetchResult<null>> {
  const url = `${process.env.NEXT_PUBLIC_BACKEND_API_URL}/feedback`
  const options: RequestInit = {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ name, feedback }),
  }
  return fetchFromAPI<null>(url, options)
}
