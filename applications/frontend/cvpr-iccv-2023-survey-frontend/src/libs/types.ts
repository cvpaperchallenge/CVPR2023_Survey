
export interface FetchResult<T> {
  error?: string;
  data?: T;
}

export interface Paper {
  title: string;
  author: string;
  abstract: string;
  cvf: string;
  pdf: string;
}