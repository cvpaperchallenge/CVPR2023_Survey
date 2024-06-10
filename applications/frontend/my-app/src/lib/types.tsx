
export interface FetchResult<T> {
  error?: string
  data?: T
}

export interface PaperInfo {
  title: string
  abstract: string
  authors: string[]
  cvfLink: string
  pdfLink: string
}

export interface Summary {
  outline: string
  contribution: string
  method: string
  evaluation: string
  discussion: string
}

export interface PaperDetails {
  paperInfo: PaperInfo
  summary: Summary
}