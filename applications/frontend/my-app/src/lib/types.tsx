
export interface FetchResult<T> {
  error?: string
  data?: T
}

export interface PaperInfo {
  title: string
  authors: string[]
  cvfLink: string
  pdfLink: string
  conference: string
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
  abstract: string
  summary: Summary
}