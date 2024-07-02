'use client'

import { useRouter } from 'next/navigation'
import { useState, useEffect } from 'react'

import { DataTable } from '@/components/data-table'
import Metadata from '@/components/metadata'
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb'
import { Skeleton } from '@/components/ui/skeleton'

import { getPaperList, handleFetchArrayResult } from '@/lib/fetch'
import { PaperInfo } from '@/lib/types'

import { columns } from './columns'

const loadPaperLists = async () => {
  const result = await getPaperList()
  return handleFetchArrayResult<PaperInfo[]>(result, 'Failed to fetch papers')
}

export default function ListPage() {
  const router = useRouter()

  const [papers, setPapers] = useState<PaperInfo[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [width, setWidth] = useState(0)

  useEffect(() => {
    const handleResize = () => {
      setWidth(window.innerWidth)
    }

    if (typeof window !== 'undefined') {
      setWidth(window.innerWidth)
      window.addEventListener('resize', handleResize)
    }

    const fetchPapers = async () => {
      const fetchedPapers = await loadPaperLists()
      if (!fetchedPapers) {
        router.push('/')
      } else {
        setPapers(fetchedPapers)
        setIsLoading(false)
      }
    }
    void fetchPapers()

    handleResize()

    return () => {
      if (typeof window !== 'undefined') {
        window.removeEventListener('resize', handleResize)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const numPagesDisplayed = width > 600 ? 5 : width > 490 ? 3 : 0

  if (isLoading || !papers) {
    return (
      <>
        <Metadata
          description={"Automated summaries of top conference papers using OpenAI's language model."}
          ogDescription={"Automated summaries of top conference papers using OpenAI's language model."}
          ogImage={`http://cvpaper-summary-frontend-bucket.s3-website-ap-northeast-1.amazonaws.com/icon.png`}
          ogSiteName='LLM Survey'
          ogTitle={`Paper List`}
          ogType='article'
          ogUrl={`http://cvpaper-summary-frontend-bucket.s3-website-ap-northeast-1.amazonaws.com/list`}
          title={`Paper List | LLM Survey`}
          twitterCard='summary'
          twitterSite='@CVpaperChalleng'
        />
      <div className="flex w-screen flex-col items-center gap-12 py-8">
        <div className="flex w-full flex-row justify-start px-10">
          <Skeleton className="h-5 w-full rounded-sm" />
        </div>
        <div className="w-[80vw] min-w-[320px] pb-10">
          <Skeleton className="my-4 h-20 w-full max-w-sm rounded-sm" />
          <Skeleton className="h-[1393px] w-full rounded-sm" />
        </div>
      </div>
      </>
    )
  }

  return (
    <>
      <Metadata
        description={"Automated summaries of top conference papers using OpenAI's language model."}
        ogDescription={"Automated summaries of top conference papers using OpenAI's language model."}
        ogImage={`http://cvpaper-summary-frontend-bucket.s3-website-ap-northeast-1.amazonaws.com/icon.png`}
        ogSiteName='LLM Survey'
        ogTitle={`Paper List`}
        ogType='article'
        ogUrl={`http://cvpaper-summary-frontend-bucket.s3-website-ap-northeast-1.amazonaws.com/list`}
        title={`Paper List | LLM Survey`}
        twitterCard='summary'
        twitterSite='@CVpaperChalleng'
      />
    <div className="flex w-screen flex-col items-center gap-12 py-8">
      <div className="flex w-full flex-row justify-start px-10">
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink onClick={() => router.push("/")}>Home</BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>
              <BreadcrumbPage>Paper List</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
      </div>
      <DataTable
        columns={columns}
        data={papers}
        numPagesDisplayed={numPagesDisplayed}
        width={width}
      />
    </div>
    </>
  )
}
