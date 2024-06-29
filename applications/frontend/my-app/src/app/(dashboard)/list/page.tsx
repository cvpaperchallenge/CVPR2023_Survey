'use client'

import { useRouter } from 'next/navigation'
import { useState, useEffect } from 'react'

import { DataTable } from '@/components/data-table'
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
  const [width, setWidth] = useState(window.innerWidth)

  useEffect(() => {
    const handleResize = () => {
      setWidth(window.innerWidth)
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
    fetchPapers()

    window.addEventListener('resize', handleResize)

    handleResize()

    return () => {
      window.removeEventListener('resize', handleResize)
    }
  }, [])

  const numPagesDisplayed = width > 600 ? 5 : width > 490 ? 3 : 0

  if (isLoading || !papers) {
    return (
      <div className="flex w-screen flex-col items-center gap-12 py-8">
        <div className="flex w-full flex-row justify-start px-10">
          <Skeleton className="h-5 w-full rounded-sm" />
        </div>
        <div className="w-[80vw] min-w-[320px] pb-10">
          <Skeleton className="my-4 h-20 w-full max-w-sm rounded-sm" />
          <Skeleton className="h-[1393px] w-full rounded-sm" />
        </div>
      </div>
    )
  }

  return (
    <div className="flex w-screen flex-col items-center gap-12 py-8">
      <div className="flex w-full flex-row justify-start px-10">
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink href="/">Home</BreadcrumbLink>
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
  )
}
