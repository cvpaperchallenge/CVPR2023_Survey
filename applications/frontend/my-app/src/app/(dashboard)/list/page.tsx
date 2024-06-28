'use client'

import { columns } from './columns'
import { DataTable } from '@/components/data-table'
import { useState, useEffect } from 'react'
import { getPaperList, handleFetchArrayResult } from '@/lib/fetch'
import { PaperInfo } from '@/lib/types'
import { useRouter } from 'next/navigation'
import { Skeleton } from '@/components/ui/skeleton'

import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb"

const loadPaperLists = async () => {
  const result = await getPaperList();
  return handleFetchArrayResult<PaperInfo[]>(result, 'Failed to fetch papers');
};

export default function ListPage() {
  const router = useRouter()

  const [papers, setPapers] = useState<PaperInfo[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [width, setWidth] = useState(window.innerWidth);

  useEffect(() => {
    const handleResize = () => {
      setWidth(window.innerWidth);
    };
    const fetchPapers = async () => {
      const fetchedPapers = await loadPaperLists()
      if (!fetchedPapers) {
        router.push("/")
      } else {
        setPapers(fetchedPapers)
        setIsLoading(false)
      }
    }
    fetchPapers()

    window.addEventListener('resize', handleResize);

    handleResize();

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [])

  const numPagesDisplayed = width > 600 ? 5 : width > 490 ? 3 : 0;

  if (isLoading || !papers) {
    return (
      <div className="flex flex-col items-center gap-12 w-screen py-8">
        <div className='flex flex-row justify-start w-full px-10'>
          <Skeleton className='h-5 w-full rounded-sm'/>
        </div>
        <div className="w-[80vw] pb-10 min-w-[320px]">
          <Skeleton className='h-20 my-4 w-full max-w-sm rounded-sm'/>
          <Skeleton className='h-[1393px] w-full rounded-sm'/>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col items-center gap-12 w-screen py-8">
      <div className='flex flex-row justify-start w-full px-10'>
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
      <DataTable columns={columns} data={papers} width={width} numPagesDisplayed={numPagesDisplayed}/>
    </div>
  )
}
