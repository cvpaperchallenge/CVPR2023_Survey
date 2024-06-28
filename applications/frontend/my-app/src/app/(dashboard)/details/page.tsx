'use client'

import { useState, useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'

import { toast } from "sonner"
import { Skeleton } from '@/components/ui/skeleton'
import { Separator } from "@/components/ui/separator"

import { PaperDetails, PaperInfo } from '@/lib/types'
import { getPaperDetails, handleFetchResult } from '@/lib/fetch'
import { Button } from '@/components/ui/button'
import { RxFile, RxGlobe, RxCaretLeft, RxCaretRight } from 'react-icons/rx'

import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb"
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
} from "@/components/ui/pagination"

const loadPaperDetails = async (id: string) => {
  const result = await getPaperDetails(id);
  return handleFetchResult<PaperDetails>(result, 'Failed to fetch paper details');
}

export default function DetailedPage() {

  const router = useRouter()
  const searchParams = useSearchParams()

  const paperId = searchParams.get('id')

  const [paperDetails, setPaperDetails] = useState<PaperDetails | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  const cachedData = localStorage.getItem("paper")
  const data: PaperInfo[] = JSON.parse(cachedData as string)

  useEffect(() => {
    if (!paperId) {
      toast.error('Invalid URL');
      router.push('/list')
    }
    else {
      const fetchPaperDetails = async () => {
        const fetchedPaperDetails = await loadPaperDetails(paperId)
        if (!fetchedPaperDetails) {
          router.push('/list')
        } else {
          setPaperDetails(fetchedPaperDetails)
          setIsLoading(false)
        }
      }
      fetchPaperDetails()
    }
  }, [paperId])

  if (isLoading || !paperDetails) {
    return (
      <div className="flex flex-col items-center gap-12 w-screen">
        <div className='flex flex-row justify-start w-full px-10'>
          <Skeleton className='h-5 w-full rounded-sm'/>
        </div>
        <div className="flex flex-col items-start gap-7 w-[75vw] max-w-[800px] min-w-[350px]">
          <Skeleton className='h-12 w-full rounded-sm'/>
          <div className='flex flex-col items-center gap-10 w-full'>
            <div className='flex flex-col items-start gap-4 w-full'>
              <Skeleton className='h-8 w-1/4 rounded-sm'/>
              <Skeleton className='h-96 w-full rounded-sm'/>
            </div>
            <div className='flex flex-col items-start gap-4 w-full'>
              <Skeleton className='h-8 w-1/4 rounded-sm'/>
              <Skeleton className='h-44 w-full rounded-sm'/>
            </div>
            <div className='flex flex-col items-start gap-4 w-full'>
              <Skeleton className='h-8 w-1/4 rounded-sm'/>
              <Skeleton className='h-44 w-full rounded-sm'/>
            </div>
            <div className='flex flex-col items-start gap-4 w-full'>
              <Skeleton className='h-8 w-1/4 rounded-sm'/>
              <Skeleton className='h-44 w-full rounded-sm'/>
            </div>
            <div className='flex flex-col items-start gap-4 w-full'>
              <Skeleton className='h-8 w-1/4 rounded-sm'/>
              <Skeleton className='h-44 w-full rounded-sm'/>
            </div>
            <div className='flex flex-col items-start gap-4 w-full'>
              <Skeleton className='h-8 w-1/4 rounded-sm'/>
              <Skeleton className='h-44 w-full rounded-sm'/>
            </div>
          </div>
        </div>
        <div className="flex flex-row gap-5 pb-0 min-[601px]:pb-10 min-[601px]:max-[774px]:gap-5 min-[775px]:gap-10 px-5 max-w-[1000px]">
          <Skeleton className='h-14 w-[40vw] rounded-sm'/>
          <Skeleton className='h-14 w-[40vw] rounded-sm'/>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col items-center gap-12 w-screen">
      <div className='flex flex-row justify-start w-full px-10'>
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbLink href="/">Home</BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>
              <BreadcrumbLink href="/list">Paper List</BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>
              <BreadcrumbPage>{paperId}. {paperDetails.paperInfo.title}</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
      </div>
      <div className="flex flex-col items-start gap-7 w-[75vw] max-w-[800px] min-w-[350px]">
        <div className='flex flex-row justify-center w-full'>
          <div className='text-xl font-extrabold text-foreground'>
            {paperDetails.paperInfo.title || ''}
          </div>
        </div>
        <div className='flex flex-col items-center gap-10 w-full'>
          <div className='flex flex-col items-start gap-0 w-full'>
            <div>
              <span className='text-xl font-semibold text-[var(--black-a10)] dark:text-[var(--white-a10)] pr-2'>基本情報</span>
              <span className='text-sm font-normal text-muted-foreground'>/ Basic Information</span>
            </div>
            <Separator className='my-2 w-60'/>
            <div className="flex flex-col gap-1 items-start min-[775px]:grid min-[775px]:grid-cols-6 min-[775px]:gap-4 min-[775px]:items-center w-full p-4 bg-[var(--teal-3)] rounded-sm">
              <div className='flex flex-col items-start min-w-20 gap-2 text-[14px] text-popover-foreground font-medium'>ID</div>
              <div className='mb-3 min-[775px]:mb-0 pl-2 min-[775px]:pl-0 col-span-5 gap-2 text-xs text-popover-foreground'>{paperId}</div>
              <div className='flex flex-col items-start min-w-20 gap-2 text-[14px] text-popover-foreground font-medium'>Authors</div>
              <div className='mb-3 min-[775px]:mb-0 pl-2 min-[775px]:pl-0 col-span-5 gap-2 text-xs text-popover-foreground'>{paperDetails.paperInfo.authors.join(", ")}</div>
              <div className='flex flex-col items-start min-w-20 gap-2 text-[14px] text-popover-foreground font-medium'>Abstract</div>
              <div className='mb-3 min-[775px]:mb-0 pl-2 min-[775px]:pl-0 col-span-5 gap-2 text-xs text-justify text-popover-foreground'>{paperDetails.abstract}</div>
              <div className='flex flex-col items-start min-w-20 gap-2 text-[14px] text-popover-foreground font-medium'>Link</div>
              <div className='mb-3 min-[775px]:mb-0 col-span-5 flex flex-row items-start gap-2'>
                <a href={paperDetails.paperInfo.cvfLink} target="_blank" rel="noreferrer">
                  <Button
                    variant="outline"
                    className="
                    gap-1
                    text-[var(--jade-11)]
                    bg-[var(--jade-4)]
                    hover:bg-[var(--jade-5)]
                    hover:text-[var(--jade-12)]
                    border-[var(--jade-6)]
                    px-2
                    h-6
                    text-xs
                  ">
                    <RxGlobe/> CVF
                  </Button>
                </a>
                <a href={paperDetails.paperInfo.pdfLink} target="_blank" rel="noreferrer">
                  <Button
                    variant="outline"
                    className="
                      gap-1
                      text-[var(--red-11)]
                      bg-[var(--red-3)]
                      hover:bg-[var(--red-4)]
                      hover:text-[var(--red-12)]
                      border-[var(--red-6)]
                      px-2
                      h-6
                      text-xs
                  ">
                      <RxFile/> PDF
                  </Button>
                </a>
              </div>
              <div className='flex flex-col items-start gap-2 text-[14px] text-popover-foreground font-medium'>Conference</div>
              <div className='pl-2 min-[775px]:pl-0 col-span-5 gap-2 text-xs text-popover-foreground'>{paperDetails.paperInfo.conference}</div>
            </div>
          </div>
          <div className='flex flex-col items-start gap-0 w-full'>
            <div>
              <span className='text-xl font-semibold text-[var(--black-a10)] dark:text-[var(--white-a10)] pr-2'>どんなもの？</span>
              <span className='text-sm font-normal text-muted-foreground'>/ Outline</span>
            </div>
            <Separator className='my-2 w-44'/>
            <div className="w-full p-4 bg-[var(--teal-3)]  rounded-sm">
              <div className='text-justify text-sm text-popover-foreground'>{paperDetails.summary.outline}</div>
            </div>
          </div>
          <div className='flex flex-col items-start gap-0 w-full'>
            <div>
              <span className='text-xl font-semibold text-[var(--black-a10)] dark:text-[var(--white-a10)] pr-2'>先行研究と比べてどこがすごい？</span>
              <span className='text-sm font-normal text-muted-foreground'>/ Contribution</span>
            </div>
            <Separator className='my-2 w-[365px]'/>
            <div className="w-full p-4 bg-[var(--teal-3)] rounded-sm">
              <div className='text-justify text-sm text-popover-foreground'>{paperDetails.summary.contribution}</div>
            </div>
          </div>
          <div className='flex flex-col items-start gap-0 w-full'>
            <div>
              <span className='text-xl font-semibold text-[var(--black-a10)] dark:text-[var(--white-a10)] pr-2'>技術や手法のキモはどこ？</span>
              <span className='text-sm font-normal text-muted-foreground'>/ Methods</span>
            </div>
            <Separator className='my-2 w-[295px]'/>
            <div className="w-full p-4 bg-[var(--teal-3)] rounded-sm">
              <div className='text-justify text-sm text-popover-foreground'>{paperDetails.summary.method}</div>
            </div>
          </div>
          <div className='flex flex-col items-start gap-0 w-full'>
            <div>
              <span className='text-xl font-semibold text-[var(--black-a10)] dark:text-[var(--white-a10)] pr-2'>どうやって有効だと検証した？</span>
              <span className='text-sm font-normal text-muted-foreground'>/ Evaluation</span>
            </div>
            <Separator className='my-2 w-[330px]'/>
            <div className="w-full p-4 bg-[var(--teal-3)] rounded-sm">
              <div className='text-justify text-sm text-popover-foreground'>{paperDetails.summary.evaluation}</div>
            </div>
          </div>
          <div className='flex flex-col items-start gap-0 w-full'>
            <div>
              <span className='text-xl font-semibold text-[var(--black-a10)] dark:text-[var(--white-a10)] pr-2'>議論はある？</span>
              <span className='text-sm font-normal text-muted-foreground'>/ Discussion</span>
            </div>
            <Separator className='my-2 w-52'/>
            <div className="w-full p-4 bg-[var(--teal-3)] rounded-sm">
              <div className='text-justify text-sm text-popover-foreground'>{paperDetails.summary.discussion}</div>
            </div>
          </div>
        </div>
      </div>
      <Pagination>
        <PaginationContent className="flex flex-row gap-5 pb-0 min-[601px]:pb-10 min-[601px]:max-[774px]:gap-5 min-[775px]:gap-10 px-5 max-w-[1000px]">
          <PaginationItem className='w-[40vw] flex flex-row justify-end min-[601px]:justify-start'>
            <PaginationLink
              aria-label="Go to previous page"
              size="default"
              onClick={() => {
                if (paperId !== '0') {
                  router.push(`/details?id=${parseInt(paperId as string) - 1}`)
                }
              }}
              className="flex flex-row items-center gap-1 p-3 h-fit"
              isDisabled={paperId === '0'}
            >
              <RxCaretLeft className="h-5 w-5 flex-shrink-0"/>
              {paperId !== '0' ?
                <div className='flex flex-row flex-grow gap-2 items-center'>
                  <div className='min-[775px]:text-lg'>{parseInt(paperId as string) - 1}</div>
                  <Separator orientation="vertical" className='ml-2 h-5 hidden min-[601px]:block'/>
                  <div className='text-[10px] leading-[12px] min-[775px]:text-xs text-pretty break-all hidden min-[601px]:block'>{data[parseInt(paperId as string)-1]?.title}</div>
                </div>
              : null}
            </PaginationLink>
          </PaginationItem>
          <PaginationItem className='w-[40vw] flex flex-row justify-start min-[601px]:justify-end'>
            <PaginationLink
              aria-label="Go to next page"
              size="default"
              onClick={() => {
                if (paperId !== (data?.length-1).toString()) {
                  router.push(`/details?id=${parseInt(paperId as string) + 1}`)
                }
              }}
              className="flex flex-row items-center gap-1 p-3 h-fit"
              isDisabled={paperId === (data?.length-1).toString()}
            >
              {paperId !== (data?.length-1).toString() ? <div className='flex flex-row flex-grow gap-2 items-center'>
                <div className='text-[10px] leading-[12px] min-[775px]:text-xs text-pretty break-all hidden min-[601px]:block'>{data[parseInt(paperId as string)+1]?.title}</div>
                  <Separator orientation="vertical" className='ml-2 h-5 hidden min-[601px]:block'/>
                  <div className='min-[775px]:text-lg'>{parseInt(paperId as string) + 1}</div>
                </div>
              : null}
              <RxCaretRight className="h-5 w-5 flex-shrink-0"/>
            </PaginationLink>
          </PaginationItem>
        </PaginationContent>
      </Pagination>
    </div>
  )
}
