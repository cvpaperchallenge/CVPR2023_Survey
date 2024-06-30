'use client'

import { useRouter, useSearchParams } from 'next/navigation'
import { useState, useEffect } from 'react'
import { RxFile, RxGlobe, RxCaretLeft, RxCaretRight } from 'react-icons/rx'
import { toast } from 'sonner'

import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb'
import { Button } from '@/components/ui/button'
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
} from '@/components/ui/pagination'
import { Separator } from '@/components/ui/separator'
import { Skeleton } from '@/components/ui/skeleton'

import { getPaperDetails, handleFetchResult } from '@/lib/fetch'
import { PaperDetails, PaperInfo } from '@/lib/types'

const loadPaperDetails = async (id: string) => {
  const result = await getPaperDetails(id)
  return handleFetchResult<PaperDetails>(
    result,
    'Failed to fetch paper details'
  )
}

export default function DetailedPage() {
  const router = useRouter()
  const searchParams = useSearchParams()

  const paperId = searchParams.get('id')

  const [paperDetails, setPaperDetails] = useState<PaperDetails | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  const cachedData = localStorage.getItem('paper')
  const data: PaperInfo[] = JSON.parse(cachedData as string) as PaperInfo[]

  useEffect(() => {
    if (!paperId) {
      toast.error('Invalid URL')
      router.push('/list')
    } else {
      const fetchPaperDetails = async () => {
        const fetchedPaperDetails = await loadPaperDetails(paperId)
        if (!fetchedPaperDetails) {
          router.push('/list')
        } else {
          setPaperDetails(fetchedPaperDetails)
          setIsLoading(false)
        }
      }
      void fetchPaperDetails()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paperId])

  if (isLoading || !paperDetails) {
    return (
      <div className="flex w-screen flex-col items-center gap-12 py-8">
        <div className="flex w-full flex-row justify-start px-10">
          <Skeleton className="h-5 w-full rounded-sm" />
        </div>
        <div className="flex w-[75vw] min-w-[350px] max-w-[800px] flex-col items-start gap-7">
          <Skeleton className="h-12 w-full rounded-sm" />
          <div className="flex w-full flex-col items-center gap-10">
            <div className="flex w-full flex-col items-start gap-4">
              <Skeleton className="h-8 w-1/4 rounded-sm" />
              <Skeleton className="h-96 w-full rounded-sm" />
            </div>
            <div className="flex w-full flex-col items-start gap-4">
              <Skeleton className="h-8 w-1/4 rounded-sm" />
              <Skeleton className="h-44 w-full rounded-sm" />
            </div>
            <div className="flex w-full flex-col items-start gap-4">
              <Skeleton className="h-8 w-1/4 rounded-sm" />
              <Skeleton className="h-44 w-full rounded-sm" />
            </div>
            <div className="flex w-full flex-col items-start gap-4">
              <Skeleton className="h-8 w-1/4 rounded-sm" />
              <Skeleton className="h-44 w-full rounded-sm" />
            </div>
            <div className="flex w-full flex-col items-start gap-4">
              <Skeleton className="h-8 w-1/4 rounded-sm" />
              <Skeleton className="h-44 w-full rounded-sm" />
            </div>
            <div className="flex w-full flex-col items-start gap-4">
              <Skeleton className="h-8 w-1/4 rounded-sm" />
              <Skeleton className="h-44 w-full rounded-sm" />
            </div>
          </div>
        </div>
        <div className="flex max-w-[1000px] flex-row gap-5 px-5 pb-0 min-[601px]:pb-10 min-[601px]:max-[774px]:gap-5 min-[775px]:gap-10">
          <Skeleton className="h-14 w-[40vw] rounded-sm" />
          <Skeleton className="h-14 w-[40vw] rounded-sm" />
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
              <BreadcrumbLink href="/list">Paper List</BreadcrumbLink>
            </BreadcrumbItem>
            <BreadcrumbSeparator />
            <BreadcrumbItem>
              <BreadcrumbPage>
                {paperId}. {paperDetails.paperInfo.title}
              </BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>
      </div>
      <div className="flex w-[75vw] min-w-[350px] max-w-[800px] flex-col items-start gap-7">
        <div className="flex w-full flex-row justify-center">
          <div className="text-xl font-extrabold text-foreground">
            {paperDetails.paperInfo.title || ''}
          </div>
        </div>
        <div className="flex w-full flex-col items-center gap-10">
          <div className="flex w-full flex-col items-start gap-0">
            <div>
              <span className="pr-2 text-xl font-semibold text-[var(--black-a10)] dark:text-[var(--white-a10)]">
                基本情報
              </span>
              <span className="text-sm font-normal text-muted-foreground">
                / Basic Information
              </span>
            </div>
            <Separator className="my-2 w-60" />
            <div className="flex w-full flex-col items-start gap-1 rounded-sm bg-[var(--teal-3)] p-4 min-[775px]:grid min-[775px]:grid-cols-6 min-[775px]:items-center min-[775px]:gap-4">
              <div className="flex min-w-20 flex-col items-start gap-2 text-[14px] font-medium text-popover-foreground">
                ID
              </div>
              <div className="col-span-5 mb-3 gap-2 pl-2 text-xs text-popover-foreground min-[775px]:mb-0 min-[775px]:pl-0">
                {paperId}
              </div>
              <div className="flex min-w-20 flex-col items-start gap-2 text-[14px] font-medium text-popover-foreground">
                Authors
              </div>
              <div className="col-span-5 mb-3 gap-2 pl-2 text-xs text-popover-foreground min-[775px]:mb-0 min-[775px]:pl-0">
                {paperDetails.paperInfo.authors.join(', ')}
              </div>
              <div className="flex min-w-20 flex-col items-start gap-2 text-[14px] font-medium text-popover-foreground">
                Abstract
              </div>
              <div className="col-span-5 mb-3 gap-2 pl-2 text-justify text-xs text-popover-foreground min-[775px]:mb-0 min-[775px]:pl-0">
                {paperDetails.abstract}
              </div>
              <div className="flex min-w-20 flex-col items-start gap-2 text-[14px] font-medium text-popover-foreground">
                Link
              </div>
              <div className="col-span-5 mb-3 flex flex-row items-start gap-2 min-[775px]:mb-0">
                <a
                  href={paperDetails.paperInfo.cvfLink}
                  rel="noreferrer"
                  target="_blank"
                >
                  <Button
                    className="
                    h-6
                    gap-1
                    border-[var(--jade-6)]
                    bg-[var(--jade-4)]
                    px-2
                    text-xs
                    text-[var(--jade-11)]
                    hover:bg-[var(--jade-5)]
                    hover:text-[var(--jade-12)]
                  "
                    variant="outline"
                  >
                    <RxGlobe /> CVF
                  </Button>
                </a>
                <a
                  href={paperDetails.paperInfo.pdfLink}
                  rel="noreferrer"
                  target="_blank"
                >
                  <Button
                    className="
                      h-6
                      gap-1
                      border-[var(--red-6)]
                      bg-[var(--red-3)]
                      px-2
                      text-xs
                      text-[var(--red-11)]
                      hover:bg-[var(--red-4)]
                      hover:text-[var(--red-12)]
                  "
                    variant="outline"
                  >
                    <RxFile /> PDF
                  </Button>
                </a>
              </div>
              <div className="flex flex-col items-start gap-2 text-[14px] font-medium text-popover-foreground">
                Conference
              </div>
              <div className="col-span-5 gap-2 pl-2 text-xs text-popover-foreground min-[775px]:pl-0">
                {paperDetails.paperInfo.conference}
              </div>
            </div>
          </div>
          <div className="flex w-full flex-col items-start gap-0">
            <div>
              <span className="pr-2 text-xl font-semibold text-[var(--black-a10)] dark:text-[var(--white-a10)]">
                どんなもの？
              </span>
              <span className="text-sm font-normal text-muted-foreground">
                / Outline
              </span>
            </div>
            <Separator className="my-2 w-44" />
            <div className="w-full rounded-sm bg-[var(--teal-3)]  p-4">
              <div className="text-justify text-sm text-popover-foreground">
                {paperDetails.summary.outline}
              </div>
            </div>
          </div>
          <div className="flex w-full flex-col items-start gap-0">
            <div>
              <span className="pr-2 text-xl font-semibold text-[var(--black-a10)] dark:text-[var(--white-a10)]">
                先行研究と比べてどこがすごい？
              </span>
              <span className="text-sm font-normal text-muted-foreground">
                / Contribution
              </span>
            </div>
            <Separator className="my-2 w-[365px]" />
            <div className="w-full rounded-sm bg-[var(--teal-3)] p-4">
              <div className="text-justify text-sm text-popover-foreground">
                {paperDetails.summary.contribution}
              </div>
            </div>
          </div>
          <div className="flex w-full flex-col items-start gap-0">
            <div>
              <span className="pr-2 text-xl font-semibold text-[var(--black-a10)] dark:text-[var(--white-a10)]">
                技術や手法のキモはどこ？
              </span>
              <span className="text-sm font-normal text-muted-foreground">
                / Methods
              </span>
            </div>
            <Separator className="my-2 w-[295px]" />
            <div className="w-full rounded-sm bg-[var(--teal-3)] p-4">
              <div className="text-justify text-sm text-popover-foreground">
                {paperDetails.summary.method}
              </div>
            </div>
          </div>
          <div className="flex w-full flex-col items-start gap-0">
            <div>
              <span className="pr-2 text-xl font-semibold text-[var(--black-a10)] dark:text-[var(--white-a10)]">
                どうやって有効だと検証した？
              </span>
              <span className="text-sm font-normal text-muted-foreground">
                / Evaluation
              </span>
            </div>
            <Separator className="my-2 w-[330px]" />
            <div className="w-full rounded-sm bg-[var(--teal-3)] p-4">
              <div className="text-justify text-sm text-popover-foreground">
                {paperDetails.summary.evaluation}
              </div>
            </div>
          </div>
          <div className="flex w-full flex-col items-start gap-0">
            <div>
              <span className="pr-2 text-xl font-semibold text-[var(--black-a10)] dark:text-[var(--white-a10)]">
                議論はある？
              </span>
              <span className="text-sm font-normal text-muted-foreground">
                / Discussion
              </span>
            </div>
            <Separator className="my-2 w-52" />
            <div className="w-full rounded-sm bg-[var(--teal-3)] p-4">
              <div className="text-justify text-sm text-popover-foreground">
                {paperDetails.summary.discussion}
              </div>
            </div>
          </div>
        </div>
      </div>
      <Pagination>
        <PaginationContent className="flex max-w-[1000px] flex-row gap-5 px-5 pb-0 min-[601px]:pb-10 min-[601px]:max-[774px]:gap-5 min-[775px]:gap-10">
          <PaginationItem className="flex w-[40vw] flex-row justify-end min-[601px]:justify-start">
            <PaginationLink
              aria-label="Go to previous page"
              className="flex h-fit flex-row items-center gap-1 p-3"
              isDisabled={paperId === '0'}
              onClick={() => {
                if (paperId !== '0') {
                  router.push(`/details?id=${parseInt(paperId as string) - 1}`)
                }
              }}
              size="default"
            >
              <RxCaretLeft className="size-5 shrink-0" />
              {paperId !== '0' ? (
                <div className="flex grow flex-row items-center gap-2">
                  <div className="min-[775px]:text-lg">
                    {parseInt(paperId as string) - 1}
                  </div>
                  <Separator
                    className="ml-2 hidden h-5 min-[601px]:block"
                    orientation="vertical"
                  />
                  <div className="hidden text-pretty break-all text-[10px] leading-[12px] min-[601px]:block min-[775px]:text-xs">
                    {data[parseInt(paperId as string) - 1]?.title}
                  </div>
                </div>
              ) : null}
            </PaginationLink>
          </PaginationItem>
          <PaginationItem className="flex w-[40vw] flex-row justify-start min-[601px]:justify-end">
            <PaginationLink
              aria-label="Go to next page"
              className="flex h-fit flex-row items-center gap-1 p-3"
              isDisabled={paperId === (data?.length - 1).toString()}
              onClick={() => {
                if (paperId !== (data?.length - 1).toString()) {
                  router.push(`/details?id=${parseInt(paperId as string) + 1}`)
                }
              }}
              size="default"
            >
              {paperId !== (data?.length - 1).toString() ? (
                <div className="flex grow flex-row items-center gap-2">
                  <div className="hidden text-pretty break-all text-[10px] leading-[12px] min-[601px]:block min-[775px]:text-xs">
                    {data[parseInt(paperId as string) + 1]?.title}
                  </div>
                  <Separator
                    className="ml-2 hidden h-5 min-[601px]:block"
                    orientation="vertical"
                  />
                  <div className="min-[775px]:text-lg">
                    {parseInt(paperId as string) + 1}
                  </div>
                </div>
              ) : null}
              <RxCaretRight className="size-5 shrink-0" />
            </PaginationLink>
          </PaginationItem>
        </PaginationContent>
      </Pagination>
    </div>
  )
}
