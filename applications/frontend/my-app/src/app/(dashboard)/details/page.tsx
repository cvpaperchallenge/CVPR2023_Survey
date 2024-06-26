'use client'

import {  useState, useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'

import { toast } from "sonner"
import { Skeleton } from '@/components/ui/skeleton'
import { Separator } from "@/components/ui/separator"

import { PaperDetails } from '@/lib/types'
import { getPaperDetails, handleFetchResult } from '@/lib/fetch'
import { useSimplePaperContext } from '@/lib/providers'
import { Button } from '@/components/ui/button'
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
} from "@/components/ui/pagination"

const loadPaperDetails = async (conference: string, id: string) => {
  const result = await getPaperDetails(conference, id);
  return handleFetchResult<PaperDetails>(result, 'Failed to fetch paper details');
}

export default function DetailedPage() {

  const router = useRouter()
  const searchParams = useSearchParams()

  const conference = searchParams.get('conference')
  const paperId = searchParams.get('id')

  const [paperDetails, setPaperDetails] = useState<PaperDetails | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  const { simplePaperStates } = useSimplePaperContext()


  useEffect(() => {
    if (!conference || !paperId) {
      toast.error('Invalid URL');
      router.push('/list')
    }
    else {
      const fetchPaperDetails = async () => {
        const renamedConference = conference.replace(/([A-Z]+)(\d+)/, (match, p1, p2) => {
          return p1.toLowerCase() + '-' + p2;
        });
        const paperDetails = await loadPaperDetails(renamedConference, paperId)
        if (!paperDetails) {
          router.push('/list')
        } else {
          setPaperDetails(paperDetails)
          setIsLoading(false)
        }
      }
      fetchPaperDetails()
    }
  }, [])

  if (isLoading || !paperDetails) {
    return (
      <div className="flex flex-col items-start gap-10 w-[75vw]">
        <Skeleton className='h-12 w-full rounded-sm'/>
        <div className='flex flex-col items-center gap-16 w-full'>
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
    )
  }

  return (
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
              <div className='flex flex-col items-start min-w-20 gap-2 text-[14px] text-popover-foreground'>ID</div>
              <div className='mb-3 min-[775px]:mb-0 pl-2 min-[775px]:pl-0 col-span-5 gap-2 text-xs text-popover-foreground'>{paperId}</div>
              <div className='flex flex-col items-start min-w-20 gap-2 text-[14px] text-popover-foreground'>Authors</div>
              <div className='mb-3 min-[775px]:mb-0 pl-2 min-[775px]:pl-0 col-span-5 gap-2 text-xs text-popover-foreground'>{paperDetails.paperInfo.authors.join(", ")}</div>
              <div className='flex flex-col items-start min-w-20 gap-2 text-[14px] text-popover-foreground'>Abstract</div>
              <div className='mb-3 min-[775px]:mb-0 pl-2 min-[775px]:pl-0 col-span-5 gap-2 text-xs text-justify text-popover-foreground'>{paperDetails.abstract}</div>
              <div className='flex flex-col items-start min-w-20 gap-2 text-[14px] text-popover-foreground'>Link</div>
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
            <div className='flex flex-col items-start gap-2 text-[14px] text-popover-foreground'>Conference</div>
              <div className='pl-2 min-[775px]:pl-0 col-span-5 gap-2 text-xs text-popover-foreground'>{conference}</div>
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
      <div>
        <Pagination className="flex flex-row gap-2 px-5">
          <PaginationContent>
            <PaginationItem>
              <PaginationLink
                aria-label="Go to previous page"
                size="default"
                onClick={() => router.push(`/details?conference=${conference}&id=${parseInt(paperId as string) - 1}`)}
                className="flex flex-row items-center gap-1 p-3 w-[40vw] h-fit"
                isDisabled={simplePaperStates.previousPaperState === null}
              >
                <RxCaretLeft className="h-5 w-5" />
                <div className='text-[10px] leading-[12px] text-pretty'>{simplePaperStates.previousPaperState?.paperTitle}</div>
              </PaginationLink>
            </PaginationItem>
            <PaginationItem>
              <PaginationLink
                aria-label="Go to next page"
                size="default"
                onClick={() => router.push(`/details?conference=${conference}&id=${parseInt(paperId as string) + 1}`)}
                className="flex flex-row items-center gap-1 p-3 w-[40vw] h-fit"
                isDisabled={simplePaperStates.nextPaperState === null}
              >
                <div className='text-[10px] leading-[12px] text-pretty'>{simplePaperStates.nextPaperState?.paperTitle}</div>
                <RxCaretRight className="h-5 w-5"/>
              </PaginationLink>
            </PaginationItem>
          </PaginationContent>
        </Pagination>
      </div>
    </div>
  )
}
