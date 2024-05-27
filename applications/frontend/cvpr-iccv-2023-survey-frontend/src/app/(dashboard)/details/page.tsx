'use client'
import { cache, useRef, useState, type RefObject, useEffect } from 'react'
import { toast } from 'sonner'
import { FileIcon, GlobeIcon } from '@radix-ui/react-icons'
import {Heading, Text, Flex, Link, DataList, Box, Separator, Badge, Button, Code} from '@radix-ui/themes'
import { useSearchParams, useRouter } from 'next/navigation'

import { PaperDetails, FetchResult } from '../../../libs/types'
import { getPaperDetails } from '../../../libs/fetch'

const handleFetchResult = (result: FetchResult<PaperDetails>, errorMessage: string): PaperDetails | null => {
  if (result.error) {
    toast.error(errorMessage);
    return null;
  }
  return result.data || null;
};

export const loadPaperDetails = cache(async (conference: string, id: string) => {
  const result = await getPaperDetails(conference, id);
  return handleFetchResult(result, 'Failed to fetch paper details');
})

export default function PaperInfo() {
  const router = useRouter()
  const searchParams = useSearchParams()

  const [paperDetails, setPaperDetails] = useState<PaperDetails | null>(null)
  const [isLoading, setIsLoading] = useState(true)

  const conference= searchParams.get('conference')
  const paperId= searchParams.get('id')

  useEffect(() => {
    if (!conference || !paperId) {
      toast.error('Invalid URL');
      router.push('/list')
    }
    else {
      const fetchPaperDetails = async () => {
        const paperDetails = await loadPaperDetails(conference, paperId)
        if (!paperDetails) {
          router.push('/list')
        }
        else {
          setPaperDetails(paperDetails)
          setIsLoading(false)
        }
      }
      fetchPaperDetails()
    }
  }, [])

  if (isLoading || !paperDetails) {
    return (
      <Flex justify="center" align="center" gap="3" width="100%">
        <Text>Loading...</Text>
      </Flex>
    )
  }

  return (
    <Flex direction="column" align="center" gap="7" width="100%">
      <Flex direction="column" align="center" gap="4" width="100%">
        <Heading size="6">{paperDetails.paperInfo.title}</Heading>
        <Separator my="2" size="4"/>
      </Flex>
      <Flex direction="column" gap="9" align="center" width="80%">
        <Flex direction="column" align="start" gap="4">
          <Flex direction="column" align="center" gap="0">
            <Heading size="5">基本情報</Heading>
            <Separator my="2" size="3"/>
          </Flex>
          <DataList.Root size="2">
            <DataList.Item>
              <DataList.Label minWidth="88px">ID</DataList.Label>
              <DataList.Value>{paperId}</DataList.Value>
            </DataList.Item>
            <DataList.Item>
              <DataList.Label minWidth="88px">Authors</DataList.Label>
              <DataList.Value>{paperDetails.paperInfo.authors.join(", ")}</DataList.Value>
            </DataList.Item>
            <DataList.Item>
              <DataList.Label minWidth="88px">Abstract</DataList.Label>
              <DataList.Value>
                {paperDetails.paperInfo.abstract}
              </DataList.Value>
            </DataList.Item>
            <DataList.Item>
              <DataList.Label minWidth="88px">Link</DataList.Label>
              <DataList.Value>
                <Flex direction="row" gap="4">
                  <Link target="_blank" href={paperDetails.paperInfo.cvfLink}>
                    <Button variant="soft" size="1">
                      <GlobeIcon /> CVF
                    </Button>
                  </Link>
                  <Link target="_blank" href={paperDetails.paperInfo.pdfLink}>
                    <Button variant="soft" size="1" color="tomato">
                      <FileIcon /> PDF
                    </Button>
                  </Link>
                </Flex>
              </DataList.Value>
            </DataList.Item>
          </DataList.Root>
        </Flex>
        <Flex direction="column" align="start" gap="4">
          <Flex direction="column" align="center" gap="0">
            <Heading size="5">どんなもの？</Heading>
            <Separator my="2" size="3"/>
          </Flex>
          <Text size="3" wrap="pretty">{paperDetails.summary.outline}</Text>
        </Flex>
        <Flex direction="column" align="start" gap="4">
          <Flex direction="column" align="center" gap="0">
            <Heading size="5">先行研究と比べてどこがすごい？</Heading>
            <Separator my="2" size="3"/>
          </Flex>
          <Text size="3" wrap="pretty">{paperDetails.summary.contribution}</Text>
        </Flex>
        <Flex direction="column" align="start" gap="4">
          <Flex direction="column" align="center" gap="0">
            <Heading size="5">技術や手法のキモはどこ？</Heading>
            <Separator my="2" size="3"/>
          </Flex>
          <Text size="3" wrap="pretty">{paperDetails.summary.method}</Text>
        </Flex>
        <Flex direction="column" align="start" gap="4">
          <Flex direction="column" align="center" gap="0">
            <Heading size="5">どうやって有効だと検証した？</Heading>
            <Separator my="2" size="3"/>
          </Flex>
          <Text size="3" wrap="pretty">{paperDetails.summary.evaluation}</Text>
        </Flex>
        <Flex direction="column" align="start" gap="4">
          <Flex direction="column" align="center" gap="0">
            <Heading size="5">議論はある？</Heading>
            <Separator my="2" size="3"/>
          </Flex>
          <Text size="3" wrap="pretty">{paperDetails.summary.discussion}</Text>
        </Flex>
      </Flex>
    </Flex>
  )
}